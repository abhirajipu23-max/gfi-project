import re
import os
import time
import requests
from urllib.parse import urlparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# --- Supabase Config ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BATCH_SIZE         = 20
API_LIMIT          = 100   # jobs per API page (Oracle max)
MAX_DETAIL_WORKERS = 8     # parallel threads for Phase 2 detail fetching
DELAY_SEC          = 0.3   # polite delay between listing-page API calls
API_TIMEOUT        = 60    # seconds per request
API_RETRIES        = 3     # attempts before giving up
RETRY_BACKOFF      = [2, 4, 8]  # seconds to wait between retries

# ---------------------------------------------------------------------------
# Sustainability keyword filter (same list as sister scrapers)
# ---------------------------------------------------------------------------
SUSTAINABILITY_KEYWORDS = [
    "Sustainability", "Sustainable", "ESG", "Environmental Social Governance",
    "CSR", "Corporate Social Responsibility", "Climate", "Climate Change", "Carbon",
    "Decarbonization", "Decarbonisation", "GHG", "Greenhouse", "Emissions", "Net Zero",
    "Mitigation", "Resilience", "Environmental", "Environment", "Ecology", "Ecologist",
    "Biodiversity", "Conservation", "Habitat", "Restoration", "EHS", "Environment Health Safety",
    "HSE", "Health Safety Environment", "HSSE", "Health Safety Security Environment",
    "Safety", "Occupational", "Hygiene", "Industrial Hygiene", "Process Safety",
    "Water", "Wastewater", "Stormwater", "Hydrology", "Hydrogeology", "Groundwater",
    "Renewable", "Solar", "Wind", "Battery", "BESS", "Battery Energy Storage System",
    "Energy Transition", "Waste", "Recycling", "Circular", "Circular Economy", "Reuse", "Landfill",
    "Compliance", "Regulatory", "Disclosure", "Governance", "GRI", "Global Reporting Initiative",
    "SASB", "Sustainability Accounting Standards Board", "TCFD",
    "Task Force on Climate related Financial Disclosures", "CDP", "Carbon Disclosure Project",
    "Remediation", "Permitting", "Contaminated", "Soil", "Environmental Impact Assessment",
    "Due Diligence", "Hazard", "Hazardous", "GIS", "Clean Energy", "Energy Storage",
]

_PATTERNS = [re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in SUSTAINABILITY_KEYWORDS]


def is_sustainability_title(title: str) -> bool:
    return any(p.search(str(title)) for p in _PATTERNS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def strip_html(html: str) -> str:
    """Convert HTML markup to plain text."""
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)


def normalize_date(date_str: str):
    """Convert ISO date string (YYYY-MM-DD) or return None."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def extract_site_info(jobs_page_url: str):
    """
    Extract the Oracle Cloud base API URL and site_name from a jobs-page URL.

    Supported URL patterns:
      1. .../sites/<SiteName>/jobs   (named site + /jobs suffix)
      2. .../sites/<SiteName>        (named site, no /jobs suffix)

    Handles Oracle Cloud subdomain variants:
      - emit.fa.ca3.oraclecloud.com
      - ecyq.fa.em2.oraclecloud.com
      - fa-eups-saasfaprod1.fa.ocs.oraclecloud.com
      - eofh.fa.em2.oraclecloud.com
      - ehpv.fa.em2.oraclecloud.com
      - hcfa.fa.us2.oraclecloud.com

    Returns: (api_base, site_name) or (None, None) on parse failure.
    """
    try:
        parsed = urlparse(jobs_page_url)
        parts = [p for p in parsed.path.split("/") if p]  # drop empty segments
        if "sites" not in parts:
            return None, None
        idx = parts.index("sites")
        if idx + 1 >= len(parts):
            return None, None
        site_name = parts[idx + 1]   # e.g. "CX_2001" or "ULSolutionsCareers"
        api_base  = f"{parsed.scheme}://{parsed.netloc}"
        return api_base, site_name
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Oracle Cloud REST API helpers
# ---------------------------------------------------------------------------
HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Per-host seeded sessions (populated lazily by seed_session)
_SEEDED_SESSIONS: dict = {}


def seed_session(api_base: str, site_name: str) -> bool:
    """
    Use a headless Chromium browser to load the Oracle Cloud careers page,
    click through any Splash Page, harvest cookies + headers, and return success.
    This bypasses Akamai / WAF and sets up a warm requests.Session.
    """
    host = urlparse(api_base).netloc
    if host in _SEEDED_SESSIONS:
        return True

    print(f"    [Playwright] Seeding session for {host} ...")
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=HEADERS["User-Agent"],
                viewport={"width": 1280, "height": 800},
            )
            page = ctx.new_page()

            # Navigate to the base UI (no /jobs suffix to avoid 404 on splash-only sites)
            ui_url = f"{api_base}/hcmUI/CandidateExperience/en/sites/{site_name}"

            response = page.goto(ui_url, timeout=30000, wait_until="domcontentloaded")
            if response and response.status >= 500:
                print(f"    [Playwright] Received HTTP {response.status}")
                browser.close()
                return False

            page.wait_for_timeout(3000)

            # Splash-page handling: simulate a real user navigating into the full job list
            _clicked = False
            try:
                for pattern in [
                    r"All\s+Jobs",
                    r"New\s+Jobs",
                    r"Search\s+Jobs",
                ]:
                    lnk = page.locator("a, span, div", has_text=re.compile(pattern, re.IGNORECASE))
                    if lnk.count() > 0:
                        lnk.first.scroll_into_view_if_needed()
                        lnk.first.click(timeout=5000)
                        page.wait_for_timeout(3000)
                        _clicked = True
                        break

                # Last-resort: click first anchor with a parenthesised count e.g. "(682)"
                if not _clicked:
                    fallback = page.locator("a", has_text=re.compile(r"\(\d+\)"))
                    if fallback.count() > 0:
                        fallback.first.scroll_into_view_if_needed()
                        fallback.first.click(timeout=5000)
                        page.wait_for_timeout(3000)
            except Exception:
                pass

            # Transfer browser cookies -> requests session
            for ck in ctx.cookies():
                session.cookies.set(
                    ck["name"], ck["value"],
                    domain=ck.get("domain", "").lstrip(".")
                )

            browser.close()

        if not session.cookies:
            print(f"    [Playwright] Warning: Flow completed but 0 cookies harvested.")
            return False

        print(f"    [Playwright] Session seeded ({len(session.cookies)} cookies).")
        _SEEDED_SESSIONS[host] = session
        return True

    except Exception as e:
        print(f"    [Playwright] Seed failed for {host}: {e}")
        return False


def _get_session(api_base: str) -> requests.Session:
    """Return an already-seeded session or a plain one."""
    host = urlparse(api_base).netloc
    if host in _SEEDED_SESSIONS:
        return _SEEDED_SESSIONS[host]
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _api_get(url: str, api_base: str, site_name: str) -> dict:
    """
    GET the URL with retry logic.
    - Attempts 1 & 2: plain requests (no cookies) — works for most Oracle Cloud hosts.
    - Attempt 3: inject Playwright-seeded cookies as fallback for WAF-blocked hosts.
    """
    last_err = None

    # A clean session (no cookies) for the first attempts
    plain_session = requests.Session()
    plain_session.headers.update(HEADERS)

    for attempt in range(1, API_RETRIES + 1):
        # Last attempt: switch to pre-warmed Playwright session
        if attempt == API_RETRIES:
            session = _get_session(api_base)  # has cookies if Playwright ran
        else:
            session = plain_session

        try:
            resp = session.get(url, timeout=API_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt < API_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                print(f"    [retry {attempt}/{API_RETRIES}] {e} — waiting {wait}s")
                time.sleep(wait)

    print(f"    [API error] All {API_RETRIES} attempts failed: {last_err}")
    return {}


def fetch_job_list(api_base: str, site_number: str, site_name: str, offset: int = 0) -> dict:
    """Call the Oracle Cloud job listing API with retry."""
    url = (
        f"{api_base}/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
        f"?onlyData=true"
        f"&expand=requisitionList.workLocation,requisitionList.secondaryLocations"
        f"&finder=findReqs;siteNumber={site_number},"
        f"facetsList=LOCATIONS%3BWORK_LOCATIONS%3BWORKPLACE_TYPES%3BTITLES"
        f"%3BCATEGORIES%3BORGANIZATIONS%3BPOSTING_DATES,"
        f"limit={API_LIMIT},offset={offset},sortBy=POSTING_DATES_DESC"
    )
    return _api_get(url, api_base, site_name)


def fetch_job_detail(api_base: str, job_id: str, site_number: str, site_name: str) -> dict:
    """Call the Oracle Cloud job detail API with retry."""
    url = (
        f"{api_base}/hcmRestApi/resources/latest/recruitingCEJobRequisitionDetails"
        f'?expand=all&onlyData=true'
        f'&finder=ById;Id="{job_id}",siteNumber={site_number}'
    )
    return _api_get(url, api_base, site_name)


def build_job_page_url(api_base: str, site_name: str, job_id: str) -> str:
    """Construct the public-facing job detail URL."""
    return (
        f"{api_base}/hcmUI/CandidateExperience/en/sites/{site_name}/job/{job_id}"
    )


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------
def load_target_urls_from_supabase():
    """Return list of Oraclecloud companies from companies_sustain."""
    targets = []
    try:
        response = (
            supabase.table("companies_sustain")
            .select("id, company_name, jobs_page_url, ats_detected")
            .execute()
        )
        for row in response.data:
            ats = (row.get("ats_detected") or "").lower()
            # Match "oraclecloud" (not "oracle hcm" which has its own scraper)
            if "oraclecloud" not in ats or "oracle hcm" in ats:
                continue
            url = (row.get("jobs_page_url") or "").strip()
            company = (row.get("company_name") or "Unknown").strip()
            company_id = row.get("id")
            if url:
                targets.append({
                    "company": company,
                    "url": url,
                    "company_sustain_id": company_id,
                })
    except Exception as e:
        print(f"Error loading from Supabase: {e}")
    return targets


def load_existing_job_urls() -> set:
    """Load all job URLs already stored in jobs_sustain to avoid duplicates."""
    urls = set()
    try:
        response = supabase.table("jobs_sustain").select("job_url").execute()
        for row in response.data:
            if row.get("job_url"):
                urls.add(row["job_url"])
        print(f"Loaded {len(urls)} existing job URLs from Supabase.")
    except Exception as e:
        print(f"Error loading existing jobs: {e}")
    return urls


def insert_jobs_to_supabase(jobs: list):
    if not jobs:
        return
    try:
        supabase.table("jobs_sustain").upsert(jobs, on_conflict="job_url").execute()
        print(f"  -> Inserted/updated {len(jobs)} jobs into jobs_sustain.")
    except Exception as e:
        print(f"  -> Error inserting jobs: {e}")


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------
def scrape_company(target: dict, existing_job_urls: set) -> list:
    """
    Scrape all sustainability jobs for one Oraclecloud company.
    Returns a list of job records ready for Supabase upsert.
    """
    company_name       = target["company"]
    jobs_page_url      = target["url"]
    company_sustain_id = target["company_sustain_id"]

    print(f"\nProcessing: {company_name}  ({jobs_page_url})")

    api_base, site_name = extract_site_info(jobs_page_url)
    if not api_base or not site_name:
        print(f"  -> Cannot parse Oracle Cloud URL, skipping.")
        return []

    # Seed session via Playwright to bypass WAF / splash pages
    is_alive = seed_session(api_base, site_name)

    if not is_alive:
        # Try plain requests as a lightweight fallback before giving up
        print("  -> Playwright seed failed. Attempting plain requests fallback...")
        test_url = (
            f"{api_base}/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
            f"?onlyData=true&finder=findReqs;siteNumber={site_name},limit=1,offset=0"
        )
        try:
            r = requests.get(test_url, headers=HEADERS, timeout=API_TIMEOUT)
            if r.status_code == 200:
                print("  -> Plain requests fallback succeeded.")
                is_alive = True
                # Cache a plain session so _get_session() returns it
                s = requests.Session()
                s.headers.update(HEADERS)
                host = urlparse(api_base).netloc
                _SEEDED_SESSIONS[host] = s
        except Exception as e:
            print(f"  -> Plain requests fallback also failed: {e}")

    if not is_alive:
        print("  -> Oracle Cloud endpoint is completely unresponsive. Skipping.")
        return []

    # ------------------------------------------------------------------
    # We need the siteNumber (e.g. CX_2001) which isn't always in the UI URL.
    # Strategy: fetch the first page with site_name as placeholder, then
    # grab the real siteNumber from the response metadata.
    # ------------------------------------------------------------------
    site_number = site_name   # initial guess; corrected from first API response

    # Phase 1 – discover matching jobs via listing API
    discovered = []   # list of lightweight job dicts
    seen_urls  = set()
    offset     = 0

    while True:
        data = fetch_job_list(api_base, site_number, site_name, offset)
        if not data:
            break

        items = data.get("items", [])
        if not items:
            break

        top_level = items[0]

        # Correct site_number using value returned by server (first page only)
        if offset == 0:
            server_site = top_level.get("SiteNumber")
            if server_site:
                site_number = server_site

        req_list    = top_level.get("requisitionList", [])
        total_count = top_level.get("TotalJobsCount", 0)

        if not req_list:
            break

        for req in req_list:
            title = clean(req.get("Title", ""))
            if not title or not is_sustainability_title(title):
                continue

            # India-specific filter: only IN country jobs
            country = (req.get("PrimaryLocationCountry") or "").upper()
            if country and country != "IN":
                continue

            job_id  = str(req.get("Id", ""))
            job_url = build_job_page_url(api_base, site_name, job_id)

            if job_url in seen_urls or job_url in existing_job_urls:
                continue
            seen_urls.add(job_url)

            discovered.append({
                "job_id":              job_id,
                "job_url":             job_url,
                "title":               title,
                "location":            clean(req.get("PrimaryLocation") or ""),
                "workplace_type":      clean(req.get("WorkplaceType") or ""),
                "posted_date":         normalize_date(req.get("PostedDate")),
                "short_description":   clean(req.get("ShortDescriptionStr") or ""),
                "department":          clean(req.get("Department") or ""),
                "organization":        clean(req.get("Organization") or ""),
                "job_type":            clean(req.get("JobType") or ""),
                "job_schedule":        clean(req.get("JobSchedule") or ""),
                "contract_type":       clean(req.get("ContractType") or ""),
            })

        # Pagination
        offset += API_LIMIT
        if offset >= total_count:
            break

        time.sleep(DELAY_SEC)

    print(f"  -> Phase 1: Found {len(discovered)} sustainability jobs (India).")

    if not discovered:
        return []

    # Phase 2 – fetch full description in parallel
    def _fetch_detail(job: dict) -> dict:
        """Worker: enrich one job with details from the detail API."""
        description  = job["short_description"]
        department   = job["department"]
        organization = job["organization"]
        job_type     = job["job_type"]
        location     = job["location"]

        detail_data = fetch_job_detail(api_base, job["job_id"], site_number, site_name)
        if detail_data:
            detail_items = detail_data.get("items", [])
            if detail_items:
                d = detail_items[0]
                ext_desc = d.get("ExternalDescriptionStr") or d.get("ShortDescriptionStr") or ""
                if ext_desc:
                    description = strip_html(ext_desc) or description
                department   = clean(d.get("Department")    or "") or department
                organization = clean(d.get("Organization")  or "") or organization
                job_type     = clean(d.get("JobType")       or "") or job_type
                location     = clean(d.get("PrimaryLocation") or "") or location

        dept_value = organization or department or None
        return {
            "company_sustain_id":   company_sustain_id,
            "title":                job["title"],
            "job_url":              job["job_url"],
            "location":             location or None,
            "department":           dept_value,
            "job_type":             job_type or job["job_schedule"] or job["contract_type"] or None,
            "published_date":       job["posted_date"],
            "original_description": description or None,
            "is_active":            True,
        }

    records = []
    total   = len(discovered)
    with ThreadPoolExecutor(max_workers=MAX_DETAIL_WORKERS) as pool:
        future_to_job = {pool.submit(_fetch_detail, j): j for j in discovered}
        for done_idx, future in enumerate(as_completed(future_to_job), 1):
            job = future_to_job[future]
            try:
                record = future.result()
                records.append(record)
                print(f"  [{done_idx}/{total}] Done: {job['title']}")
            except Exception as exc:
                print(f"  [{done_idx}/{total}] Failed: {job['title']} — {exc}")

    return records


def run():
    companies = load_target_urls_from_supabase()
    if not companies:
        print("No Oraclecloud companies found in companies_sustain table.")
        return

    print(f"Found {len(companies)} Oraclecloud companies to scrape.")

    existing_job_urls = load_existing_job_urls()

    batch = []
    for target in companies:
        records = scrape_company(target, existing_job_urls)
        batch.extend(records)

        if len(batch) >= BATCH_SIZE:
            insert_jobs_to_supabase(batch)
            batch = []

    if batch:
        insert_jobs_to_supabase(batch)

    print("\nDone! All Oraclecloud jobs processed.")


if __name__ == "__main__":
    run()

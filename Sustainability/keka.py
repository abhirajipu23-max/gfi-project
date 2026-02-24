from playwright.sync_api import sync_playwright, TimeoutError as SyncTimeoutError
import re
import os
import time
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
# --- Supabase Config ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BATCH_SIZE = 20

JOB_CARD_SELECTOR = (
    "a.kh-card, "
    "a.kh-job-card, "
    "a.card.job-card, "
    "a[href*='jobdetails']"
)

SUSTAINABILITY_KEYWORDS = [
    "Sustainability","Sustainable","ESG","Environmental Social Governance",
    "CSR","Corporate Social Responsibility","Climate","Climate Change","Carbon",
    "Decarbonization","Decarbonisation","GHG","Greenhouse","Emissions","Net Zero",
    "Mitigation","Resilience","Environmental","Environment","Ecology","Ecologist",
    "Biodiversity","Conservation","Habitat","Restoration","EHS","Environment Health Safety",
    "HSE","Health Safety Environment","HSSE","Health Safety Security Environment",
    "Safety","Occupational","Hygiene","Industrial Hygiene","Process Safety",
    "Water","Wastewater","Stormwater","Hydrology","Hydrogeology","Groundwater",
    "Renewable","Solar","Wind","Battery","BESS","Battery Energy Storage System",
    "Energy Transition","Waste","Recycling","Circular","Circular Economy","Reuse","Landfill",
    "Compliance","Regulatory","Disclosure","Governance","GRI","Global Reporting Initiative",
    "SASB","Sustainability Accounting Standards Board","TCFD",
    "Task Force on Climate related Financial Disclosures","CDP","Carbon Disclosure Project",
    "Remediation","Permitting","Contaminated","Soil","Environmental Impact Assessment",
    "Due Diligence","Hazard","Hazardous","GIS","Clean Energy","Energy Storage"
]

_PATTERNS = [re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in SUSTAINABILITY_KEYWORDS]

def is_sustainability_title(title):
    return any(p.search(title) for p in _PATTERNS)

def load_existing_job_urls():
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

def load_target_urls_from_supabase():
    targets = []
    try:
        response = (
            supabase.table("companies_sustain")
            .select("id, company_name, jobs_page_url, ats_detected")
            .execute()
        )
        for row in response.data:
            ats = row.get("ats_detected", "") or ""
            if "keka" not in ats.lower():
                continue
            url = (row.get("jobs_page_url") or "").strip()
            company = (row.get("company_name") or "Unknown").strip()
            company_id = row.get("id")
            if url:
                targets.append({"company": company, "url": url, "company_sustain_id": company_id})
    except Exception as e:
        print(f"Error loading from Supabase: {e}")
    return targets

def insert_jobs_to_supabase(jobs: list):
    if not jobs:
        return
    try:
        supabase.table("jobs_sustain").upsert(jobs, on_conflict="job_url").execute()
        print(f"  -> Inserted/updated {len(jobs)} jobs into jobs_sustain.")
    except Exception as e:
        print(f"  -> Error inserting jobs: {e}")

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""

def normalize_posted_date(text: str):
    if not text:
        return None

    text = text.lower().strip()
    today = datetime.now(timezone.utc)

    match = re.search(r"(\d+)\s+day[s]?\s+ago", text)
    if match:
        dt = today - timedelta(days=int(match.group(1)))
        return dt.strftime("%Y-%m-%d")

    if "yesterday" in text:
        dt = today - timedelta(days=1)
        return dt.strftime("%Y-%m-%d")

    if "today" in text or "just now" in text:
        return today.strftime("%Y-%m-%d")

    return None

def parse_experience(exp_text: str):
    if not exp_text:
        return None, None

    exp_text = exp_text.lower().strip()

    if "fresher" in exp_text or re.fullmatch(r"0\s*(years|yrs)?", exp_text):
        return 0, 0

    plus_match = re.search(r"(\d+)\s*\+", exp_text)
    if plus_match:
        return int(plus_match.group(1)), None

    range_match = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)", exp_text)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    single_match = re.search(r"(\d+)", exp_text)
    if single_match:
        val = int(single_match.group(1))
        return val, val

    return None, None

def wait_for_html_ready_sync(page, timeout=25000):
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except Exception:
        pass
    try:
        page.wait_for_function("() => document.readyState === 'complete'", timeout=timeout)
    except Exception:
        pass
    try:
        page.wait_for_selector(JOB_CARD_SELECTOR, timeout=timeout)
    except Exception:
        pass

def wait_for_html_ready_stage2(page, timeout=25000):
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except Exception:
        pass
    try:
        page.wait_for_function("() => document.readyState === 'complete'", timeout=timeout)
    except Exception:
        pass
    try:
        page.wait_for_selector(
            ".job-description-container, .job-description, quill-view-html, span.ki-user-tie, span.ki-location, span.ki-briefcase",
            timeout=timeout
        )
    except Exception:
        pass

def run():
    companies_to_scrape = load_target_urls_from_supabase()
    if not companies_to_scrape:
        print("No Keka companies found in companies_sustain table.")
        return

    print(f"Found {len(companies_to_scrape)} Keka domains to scrape.")

    discovered_jobs = []
    seen_urls = set()

    # NEW: Load already existing job URLs
    existing_job_urls = load_existing_job_urls()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context()
        page = context.new_page()

        print("\n--- PHASE 1: Discovering Jobs ---")
        for target in companies_to_scrape:
            company_name = target["company"]
            base_url = target["url"]
            company_sustain_id = target["company_sustain_id"]
            print(f"Scanning {company_name}: {base_url}")

            try:
                page.goto(base_url, timeout=60000, wait_until="domcontentloaded")
                wait_for_html_ready_sync(page, timeout=25000)
            except SyncTimeoutError:
                print(f"  -> Timeout reaching: {base_url}")
                continue
            except Exception as e:
                print(f"  -> Connection error: {e}")
                continue

            job_cards = page.query_selector_all(JOB_CARD_SELECTOR)
            if not job_cards:
                print(f"  -> No job cards found.")
                continue

            for card in job_cards:
                def safe_text(selector):
                    el = card.query_selector(selector)
                    return clean_text(el.inner_text()) if el else ""

                raw_url = card.get_attribute("href") or ""
                if raw_url and not raw_url.startswith("http"):
                    parsed_base = urlparse(base_url)
                    domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

                    match = re.search(r"jobdetails/([A-Za-z0-9\-]+)", raw_url)
                    if match:
                        job_id = match.group(1)
                        job_url = f"{domain}/careers/jobdetails/{job_id}"
                    else:
                        job_url = urljoin(base_url, raw_url)
                else:
                    job_url = raw_url

                if not job_url or job_url in seen_urls or job_url in existing_job_urls:
                    continue

                title = safe_text("h4.kh-job-title, h3.job-title")

                if not title or not is_sustainability_title(title):
                    continue

                seen_urls.add(job_url)

                posted_raw = safe_text("small, span.text-secondary")
                job_type = safe_text(".job-type, .type")

                location = ""
                try:
                    loc_element = card.query_selector("span.font-large")
                    if loc_element:
                        location = clean_text(loc_element.inner_text())
                except Exception:
                    pass

                discovered_jobs.append({
                    "company_name": company_name,
                    "company_sustain_id": company_sustain_id,
                    "title": title,
                    "url": job_url,
                    "location": location,
                    "job_type": job_type,
                    "posted": normalize_posted_date(posted_raw),
                })

        print(f"\nPhase 1 Complete: Found {len(discovered_jobs)} sustainability jobs.")

        print("\n--- PHASE 2: Fetching Job Details ---")
        batch = []

        for index, job in enumerate(discovered_jobs):
            job_url = job["url"]
            print(f"Scraping [{index + 1}/{len(discovered_jobs)}]: {job['title']} at {job['company_name']}")

            try:
                page.goto(job_url, timeout=60000, wait_until="domcontentloaded")
                wait_for_html_ready_stage2(page, timeout=25000)
                page.wait_for_timeout(800)

                def page_safe_text(selector):
                    try:
                        loc = page.locator(selector)
                        if loc.count() == 0:
                            return ""
                        return clean_text(loc.first.inner_text())
                    except Exception:
                        return ""

                experience_text = page_safe_text("span.ki-user-tie >> xpath=../span[2]")
                extracted_location = page_safe_text("span.ki-location >> xpath=../span[2]")
                enrich_job_type = page_safe_text("span.ki-briefcase >> xpath=../span[2]")

                description = ""
                try:
                    desc_selectors = [
                        ".job-description-container",
                        ".job-description",
                        "quill-view-html",
                        "quill-view",
                        ".ql-editor",
                        "div[class*='description']"
                    ]
                    for sel in desc_selectors:
                        container = page.locator(sel)
                        if container.count() > 0:
                            li_texts = container.first.locator("li").all_inner_texts()
                            if li_texts:
                                description = " | ".join(clean_text(t) for t in li_texts if t.strip())
                            else:
                                description = clean_text(container.first.inner_text())
                            if description:
                                break
                except Exception:
                    pass

                min_exp, max_exp = parse_experience(experience_text)

                final_location = job["location"] or extracted_location or None
                final_job_type = job["job_type"] or enrich_job_type or None

                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": job["title"],
                    "job_url": job_url,
                    "location": final_location,
                    "published_date": job["posted"],
                    "job_type": final_job_type,
                    "min_exp": min_exp,
                    "max_exp": max_exp,
                    "original_description": description or None,
                    "is_active": True,
                })

            except Exception as e:
                print(f"  -> Failed fetching details: {e}")
                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": job["title"],
                    "job_url": job_url,
                    "location": job["location"] or None,
                    "published_date": job["posted"],
                    "job_type": job["job_type"] or None,
                    "min_exp": None,
                    "max_exp": None,
                    "original_description": None,
                    "is_active": True,
                })

            if len(batch) >= BATCH_SIZE:
                insert_jobs_to_supabase(batch)
                batch = []

        if batch:
            insert_jobs_to_supabase(batch)

        browser.close()

    print("\nDone! All jobs inserted into jobs_sustain table.")

if __name__ == "__main__":
    start_time = time.time()
    run()
    print(f"Total Runtime: {time.time() - start_time:.2f}s")
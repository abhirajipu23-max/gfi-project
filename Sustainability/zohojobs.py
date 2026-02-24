from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import requests
import re
import os
import time
from urllib.parse import urljoin
from datetime import datetime, timezone
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
# --- Supabase Config ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BATCH_SIZE = 20

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
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
            if "zoho" not in ats.lower():
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

def format_date_iso(s):
    if not s:
        return None

    s = s.strip()
    s = re.sub(r"(?i)posted\s*on", "", s).strip()

    formats = (
        "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d",
        "%b %d, %Y", "%B %d, %Y", "%d-%b-%Y", "%d/%m/%Y",
    )

    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None

def parse_experience(exp_text):
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

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""

def dedupe_jobs(jobs):
    seen = {}
    for j in jobs:
        if j.get("url"):
            key = f"{j.get('url')}::{j.get('title','')}"
            if key not in seen:
                seen[key] = j
    return list(seen.values())

# --- ZOHO PARSING LOGICS ---

def _scrape_zoho_table(soup, url, company_name, company_sustain_id):
    jobs = []
    rows = soup.select("tr.jobDetailRow")
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        link = cols[0].find("a", class_="jobdetail")
        job_type = link.get_text(strip=True) if link else ""
        job_url_parsed = urljoin(url, link["href"]) if link and link.get("href") else ""
        title = cols[1].get_text(strip=True)
        posted = cols[2].get_text(strip=True)
        location = cols[3].get_text(strip=True)

        if is_sustainability_title(title):
            jobs.append({
                "company_name": company_name,
                "company_sustain_id": company_sustain_id,
                "title": title,
                "url": job_url_parsed,
                "job_type": job_type or None,
                "posted": format_date_iso(posted),
                "location": location or None,
            })
    return jobs

def _scrape_zoho_layout3_cards(soup, url, company_name, company_sustain_id):
    jobs = []
    cards = soup.select("div.cw-filter-joblist")
    if not cards:
        return []

    for card in cards:
        a = card.select_one("a.cw-3-title")
        if not a:
            continue

        title = a.get_text(strip=True)
        if not is_sustainability_title(title):
            continue

        job_url_parsed = urljoin(url, a.get("href", ""))

        location = ""
        loc_el = card.select_one("p.filter-subhead")
        if loc_el:
            exp = loc_el.select_one("span.search-work-experience")
            if exp:
                exp.extract()
            location = loc_el.get_text(" ", strip=True)

        job_type, posted = None, None
        type_el = card.select_one("span.cw-full-time")
        if type_el:
            job_type = type_el.get_text(strip=True) or None
        posted_el = card.select_one("span.search-date-opened")
        if posted_el:
            posted = format_date_iso(posted_el.get_text(strip=True))

        jobs.append({
            "company_name": company_name,
            "company_sustain_id": company_sustain_id,
            "title": title,
            "url": job_url_parsed,
            "job_type": job_type,
            "posted": posted,
            "location": location or None,
        })
    return jobs

def parse_job_details_from_soup(soup):
    data = {
        "title": "", "posted": "", "experience": "",
        "job_type": "", "raw_location": "", "description": "",
        "department": "", "industry": "",
    }

    for sel in [
        "div.tem3-titleBlock h1", "div.tem4-col2_hdr h2",
        "div.cw-jobheader-information h1", "h1", "h2",
    ]:
        el = soup.select_one(sel)
        if el:
            data["title"] = el.get_text(" ", strip=True).split("\n")[0].strip()
            break

    for sel in [
        "div.job-dec-block_new", "div.cw-jobdescription",
        "div.temp4-content", "span#spandesc", "div.tem3ContentBlock",
    ]:
        el = soup.select_one(sel)
        if el:
            data["description"] = el.get_text("\n", strip=True)
            break

    def apply_label_value(label, value):
        if not label or not value:
            return
        label = label.lower().strip()
        value = value.strip()

        if any(x in label for x in ["date opened", "posted", "date"]):
            if not data["posted"]: data["posted"] = value
        elif "job type" in label:
            if not data["job_type"]: data["job_type"] = value
        elif any(x in label for x in ["work experience", "required experience", "experience"]):
            if not data["experience"]: data["experience"] = value
        elif any(x in label for x in ["department name", "department", "function"]):
            if not data["department"]: data["department"] = value
        elif "industry" in label:
            if not data["industry"]: data["industry"] = value
        elif any(x in label for x in ["location", "city", "state", "province", "country", "region"]):
            if not data["raw_location"]:
                data["raw_location"] = value
            else:
                data["raw_location"] += f", {value}"

    summary_items = soup.select("ul.cw-summary-list li")
    for li in summary_items:
        spans = li.find_all("span")
        if len(spans) < 2:
            continue
        apply_label_value(spans[0].get_text(strip=True), spans[1].get_text(strip=True))

    tem3_blocks = soup.select("div.tem3ContentBlock div.tem3-TextCont")
    for block in tem3_blocks:
        label_el = block.find("span")
        value_el = block.select_one("div.tem3-RightSideText")
        if label_el and value_el:
            apply_label_value(label_el.get_text(" ", strip=True), value_el.get_text(" ", strip=True))

    return data if data["title"] else None

# --- MAIN RUNNER ---

def run():
    companies_to_scrape = load_target_urls_from_supabase()
    if not companies_to_scrape:
        print("No Zoho companies found in companies_sustain table.")
        return

    print(f"Found {len(companies_to_scrape)} Zoho domains to scrape.")
    discovered_jobs = []

    # NEW: preload existing Supabase jobs
    existing_job_urls = load_existing_job_urls()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(user_agent=USER_AGENT)

        print("\n--- PHASE 1: Discovering Jobs ---")
        for target in companies_to_scrape:
            company_name = target["company"]
            base_url = target["url"]
            company_sustain_id = target["company_sustain_id"]
            print(f"Scanning {company_name}: {base_url}")

            company_jobs = []

            # 1. Try requests + bs4
            try:
                r = requests.get(base_url, headers={"User-Agent": USER_AGENT}, timeout=15)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    company_jobs = _scrape_zoho_table(soup, base_url, company_name, company_sustain_id)
                    if not company_jobs:
                        company_jobs = _scrape_zoho_layout3_cards(soup, base_url, company_name, company_sustain_id)

                    if not company_jobs:
                        anchors = soup.select(
                            "a.cw-1-title, a.cw-3-title, a.zp_career_site_job_title, "
                            "a.jobdetail, a[href*='/jobs/'], a[href*='JobOpenings'], "
                            "a[href*='jobdetail'], a[href*='recruit']"
                        )
                        for a in anchors:
                            title = a.get_text(" ", strip=True)
                            href = a.get("href")
                            if not href or not title or "visit website" in title.lower() or "zoho recruit" in title.lower():
                                continue
                            if is_sustainability_title(title):
                                company_jobs.append({
                                    "company_name": company_name,
                                    "company_sustain_id": company_sustain_id,
                                    "title": title,
                                    "url": urljoin(base_url, href),
                                    "job_type": None, "posted": None, "location": None,
                                })
            except Exception:
                pass

            # 2. Playwright fallback
            if not company_jobs:
                try:
                    page.goto(base_url, timeout=60000, wait_until="domcontentloaded")
                    page.wait_for_timeout(2500)
                    html = page.content()
                    soup = BeautifulSoup(html, "html.parser")

                    company_jobs = _scrape_zoho_table(soup, base_url, company_name, company_sustain_id)
                    if not company_jobs:
                        company_jobs = _scrape_zoho_layout3_cards(soup, base_url, company_name, company_sustain_id)

                    if not company_jobs:
                        selectors = [
                            "a.cw-1-title", "a.cw-3-title", "a.zp_career_site_job_title",
                            "a.jobdetail", "a[href*='/jobs/']", "a[href*='JobOpenings']", "a[href*='jobdetail']"
                        ]
                        for sel in selectors:
                            loc = page.locator(sel)
                            for i in range(loc.count()):
                                a = loc.nth(i)
                                try:
                                    title = a.inner_text().strip()
                                    href = a.get_attribute("href") or ""
                                    if not href or not title or "zoho recruit" in title.lower() or "visit website" in title.lower():
                                        continue
                                    if is_sustainability_title(title):
                                        company_jobs.append({
                                            "company_name": company_name,
                                            "company_sustain_id": company_sustain_id,
                                            "title": title,
                                            "url": urljoin(base_url, href),
                                            "job_type": None, "posted": None, "location": None,
                                        })
                                except Exception:
                                    pass
                except Exception as e:
                    print(f"  -> Playwright fallback failed: {e}")

            company_jobs = dedupe_jobs(company_jobs)

            # NEW: skip jobs already in Supabase
            company_jobs = [
                j for j in company_jobs
                if j.get("url") and j["url"] not in existing_job_urls
            ]

            discovered_jobs.extend(company_jobs)
            if company_jobs:
                print(f"  -> Found {len(company_jobs)} sustainability jobs")

        print(f"\nPhase 1 Complete: Found {len(discovered_jobs)} total sustainability jobs.")

        print("\n--- PHASE 2: Fetching Job Details ---")
        batch = []

        for index, job in enumerate(discovered_jobs):
            job_url = job["url"]
            print(f"Scraping [{index + 1}/{len(discovered_jobs)}]: {job['title']} at {job['company_name']}")

            details = None

            # 1. Try requests + bs4
            try:
                r = requests.get(job_url, headers={"User-Agent": USER_AGENT}, timeout=20)
                if r.status_code == 200:
                    details = parse_job_details_from_soup(BeautifulSoup(r.text, "html.parser"))
            except Exception:
                pass

            # 2. Playwright fallback
            if not details:
                try:
                    page.goto(job_url, timeout=60000, wait_until="domcontentloaded")
                    page.wait_for_timeout(1500)
                    html = page.content()
                    details = parse_job_details_from_soup(BeautifulSoup(html, "html.parser"))
                except Exception as e:
                    print(f"  -> Playwright fallback failed: {e}")

            if not details:
                print(f"  -> Failed to parse details for {job_url}")
                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": job["title"],
                    "job_url": job_url,
                    "location": job.get("location"),
                    "department": None,
                    "published_date": job.get("posted"),
                    "job_type": job.get("job_type"),
                    "min_exp": None,
                    "max_exp": None,
                    "original_description": None,
                    "is_active": True,
                })
            else:
                min_exp, max_exp = parse_experience(details["experience"])
                posted_date = format_date_iso(details["posted"])

                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": details["title"] if details["title"] else job["title"],
                    "job_url": job_url,
                    "location": details["raw_location"] or job.get("location") or None,
                    "department": details["department"] or None,
                    "published_date": posted_date or job.get("posted"),
                    "job_type": details["job_type"] or job.get("job_type") or None,
                    "min_exp": min_exp,
                    "max_exp": max_exp,
                    "original_description": details["description"] or None,
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
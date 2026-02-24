import re
import os
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SEARCH_KEYWORD = "india"
MAX_WORKERS = 5
BATCH_SIZE = 20

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
    return any(p.search(str(title)) for p in _PATTERNS)

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
            if "workday" not in ats.lower():
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

def clean(text):
    return str(text).replace(",", " ").strip()

def posted_to_date(text):
    if not text:
        return None
    today = datetime.today().date()
    text = str(text).lower()

    if "yesterday" in text:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")

    m = re.search(r'(\d+)', text)
    if not m:
        return None

    return (today - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")

def scrape_detail(job):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def safe(sel):
                loc = page.locator(sel)
                return loc.first.inner_text() if loc.count() else ""

            page.goto(job["url"], timeout=60000)
            page.wait_for_selector('[data-automation-id="jobPostingDescription"]', timeout=60000)

            record = {
                "company_sustain_id": job["company_sustain_id"],
                "title": job["title"],
                "job_url": job["url"],
                "location": safe('[data-automation-id="locations"] dd') or None,
                "job_type": safe('[data-automation-id="time"] dd') or None,
                "published_date": posted_to_date(safe('[data-automation-id="postedOn"] dd')),
                "original_description": clean(safe("//div[@data-automation-id='jobPostingDescription']")) or None,
                "is_active": True,
            }

            browser.close()
            return record

    except Exception as e:
        print("Detail failed:", job["url"], e)
        return None

def run():
    companies_to_scrape = load_target_urls_from_supabase()
    if not companies_to_scrape:
        print("No Workday companies found in companies_sustain table.")
        return

    print(f"Found {len(companies_to_scrape)} Workday domains to scrape.")

    jobs = []
    seen = set()

    # NEW: preload existing Supabase jobs
    existing_job_urls = load_existing_job_urls()

    # -------- PHASE 1: Discover Jobs --------
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for target in companies_to_scrape:
            company = target["company"]
            base = target["url"]
            company_sustain_id = target["company_sustain_id"]

            print(f"Scanning: {company}")

            try:
                page.goto(base, timeout=60000,wait_until="domcontentloaded")
                # page.wait_for_load_state("networkidle")
            except Exception:
                continue

            try:
                s = page.get_by_placeholder("Search")
                s.fill(SEARCH_KEYWORD)
                s.press("Enter")
                page.wait_for_timeout(2000)
            except Exception:
                pass

            try:
                page.wait_for_selector('[data-automation-id="jobTitle"]', timeout=10000)
            except Exception:
                continue

            while True:
                cards = page.locator('//li[.//*[@data-automation-id="jobTitle"]]')

                for i in range(cards.count()):
                    c = cards.nth(i)
                    t = c.locator('[data-automation-id="jobTitle"]')
                    title = t.inner_text().strip()

                    if not is_sustainability_title(title):
                        continue

                    url = urljoin(base, t.get_attribute("href"))

                    if url in seen or url in existing_job_urls:
                        continue

                    seen.add(url)

                    jobs.append({
                        "company": company,
                        "company_sustain_id": company_sustain_id,
                        "title": clean(title),
                        "url": url
                    })

                nxt = page.locator('button[aria-label="next"]')
                if nxt.count() == 0:
                    break

                nxt.click()
                page.wait_for_timeout(1500)

        browser.close()

    print(f"\nPhase 1 Complete: {len(jobs)} sustainability jobs found.")

    # -------- PHASE 2: Parallel Detail Scraping --------
    final = []

    with ThreadPoolExecutor(MAX_WORKERS) as exe:
        futures = [exe.submit(scrape_detail, j) for j in jobs]

        for f in as_completed(futures):
            r = f.result()
            if r:
                final.append(r)

            # Insert in batches as results come in
            if len(final) >= BATCH_SIZE:
                insert_jobs_to_supabase(final)
                final = []

    # Insert any remaining
    if final:
        insert_jobs_to_supabase(final)

    print(f"\nDone! Total detailed jobs inserted: {len(jobs)}")

if __name__ == "__main__":
    run()
from playwright.sync_api import sync_playwright
import re
import os
import time
from urllib.parse import urljoin
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# --- Supabase Config ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SEARCH_KEYWORD = "india"
BATCH_SIZE = 20

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
            if "eightfold" not in ats.lower():
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
    return " ".join(str(text).split())

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

def run():
    companies_to_scrape = load_target_urls_from_supabase()
    if not companies_to_scrape:
        print("No Eightfold companies found in companies_sustain table.")
        return

    print(f"Found {len(companies_to_scrape)} Eightfold domains to scrape.")

    discovered_jobs = []
    seen_urls = set()

    # NEW: Load existing jobs from Supabase
    existing_job_urls = load_existing_job_urls()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()

        print("\n--- PHASE 1: Discovering Jobs ---")
        for target in companies_to_scrape:
            company_name = target["company"]
            base_url = target["url"]
            company_sustain_id = target["company_sustain_id"]
            print(f"Checking {company_name} at {base_url}")

            page = context.new_page()

            try:
                page.goto(base_url, timeout=60000)
                page.wait_for_load_state("networkidle")

                try:
                    search = page.locator('[data-testid="position-location-search-search"]')
                    search.wait_for(timeout=5000)
                    search.click()
                    search.fill(SEARCH_KEYWORD)
                    search.press("Enter")
                    page.wait_for_timeout(2500)
                except Exception:
                    print(f"  -> Search box not found or timeout. Proceeding without filter...")

                while True:
                    try:
                        page.wait_for_selector('[data-test-id="job-listing"]', timeout=10000)
                    except Exception:
                        break

                    job_cards = page.locator('[data-test-id="job-listing"]')
                    count = job_cards.count()

                    for i in range(count):
                        card = job_cards.nth(i)

                        try:
                            title = card.locator(".title-1aNJK").inner_text().strip()
                        except Exception:
                            continue

                        if not is_sustainability_title(title):
                            continue

                        link = card.locator("a").get_attribute("href")
                        job_url = urljoin(base_url, link)

                        if job_url in seen_urls or job_url in existing_job_urls:
                            continue

                        seen_urls.add(job_url)

                        location = card.locator(".fieldValue-3kEar").first.inner_text() if card.locator(".fieldValue-3kEar").count() else ""
                        posted_initial = card.locator(".subData-13Lm1").inner_text() if card.locator(".subData-13Lm1").count() else ""

                        discovered_jobs.append({
                            "company_name": company_name,
                            "company_sustain_id": company_sustain_id,
                            "title": title,
                            "url": job_url,
                            "location": location,
                            "posted_initial": posted_initial
                        })

                    next_btn = page.locator('button[aria-label="Next jobs"]')
                    if next_btn.count() == 0 or next_btn.get_attribute("aria-disabled") == "true":
                        break
                    next_btn.click()
                    page.wait_for_timeout(2000)

            except Exception as e:
                print(f"  -> Error on {company_name}: {e}")
            finally:
                page.close()

        print(f"\nPhase 1 Complete: Found {len(discovered_jobs)} sustainability jobs.")

        print("\n--- PHASE 2: Fetching Job Details ---")
        batch = []

        for index, job in enumerate(discovered_jobs):
            job_url = job["url"]
            print(f"Scraping [{index + 1}/{len(discovered_jobs)}]: {job['title']} at {job['company_name']}")

            page = context.new_page()

            try:
                page.goto(job_url, timeout=60000)
                page.wait_for_selector("#job-description-container", timeout=15000)

                try:
                    posted = page.locator("//div[text()='Date posted']/following-sibling::div").inner_text()
                except Exception:
                    posted = job["posted_initial"]

                try:
                    description = page.locator("#job-description-container").inner_text()
                except Exception:
                    description = ""

                match = re.search(r"Time Type:\s*([A-Za-z ]+)", description)
                job_type = match.group(1).strip() if match else None

                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": job["title"],
                    "job_url": job_url,
                    "location": job["location"] or None,
                    "published_date": posted_to_date(posted),
                    "job_type": job_type,
                    "original_description": clean(description) or None,
                    "is_active": True,
                })

            except Exception as e:
                print(f"  -> Skipping URL due to error: {e}")
                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": job["title"],
                    "job_url": job_url,
                    "location": job["location"] or None,
                    "published_date": posted_to_date(job["posted_initial"]),
                    "job_type": None,
                    "original_description": "FAILED_TO_LOAD",
                    "is_active": True,
                })

            finally:
                time.sleep(1)
                page.close()

            if len(batch) >= BATCH_SIZE:
                insert_jobs_to_supabase(batch)
                batch = []

        if batch:
            insert_jobs_to_supabase(batch)

        browser.close()

    print("\nDone! All jobs inserted into jobs_sustain table.")

if __name__ == "__main__":
    run()
from playwright.sync_api import sync_playwright
import re
import os
import time
from urllib.parse import urljoin
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
# --- Supabase Config ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
            if "smartrecruiters" not in ats.lower():
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
    return " ".join(str(text).split()) if text else ""

def run():
    companies_to_scrape = load_target_urls_from_supabase()
    if not companies_to_scrape:
        print("No SmartRecruiters companies found in companies_sustain table.")
        return

    print(f"Found {len(companies_to_scrape)} SmartRecruiters domains to scrape.")

    discovered_jobs = []
    seen_urls = set()

    # NEW: Load already existing job URLs
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
                page.goto(base_url, timeout=60000, wait_until="domcontentloaded")
                # page.wait_for_load_state("networkidle")

                try:
                    page.wait_for_selector('xpath=//div[contains(@class,"openings-body")]', timeout=10000)
                except Exception:
                    print("  -> No job list found or timeout.")
                    continue

                sections = page.locator('xpath=//section[contains(@class,"openings-section")]')

                for i in range(sections.count()):
                    section = sections.nth(i)
                    job_items = section.locator('xpath=.//li[contains(@class,"opening-job")]')

                    for j in range(job_items.count()):
                        job = job_items.nth(j)

                        try:
                            title_el = job.locator('xpath=.//h4[contains(@class,"job-title")]')
                            title = title_el.inner_text().strip() if title_el.count() else ""
                        except Exception:
                            continue

                        if not is_sustainability_title(title):
                            continue

                        try:
                            link_el = job.locator('xpath=.//a[contains(@class,"details")]')
                            link = link_el.get_attribute("href") if link_el.count() else ""
                            job_url = urljoin(base_url, link)
                        except Exception:
                            continue

                        if not job_url or job_url in seen_urls or job_url in existing_job_urls:
                            continue

                        seen_urls.add(job_url)

                        discovered_jobs.append({
                            "company_name": company_name,
                            "company_sustain_id": company_sustain_id,
                            "title": title,
                            "url": job_url,
                        })

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
                page.wait_for_load_state("domcontentloaded")

                # Extract Location
                location = None
                try:
                    loc_el = page.locator(".c-spl-job-location__place").first
                    if loc_el.count():
                        location = clean(loc_el.inner_text()) or None
                    else:
                        meta_loc = page.locator('li[itemprop="jobLocation"]').first
                        if meta_loc.count():
                            location = clean(meta_loc.inner_text().replace("Location", "")) or None
                except Exception:
                    pass

                # Extract Description
                description = None
                try:
                    desc_parts = []
                    for sec_id, sec_title in [
                        ("#st-companyDescription .wysiwyg", "### Company Description"),
                        ("#st-jobDescription .wysiwyg", "### Job Description"),
                        ("#st-qualifications .wysiwyg", "### Qualifications"),
                        ("#st-additionalInformation .wysiwyg", "### Additional Info"),
                    ]:
                        el = page.locator(sec_id)
                        if el.count():
                            text = clean(el.inner_text())
                            if text:
                                desc_parts.append(f"{sec_title}\n{text}")

                    if desc_parts:
                        description = "\n\n".join(desc_parts)
                    else:
                        el_gen = page.locator('[itemprop="description"]')
                        if el_gen.count():
                            description = clean(el_gen.inner_text()) or None
                except Exception:
                    pass

                # Extract Job Type & Department
                job_type = None
                department = None
                try:
                    type_el = page.locator('[itemprop="employmentType"]').first
                    if type_el.count():
                        job_type = clean(type_el.inner_text()) or None

                    dept_meta = page.locator('meta[itemprop="industry"]').first
                    if dept_meta.count():
                        department = dept_meta.get_attribute("content") or None
                except Exception:
                    pass

                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": job["title"],
                    "job_url": job_url,
                    "location": location,
                    "job_type": job_type,
                    "department": department,
                    "original_description": description,
                    "is_active": True,
                })

            except Exception as e:
                print(f"  -> Skipping URL due to error: {e}")
                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": job["title"],
                    "job_url": job_url,
                    "location": None,
                    "job_type": None,
                    "department": None,
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
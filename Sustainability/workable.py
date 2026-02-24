from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urljoin
from datetime import datetime, timedelta, timezone
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
            if "workable" not in ats.lower():
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
    if not text:
        return ""
    text = str(text).replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()

def safe_text(locator) -> str:
    try:
        return locator.first.inner_text().strip() if locator.count() > 0 else ""
    except Exception:
        return ""

def safe_attr(locator, attr) -> str:
    try:
        val = locator.first.get_attribute(attr) if locator.count() > 0 else None
        return val.strip() if val else ""
    except Exception:
        return ""

def normalize_job_type(text: str):
    if not text:
        return None
    t = clean_text(text).lower()
    mapping = {
        "full time": "Full Time", "full-time": "Full Time",
        "part time": "Part Time", "part-time": "Part Time",
        "contract": "Contract", "temporary": "Temporary",
        "internship": "Internship", "freelance": "Freelance",
    }
    return mapping.get(t, text.strip()) or None

def normalize_workable_posted_date(text: str):
    if not text:
        return None

    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("posted", "").strip()
    t = t.replace("about", "").replace("approx", "").strip()

    now = datetime.now(timezone.utc)

    if "today" in t:
        return now.strftime("%Y-%m-%d")
    if "yesterday" in t:
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")

    m = re.search(r"(\d+)\s*minute", t)
    if m:
        return now.strftime("%Y-%m-%d")

    m = re.search(r"(\d+)\s*hour", t)
    if m:
        return now.strftime("%Y-%m-%d")

    m = re.search(r"(\d+)\s*day", t)
    if m:
        return (now - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")

    m = re.search(r"(\d+)\s*week", t)
    if m:
        return (now - timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d")

    m = re.search(r"(\d+)\s*month", t)
    if m:
        return (now - timedelta(days=int(m.group(1)) * 30)).strftime("%Y-%m-%d")

    m = re.search(r"(\d+)\s*year", t)
    if m:
        return (now - timedelta(days=int(m.group(1)) * 365)).strftime("%Y-%m-%d")

    return None

def run():
    companies_to_scrape = load_target_urls_from_supabase()
    if not companies_to_scrape:
        print("No Workable companies found in companies_sustain table.")
        return

    print(f"Found {len(companies_to_scrape)} Workable domains to scrape.")

    discovered_jobs = []
    seen_urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        context.add_init_script("""
            document.addEventListener('DOMContentLoaded', () => {
                [...document.querySelectorAll('button')]
                  .find(b => /accept|agree/i.test(b.innerText))
                  ?.click();
            });
        """)

        print("\n--- PHASE 1: Discovering Jobs ---")
        for target in companies_to_scrape:
            company_name = target["company"]
            base_url = target["url"]
            company_sustain_id = target["company_sustain_id"]
            print(f"Checking {company_name} at {base_url}")

            page = context.new_page()

            try:
                page.goto(base_url, timeout=60000, wait_until="domcontentloaded")
                page.wait_for_timeout(2500)

                try:
                    btn = page.locator('xpath=//button[contains(text(),"Accept") or contains(text(),"Agree")]')
                    if btn.count() > 0:
                        btn.first.click(force=True)
                        page.wait_for_timeout(500)
                except Exception:
                    pass

                jobs = page.locator('xpath=//li[@data-ui="job"]')
                page.wait_for_timeout(1500)

                if jobs.count() == 0:
                    print("  -> No jobs found.")
                    continue

                while True:
                    more = page.locator('xpath=//button[@data-ui="load-more-button"]')
                    if more.count() == 0:
                        break
                    try:
                        more.first.scroll_into_view_if_needed()
                        more.first.click(force=True)
                        page.wait_for_timeout(1200)
                    except Exception:
                        break

                jobs = page.locator('xpath=//li[@data-ui="job"]')

                for i in range(jobs.count()):
                    job = jobs.nth(i)

                    title = safe_text(job.locator('xpath=.//h3[@data-ui="job-title"]'))

                    if not title or not is_sustainability_title(title):
                        continue

                    href = safe_attr(job.locator('xpath=.//a[1]'), "href")
                    if not href:
                        continue

                    if job_url in seen_urls or job_url in existing_job_urls:
                        continue

                    seen_urls.add(job_url)

                    dept = safe_text(job.locator('xpath=.//span[@data-ui="job-department"]'))
                    jtype = safe_text(job.locator('xpath=.//span[@data-ui="job-type"]'))
                    posted_text = safe_text(job.locator('xpath=.//small[@data-ui="job-posted"][1]'))
                    loc = safe_text(job.locator('xpath=.//*[@data-ui="job-location"]'))

                    discovered_jobs.append({
                        "company_name": company_name,
                        "company_sustain_id": company_sustain_id,
                        "title": title,
                        "url": job_url,
                        "department": dept or None,
                        "location": loc or None,
                        "job_type": normalize_job_type(jtype),
                        "posted": normalize_workable_posted_date(posted_text),
                    })

            except Exception as e:
                print(f"  -> Error on {company_name}: {e}")
            finally:
                page.close()

        print(f"\nPhase 1 Complete: Found {len(discovered_jobs)} sustainability jobs.")

        print("\n--- PHASE 2: Fetching Job Details (BeautifulSoup) ---")
        batch = []

        for index, job in enumerate(discovered_jobs):
            job_url = job["url"]
            print(f"Scraping [{index + 1}/{len(discovered_jobs)}]: {job['title']} at {job['company_name']}")

            page = context.new_page()

            try:
                page.goto(job_url, timeout=60000, wait_until="domcontentloaded")
                page.wait_for_timeout(1000)

                try:
                    page.wait_for_selector('section[data-ui="job-description"]', timeout=12000)
                except Exception:
                    pass

                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                description = None
                location = job["location"]
                job_type = job["job_type"]
                department = job["department"]

                loc_block = soup.select_one('div[data-ui="job-location"]')
                if loc_block:
                    location = clean_text(loc_block.get_text(" ", strip=True)) or location

                type_tag = soup.find(attrs={"data-ui": "job-type"})
                if type_tag:
                    job_type = normalize_job_type(type_tag.get_text(strip=True)) or job_type

                dept_tag = soup.find(attrs={"data-ui": "job-department"})
                if dept_tag:
                    department = clean_text(dept_tag.get_text(strip=True)) or department

                parts = []
                desc_section = soup.find("section", {"data-ui": "job-description"})
                if desc_section:
                    parts.append(desc_section.get_text(separator="\n", strip=True))

                req_section = soup.find("section", {"data-ui": "job-requirements"})
                if req_section:
                    parts.append(req_section.get_text(separator="\n", strip=True))

                ben_section = soup.find("section", {"data-ui": "job-benefits"})
                if ben_section:
                    parts.append(ben_section.get_text(separator="\n", strip=True))

                full_desc = "\n\n".join([p for p in parts if p]).strip()
                if full_desc:
                    description = full_desc

                if not location:
                    meta_loc = soup.find("meta", property="og:locality")
                    if meta_loc and meta_loc.get("content"):
                        location = meta_loc.get("content")

                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": job["title"],
                    "job_url": job_url,
                    "location": location,
                    "department": department,
                    "published_date": job["posted"],
                    "job_type": job_type,
                    "original_description": description,
                    "is_active": True,
                })

            except Exception as e:
                print(f"  -> Skipping URL due to error: {e}")
                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": job["title"],
                    "job_url": job_url,
                    "location": job["location"],
                    "department": job["department"],
                    "published_date": job["posted"],
                    "job_type": job["job_type"],
                    "original_description": "FAILED_TO_LOAD",
                    "is_active": True,
                })

            finally:
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
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
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
    """Load Darwinbox companies from companies_sustain Supabase table."""
    targets = []
    try:
        response = (
            supabase.table("companies_sustain")
            .select("id, company_name, jobs_page_url, ats_detected")
            .execute()
        )
        for row in response.data:
            ats = row.get("ats_detected", "") or ""
            if "darwinbox" not in ats.lower():
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

def insert_jobs_to_supabase(jobs: list):
    """Insert a batch of job records into jobs_sustain table."""
    if not jobs:
        return
    try:
        # Upsert on job_url to avoid duplicates on re-runs
        supabase.table("jobs_sustain").upsert(jobs, on_conflict="job_url").execute()
        print(f"  -> Inserted/updated {len(jobs)} jobs into jobs_sustain.")
    except Exception as e:
        print(f"  -> Error inserting jobs: {e}")

def clean(text):
    if not text:
        return ""
    text = str(text).replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()

def normalize_date(text):
    if not text or text == "NA":
        return None

    text = text.strip()
    lower = text.lower()
    today = datetime.now(timezone.utc)

    match = re.search(r"(\d+)\s+day[s]?\s+ago", lower)
    if match:
        dt = today - timedelta(days=int(match.group(1)))
        return dt.strftime("%Y-%m-%d")

    if "yesterday" in lower:
        dt = today - timedelta(days=1)
        return dt.strftime("%Y-%m-%d")

    if "today" in lower or "just now" in lower:
        return today.strftime("%Y-%m-%d")

    for fmt in ("%b %d, %Y", "%d-%m-%Y", "%d-%b-%Y"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None

def parse_experience_range(exp_text):
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

def safe_text(locator):
    try:
        return locator.first.inner_text().strip() if locator.count() > 0 else ""
    except Exception:
        return ""

def safe_attr(locator, attr):
    try:
        val = locator.first.get_attribute(attr) if locator.count() > 0 else None
        return val.strip() if val else ""
    except Exception:
        return ""

def click_if_exists(page, locator, wait=1500):
    try:
        if locator.count() > 0:
            locator.first.scroll_into_view_if_needed()
            page.wait_for_timeout(250)
            locator.first.click(force=True, timeout=5000)
            page.wait_for_timeout(wait)
            return True
    except Exception:
        return False
    return False

def wait_for_html_ready(page, timeout=30000):
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout)
        page.wait_for_selector("ui-job-tile, table.db-table-one", state="attached", timeout=timeout)
    except PlaywrightTimeoutError:
        pass
    except Exception:
        pass

def run():
    companies_to_scrape = load_target_urls_from_supabase()
    if not companies_to_scrape:
        print("No Darwinbox companies found in companies_sustain table.")
        return

    print(f"Found {len(companies_to_scrape)} Darwinbox domains to scrape.")

    discovered_jobs = []
    seen_urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--window-size=1920,1080"
            ]
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            bypass_csp=True
        )

        print("\n--- PHASE 1: Discovering Jobs ---")
        for target in companies_to_scrape:
            company_name = target["company"]
            base_url = target["url"]
            company_sustain_id = target["company_sustain_id"]
            print(f"Checking {company_name} at {base_url}")

            page = context.new_page()

            try:
                page.goto(base_url, timeout=60000, wait_until="domcontentloaded")
                wait_for_html_ready(page, timeout=20000)

                # --- LAYOUT 1: CARDS ---
                if page.locator("ui-job-tile").count() > 0:
                    previous_count = 0
                    while True:
                        load_more = page.locator('xpath=//span[normalize-space()="Load More Jobs"]')
                        if load_more.count() > 0:
                            click_if_exists(page, load_more, wait=1500)
                            current_count = page.locator("ui-job-tile").count()
                            if current_count == previous_count:
                                break
                            previous_count = current_count
                        else:
                            break

                    cards = page.locator("ui-job-tile")
                    for i in range(cards.count()):
                        card = cards.nth(i)
                        title = safe_text(card.locator(".job-title"))

                        if not is_sustainability_title(title):
                            continue

                        rel_link = safe_attr(card.locator("a.action-btn"), "href")
                        job_url = urljoin(page.url, rel_link) if rel_link else ""
                        dept = safe_text(card.locator('img[src*="department"] >> xpath=following-sibling::span'))

                        if not job_url or job_url in seen_urls:
                            continue
                        seen_urls.add(job_url)

                        discovered_jobs.append({
                            "company_name": company_name,
                            "company_sustain_id": company_sustain_id,
                            "title": title,
                            "url": job_url,
                            "department": dept,
                            "location": "",
                            "job_type": "",
                            "posted": None,
                        })

                # --- LAYOUT 2: TABLE ---
                elif page.locator("table.db-table-one").count() > 0:
                    while True:
                        rows = page.locator("table.db-table-one tbody tr")
                        for i in range(rows.count()):
                            row = rows.nth(i)
                            title_el = row.locator('td[data-th="Job title"] a')
                            title = safe_text(title_el)

                            if not is_sustainability_title(title):
                                continue

                            rel_link = safe_attr(title_el, "href")
                            job_url = urljoin(page.url, rel_link) if rel_link else ""
                            dept = safe_text(row.locator('td[data-th="Home Team"] span'))
                            location = safe_text(row.locator('td[data-th="Location"] span'))
                            emp_type = safe_text(row.locator('td[data-th="Employee Type"] span'))
                            posted_raw = safe_text(row.locator('td[data-th="Job posted on"] span'))

                            if not job_url or job_url in seen_urls:
                                continue
                            seen_urls.add(job_url)

                            discovered_jobs.append({
                                "company_name": company_name,
                                "company_sustain_id": company_sustain_id,
                                "title": title,
                                "url": job_url,
                                "department": dept,
                                "location": location,
                                "job_type": emp_type,
                                "posted": normalize_date(posted_raw),
                            })

                        next_btn = page.locator("li.pagination-next:not(.disabled) a")
                        if next_btn.count() > 0:
                            next_btn.click(force=True)
                            page.wait_for_timeout(2000)
                        else:
                            break

            except Exception as e:
                print(f"  -> Error on {company_name}: {e}")
            finally:
                page.close()

        print(f"\nPhase 1 Complete: Found {len(discovered_jobs)} sustainability jobs.")

        print("\n--- PHASE 2: Fetching Job Details ---")
        batch = []
        BATCH_SIZE = 20

        for index, job in enumerate(discovered_jobs):
            job_url = job["url"]
            print(f"Scraping [{index + 1}/{len(discovered_jobs)}]: {job['title']} at {job['company_name']}")

            page = context.new_page()

            try:
                page.goto(job_url, timeout=45000, wait_until="domcontentloaded")
                try:
                    page.wait_for_selector(
                        ".jd-container, .job-summary, .jd, .job-description, .job-details, .section.mobile-section",
                        timeout=15000
                    )
                except Exception:
                    pass

                page.wait_for_timeout(1000)

                description = ""
                experience_text = ""
                location = job["location"]
                job_type = job["job_type"]
                department = job["department"]
                posted = job["posted"]

                # Extract Description
                for selector in [".jd-container", ".job-summary", ".jd", ".job-description"]:
                    try:
                        loc = page.locator(selector)
                        if loc.count() > 0:
                            text = clean(loc.first.inner_text())
                            if text and len(text) > 50:
                                description = text
                                break
                    except Exception:
                        continue

                # Extract Snapshot Items
                try:
                    snapshot_items = page.locator(".section.mobile-section .grid-item")
                    for i in range(snapshot_items.count()):
                        label = clean(snapshot_items.nth(i).locator(".label").inner_text()).lower()
                        val_loc = snapshot_items.nth(i).locator(".value")
                        val = clean(val_loc.first.inner_text()) if val_loc.count() > 0 else ""

                        if label == "location": location = location or val
                        elif "employee type" in label or label == "type": job_type = job_type or val
                        elif "department" in label or "function" in label: department = department or val
                        elif "experience" in label: experience_text = val
                        elif "updated date" in label or "job posted on" in label: posted = posted or normalize_date(val)
                except Exception:
                    pass

                # Extract Details from Job Details block
                try:
                    details = page.locator(".job-details .job-details-item")
                    for i in range(details.count()):
                        label = clean(details.nth(i).locator("label").first.inner_text()).lower()
                        val = clean(details.nth(i).locator("p").first.inner_text())

                        if "location" in label: location = location or val
                        elif "employee type" in label or "job type" in label or "type" in label: job_type = job_type or val
                        elif "function" in label or "department" in label: department = department or val
                        elif "experience range" in label or "experience" in label: experience_text = experience_text or val
                        elif "job posted on" in label or "updated date" in label: posted = posted or normalize_date(val)
                except Exception:
                    pass

                # Fallback Experience
                if not experience_text:
                    try:
                        exp_item = page.locator(".job-details-item.experience-range p")
                        if exp_item.count() > 0:
                            experience_text = clean(exp_item.first.inner_text())
                    except Exception:
                        pass

                min_exp, max_exp = parse_experience_range(experience_text)

                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": job["title"],
                    "original_description": description,
                    "job_type": job_type or None,
                    "department": department or None,
                    "job_url": job_url,
                    "location": location or None,
                    "published_date": posted,
                    "min_exp": min_exp,
                    "max_exp": max_exp,
                    "is_active": True,
                })

            except Exception as e:
                print(f"  -> Skipping URL due to error: {e}")
                batch.append({
                    "company_sustain_id": job["company_sustain_id"],
                    "title": job["title"],
                    "original_description": "FAILED_TO_LOAD",
                    "job_type": job["job_type"] or None,
                    "department": job["department"] or None,
                    "job_url": job_url,
                    "location": job["location"] or None,
                    "published_date": job["posted"],
                    "min_exp": None,
                    "max_exp": None,
                    "is_active": True,
                })

            finally:
                page.close()

            # Insert in batches to avoid large payloads
            if len(batch) >= BATCH_SIZE:
                insert_jobs_to_supabase(batch)
                batch = []

        # Insert remaining
        if batch:
            insert_jobs_to_supabase(batch)

        browser.close()

    print("\nDone! All jobs inserted into jobs_sustain table.")


if __name__ == "__main__":
    run()
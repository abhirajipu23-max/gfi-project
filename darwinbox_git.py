import time
import re
import pandas as pd
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

# Playwright imports
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

# Database
from dotenv import load_dotenv
from supabase import create_client, Client

# ================= CONFIGURATION =================
load_dotenv()
SUPABASE_URL = "https://ufnaxahhlblwpdomlybs.supabase.co"
SUPABASE_KEY = "sb_publishable_1d4J1Ll81KwhYPOS40U8mQ_qtCccNsa"

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Settings
BATCH_SIZE = 100
MAX_CONCURRENT_PAGES = 15 

# ================= HELPERS (Restored from your code) =================
def clean_text(text: str) -> str:
    if not text: return ""
    text = text.replace('\u200b', '')
    return re.sub(r'\s+', ' ', text).strip()

def normalize_date(text: str) -> str:
    if not text or text == "NA": return None 
    text = text.lower().strip()
    today = datetime.now(timezone.utc)

    match = re.search(r"(\d+)\s+day[s]?\s+ago", text)
    if match:
        dt = today - timedelta(days=int(match.group(1)))
        return dt.isoformat()
    if "yesterday" in text:
        dt = today - timedelta(days=1)
        return dt.isoformat()
    if "today" in text or "just now" in text:
        return today.isoformat()
    return None

def safe_text(locator):
    try:
        return locator.first.inner_text().strip() if locator.count() > 0 else "NA"
    except:
        return "NA"

def safe_attr(locator, attr):
    try:
        val = locator.first.get_attribute(attr) if locator.count() > 0 else None
        return val.strip() if val else "NA"
    except:
        return "NA"

def click_if_exists(page, locator, wait=1200):
    try:
        if locator.count() > 0:
            locator.first.scroll_into_view_if_needed()
            page.wait_for_timeout(250)
            locator.first.click(force=True, timeout=5000)
            page.wait_for_timeout(wait)
            return True
    except:
        return False
    return False

def wait_for_any_layout(page, timeout=20000):
    """Waits for either Darwinbox Card or Table layout using JS evaluation."""
    page.wait_for_timeout(1000)
    try:
        page.wait_for_function(
            """
            () => {
                const card = document.querySelector("ui-job-tile");
                const tableRow = document.querySelector("table.db-table-one tbody tr");
                return !!card || !!tableRow;
            }
            """,
            timeout=timeout
        )
    except:
        pass

# ================= STAGE 1: DISCOVERY (SYNC) =================

def get_companies():
    try:
        res = supabase.table("ats_website").select("*").execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        print(f"Error fetching companies: {e}")
        return pd.DataFrame()

def save_to_supabase(scrapes_data, jobs_data):
    # 1. Upload Scrapes to scrapes_duplicate
    if scrapes_data:
        print(f"Uploading {len(scrapes_data)} scrape records...")
        try:
            supabase.table("scrapes_duplicate").upsert(scrapes_data).execute()
            print("✅ Scrape records uploaded.")
        except Exception as e:
            print(f"❌ Error uploading scrapes: {e}")

    if not jobs_data:
        print("No jobs to process.")
        return

    # 2. Deduplicate internally
    unique_map = {j["job_url"]: j for j in jobs_data if j["job_url"] and j["job_url"] != "NA"}
    unique_jobs_list = list(unique_map.values())
    all_scraped_urls = list(unique_map.keys())

    # 3. Check DB for existing URLs in jobs_duplicate
    print(f"Checking {len(all_scraped_urls)} jobs against database...")
    existing_urls = set()
    CHECK_BATCH_SIZE = 100 
    
    for i in range(0, len(all_scraped_urls), CHECK_BATCH_SIZE):
        batch_urls = all_scraped_urls[i : i + CHECK_BATCH_SIZE]
        try:
            response = supabase.table("jobs_duplicate") \
                .select("job_url") \
                .in_("job_url", batch_urls) \
                .execute()
            for row in response.data:
                existing_urls.add(row["job_url"])
        except Exception as e:
            print(f"   ⚠️ Error checking duplicates (Batch {i}): {e}")

    # 4. Filter & Insert into jobs_duplicate
    new_jobs = [j for j in unique_jobs_list if j["job_url"] not in existing_urls]
    print(f"   -> Skipped {len(unique_jobs_list) - len(new_jobs)} existing jobs.")
    print(f"   -> Inserting {len(new_jobs)} new jobs...")

    if not new_jobs:
        return

    for i in range(0, len(new_jobs), BATCH_SIZE):
        batch = new_jobs[i : i + BATCH_SIZE]
        try:
            supabase.table("jobs_duplicate").insert(batch).execute()
            print(f"   ↳ Uploaded batch {i} - {i + len(batch)}")
        except Exception as e:
            print(f"   ❌ Error uploading batch: {e}")

def scrape_darwinbox_discovery(page, url, scrape_uuid):
    jobs_collected = []
    print(f"   Scanning: {url}")
    
    try:
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        wait_for_any_layout(page, timeout=15000)
    except Exception as e:
        print(f"   ⚠️ Connection failed: {e}")
        return []

    # --- Strategy 1: Card Layout ---
    if page.locator("ui-job-tile").count() > 0:
        print("   -> Detected CARD layout")
        
        # Load More (Using your specific XPath)
        while True:
            load_more = page.locator('xpath=//span[normalize-space()="Load More Jobs"]')
            if not click_if_exists(page, load_more, wait=1500):
                break
        
        cards = page.locator("ui-job-tile")
        for i in range(cards.count()):
            card = cards.nth(i)
            
            title = safe_text(card.locator(".job-title"))
            rel_link = safe_attr(card.locator("a.action-btn"), "href")
            
            # Dept using your specific XPath
            dept = safe_text(card.locator('img[src*="department"] >> xpath=following-sibling::span'))
            
            job_url = urljoin(page.url, rel_link)
            if not rel_link or rel_link == "NA":
                job_url = "NA"

            job_id = uuid.uuid4().int % (2**63 - 1)
            now_iso = datetime.now(timezone.utc).isoformat()

            jobs_collected.append({
                "id": job_id,
                "created_at": now_iso,
                "updated_at": now_iso,
                "scrape_id": scrape_uuid,
                "title": title,
                "job_url": job_url,
                "is_active": True,
                "department": dept if dept != "NA" else None,
                "original_description": None 
            })

    # --- Strategy 2: Table Layout ---
    elif page.locator("table.db-table-one").count() > 0:
        print("   -> Detected TABLE layout")
        try:
            page.wait_for_selector("table.db-table-one tbody tr", timeout=10000)
        except: pass

        while True:
            rows = page.locator("table.db-table-one tbody tr")
            for i in range(rows.count()):
                row = rows.nth(i)
                
                title_el = row.locator('td[data-th="Job title"] a')
                dept_el = row.locator('td[data-th="Department"] span')
                type_el = row.locator('td[data-th="Employee Type"] span')
                date_el = row.locator('td[data-th="Job posted on"] span')

                title = safe_text(title_el)
                rel_link = safe_attr(title_el, "href")
                
                dept = safe_text(dept_el)
                emp_type = safe_text(type_el)
                posted_raw = safe_text(date_el)
                
                job_url = urljoin(page.url, rel_link)
                if not rel_link or rel_link == "NA":
                    job_url = "NA"

                job_id = uuid.uuid4().int % (2**63 - 1)
                now_iso = datetime.now(timezone.utc).isoformat()

                jobs_collected.append({
                    "id": job_id,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                    "scrape_id": scrape_uuid,
                    "title": title,
                    "job_url": job_url,
                    "is_active": True,
                    "published_date": normalize_date(posted_raw),
                    "job_type": emp_type if emp_type != "NA" else None,
                    "department": dept if dept != "NA" else None,
                    "original_description": None
                })
            
            # Pagination using your selector
            next_btn = page.locator("li.pagination-next:not(.disabled) a")
            if next_btn.count() > 0:
                next_btn.click(force=True)
                page.wait_for_timeout(2000)
            else:
                break
    else:
        print("   ⚠️ No recognizable jobs found (Layout check failed).")

    return jobs_collected

def run_stage_1_discovery():
    print("\n--- STAGE 1: DISCOVERY (SYNC) ---")
    start_time = time.time()
    
    df = get_companies()
    if "ats_name" in df.columns:
        df = df[df["ats_name"].astype(str).str.lower().str.strip() == "darwinbox"]
    print(f"Found {len(df)} Darwinbox companies.")

    all_jobs = []
    all_scrapes = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use context with UA
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for _, row in df.iterrows():
            url = row["ats_url"]
            scrape_uuid = str(uuid.uuid4())
            now_iso = datetime.now(timezone.utc).isoformat()
            
            scrape_record = {
                "id": scrape_uuid, 
                "ats_website_id": row["id"], 
                "status": "pending",
                "created_at": now_iso, 
                "updated_at": now_iso,
                "finished_at": None
            }

            try:
                company_jobs = scrape_darwinbox_discovery(page, url, scrape_uuid)
                finish_iso = datetime.now(timezone.utc).isoformat()
                scrape_record["finished_at"] = finish_iso
                scrape_record["status"] = "success"
                if company_jobs: 
                    all_jobs.extend(company_jobs)
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                scrape_record["status"] = "failed"
                scrape_record["finished_at"] = datetime.now(timezone.utc).isoformat()
            
            all_scrapes.append(scrape_record)

        browser.close()

    clean_jobs = [j for j in all_jobs if j["job_url"] and j["job_url"] != "NA"]
    save_to_supabase(all_scrapes, clean_jobs)
    print(f"Stage 1 Runtime: {time.time() - start_time:.2f}s")


# ================= STAGE 2: ENRICHMENT (ASYNC) =================

def get_pending_darwinbox_jobs():
    print("Fetching pending Darwinbox jobs from Supabase...")
    try:
        res = (
            supabase.table("jobs_duplicate")
            .select("id, job_url, location, job_type, department, scrapes_duplicate(ats_website(ats_name))")
            .is_("original_description", "null")
            .execute()
        )
        
        all_jobs = res.data
        darwin_jobs = [
            j for j in all_jobs 
            if j.get("scrapes_duplicate") and "Darwinbox" in j["scrapes_duplicate"]["ats_website"]["ats_name"]
        ]
        
        print(f"Total jobs pending enrichment: {len(darwin_jobs)}")
        return darwin_jobs
    except Exception as e:
        print(f"Error fetching jobs: {e}")
        return []

def update_job_sync(job_id, payload):
    try:
        clean_payload = {k: v for k, v in payload.items() if v}
        if clean_payload:
            clean_payload["updated_at"] = datetime.utcnow().isoformat()
            supabase.table("jobs_duplicate").update(clean_payload).eq("id", job_id).execute()
            print(f"   ✅ [ID {job_id}] Updated")
    except Exception as e:
        print(f"   ❌ [ID {job_id}] Update Failed: {e}")

async def extract_details_async(page):
    data = {"original_description": None, "location": None, "job_type": None, "department": None}
    
    # === 1. FIND THE DESCRIPTION (Used your selectors) ===
    candidates = [".jd-container", ".job-summary"]
    for selector in candidates:
        try:
            if await page.locator(selector).count() > 0:
                await page.evaluate(f"document.querySelector('{selector}').querySelectorAll('.hidden').forEach(el => el.remove())")
                text = await page.locator(selector).first.inner_text()
                if text and len(text) > 50:
                    data["original_description"] = clean_text(text)
                    break
        except: continue
    
    # === 2. EXTRACT METADATA (Strategies A, B, C) ===

    # Strategy A: Grid (.section.mobile-section)
    try:
        snapshot_items = page.locator(".section.mobile-section .grid-item")
        if await snapshot_items.count() > 0:
            for i in range(await snapshot_items.count()):
                try:
                    label = await snapshot_items.nth(i).locator(".label").inner_text()
                    val = await snapshot_items.nth(i).locator(".value").inner_text()
                    label, val = label.strip(), clean_text(val)

                    if "Location" in label: data["location"] = val
                    elif "Type" in label: data["job_type"] = val
                    elif "Department" in label: data["department"] = val
                except: continue
    except: pass

    # Strategy B: List (.job-details .job-details-item)
    if not data.get("location") or not data.get("department"):
        try:
            details = page.locator(".job-details .job-details-item")
            if await details.count() > 0:
                for i in range(await details.count()):
                    try:
                        label = await details.nth(i).locator("label").first.inner_text()
                        val = await details.nth(i).locator("p").first.inner_text()
                        label, val = label.strip(), clean_text(val)
                        
                        if "Location" in label: data["location"] = val
                        elif "Type" in label: data["job_type"] = val
                        elif "Function" in label or "Department" in label: data["department"] = val
                    except: continue
        except: pass

    # Strategy C: Inline Classes (Fallback)
    if not data.get("location"):
        try:
            loc_text = await page.locator(".Office.Location").first.inner_text()
            data["location"] = clean_text(loc_text.replace("Office Location:", "").strip())
        except: pass

    if not data.get("department"):
        try:
            dept_text = await page.locator(".Designation").first.inner_text()
            data["department"] = clean_text(dept_text.replace("Designation:", "").strip())
        except: pass
            
    return data

async def process_job(sem, context, job):
    async with sem:
        url = job.get("job_url")
        job_id = job.get("id")
        if not url: return

        page = await context.new_page()
        try:
            try:
                await page.goto(url, timeout=30000, wait_until="networkidle")
            except: pass 

            await asyncio.sleep(2)
            
            # Wait for specific containers
            try:
                await page.wait_for_selector(".jd-container, .job-summary, .job-details", timeout=5000)
            except: pass

            scraped = await extract_details_async(page)
            
            payload = {
                "original_description": scraped["original_description"],
                "location": scraped["location"] or job.get("location"),
                "job_type": scraped["job_type"] or job.get("job_type"),
                "department": scraped["department"] or job.get("department")
            }

            if payload["original_description"]:
                await asyncio.to_thread(update_job_sync, job_id, payload)
            else:
                print(f"   ⚠️ [ID {job_id}] No description found.")

        except Exception as e:
            print(f"   ❌ [ID {job_id}] Failed: {str(e)[:50]}")
        finally:
            await page.close()

async def run_stage_2_enrichment():
    print("\n--- STAGE 2: ENRICHMENT (ASYNC) ---")
    jobs = get_pending_darwinbox_jobs()
    if not jobs: return

    sem = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
             user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        tasks = [process_job(sem, context, job) for job in jobs]
        await asyncio.gather(*tasks)
        
        await browser.close()
    print("Enrichment completed.")

# ================= MAIN =================
if __name__ == "__main__":
    run_stage_1_discovery()
    asyncio.run(run_stage_2_enrichment())
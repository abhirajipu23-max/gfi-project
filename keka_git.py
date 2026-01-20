import time
import re
import pandas as pd
import uuid
import asyncio
import atexit
from datetime import datetime, timedelta, timezone
from playwright.sync_api import sync_playwright, TimeoutError as SyncTimeoutError
from playwright.async_api import async_playwright, TimeoutError as AsyncTimeoutError
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ---------------------------
# CONFIGURATION
# ---------------------------
SUPABASE_URL = "https://ufnaxahhlblwpdomlybs.supabase.co"
SUPABASE_KEY = "sb_publishable_1d4J1Ll81KwhYPOS40U8mQ_qtCccNsa" 

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OUTPUT_JOBS_FILE = "keka_jobs.csv"
OUTPUT_SCRAPES_FILE = "keka_scrapes.csv"

# ---------------------------
# STAGE 1: DISCOVERY (Sync)
# ---------------------------

JOB_CARD_SELECTOR = (
    "a.kh-card, "
    "a.kh-job-card, "
    "a.card.job-card, "
    "a[href*='jobdetails']"
)

def get_companies():
    """Fetch companies from the 'ats_website' table."""
    try:
        res = supabase.table("ats_website").select("*").execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        print(f"Error fetching companies: {e}")
        return pd.DataFrame()

def normalize_posted_date(text: str) -> str:
    if not text: return None 
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

def save_to_supabase(scrapes_data, jobs_data):
    """
    1. Uploads scrape history.
    2. Checks DB for existing job URLs (Batched).
    3. Inserts ONLY new jobs.
    """
    # --- 1. Upload Scrapes ---
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

    # --- 2. Deduplicate internally ---
    unique_map = {j["job_url"]: j for j in jobs_data if j["job_url"]}
    unique_jobs_list = list(unique_map.values())
    all_scraped_urls = list(unique_map.keys())

    # --- 3. Check DB for existing URLs ---
    print(f"Checking {len(all_scraped_urls)} jobs against database...")
    
    existing_urls = set()
    
    # --- FIX: Reduced Batch Size from 100 to 40 to avoid 'Bad Request' ---
    CHECK_BATCH_SIZE = 40 
    
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

    # --- 4. Filter & Insert ---
    new_jobs = [j for j in unique_jobs_list if j["job_url"] not in existing_urls]
    
    print(f"   -> Skipped {len(unique_jobs_list) - len(new_jobs)} existing jobs.")
    print(f"   -> Inserting {len(new_jobs)} new jobs...")

    if not new_jobs:
        return

    INSERT_BATCH_SIZE = 100
    for i in range(0, len(new_jobs), INSERT_BATCH_SIZE):
        batch = new_jobs[i : i + INSERT_BATCH_SIZE]
        try:
            supabase.table("jobs_duplicate").insert(batch).execute()
            print(f"   ↳ Uploaded batch {i} - {i + len(batch)}")
        except Exception as e:
            print(f"   ❌ Error uploading batch: {e}")

def scrape_keka_jobs(page, url, scrape_uuid):
    jobs = []
    print(f"   Scanning: {url}")
    try:
        page.goto(url, timeout=30000)
        try:
            page.wait_for_selector(JOB_CARD_SELECTOR, timeout=10000)
        except:
            print(f"   ℹ️ No cards found immediately, checking page content...")
    except SyncTimeoutError:
        print(f"   ⚠️ Timeout reaching: {url}")
        return []
    except Exception as e:
        print(f"   ⚠️ Connection error: {e}")
        return []

    job_cards = page.query_selector_all(JOB_CARD_SELECTOR)

    if not job_cards:
        print(f"   ⚠️ No job cards found on {url}")
        return []

    for card in job_cards:
        def safe_text(selector):
            el = card.query_selector(selector)
            return el.inner_text().strip() if el else ""

        raw_url = card.get_attribute("href") or ""
        if raw_url and not raw_url.startswith("http"):
            job_url = url.rstrip("/") + "/" + raw_url.lstrip("/")
        else:
            job_url = raw_url

        title = safe_text("h4.kh-job-title, h3.job-title")
        posted_raw = safe_text("small, span.text-secondary")
        job_type = safe_text(".job-type, .type")
        
        now_iso = datetime.now(timezone.utc).isoformat()
        job_id = uuid.uuid4().int % (2**63 - 1) # BigInt Safe

        jobs.append({
            "id": job_id,
            "created_at": now_iso,
            "updated_at": now_iso,
            "scrape_id": scrape_uuid,
            "title": title,
            "job_url": job_url,
            "is_active": True,
            "published_date": normalize_posted_date(posted_raw),
            "job_type": job_type if job_type else None,
            "original_description": None,
            "internal_slug": None
        })

    return jobs

def run_stage_1_discovery():
    print("\n--- STAGE 1: DISCOVERY ---")
    start_time = time.time()
    
    df = get_companies()
    if "ats_name" in df.columns:
        df = df[df["ats_name"].astype(str).str.lower().str.strip() == "keka"]
    print(f"Found {len(df)} Keka companies to scrape.")

    all_jobs, all_scrapes = [], []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()

        for _, row in df.iterrows():
            company_url = row["ats_url"]
            scrape_uuid = str(uuid.uuid4())
            now_iso = datetime.now(timezone.utc).isoformat()
            
            scrape_record = {
                "id": scrape_uuid, "ats_website_id": row["id"], 
                "status": "pending", "created_at": now_iso, "updated_at": now_iso
            }

            try:
                company_jobs = scrape_keka_jobs(page, company_url, scrape_uuid)
                scrape_record["finished_at"] = datetime.now(timezone.utc).isoformat()
                scrape_record["status"] = "success"
                if company_jobs: all_jobs.extend(company_jobs)
            except Exception as e:
                print(f"   ❌ Critical failure: {e}")
                scrape_record["status"] = "failed"
            
            all_scrapes.append(scrape_record)

        browser.close()

    clean_jobs = [j for j in all_jobs if j["job_url"]]
    save_to_supabase(all_scrapes, clean_jobs)
    print(f"Stage 1 Runtime: {time.time() - start_time:.2f}s")


# ---------------------------
# STAGE 2: ENRICHMENT (Async)
# ---------------------------

MAX_CONCURRENT_PAGES = 5

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""

def parse_experience(exp_text):
    if not exp_text: return None, None
    exp_text = exp_text.lower().strip()
    if "fresher" in exp_text or re.fullmatch(r"0\s*(years|yrs)?", exp_text):
        return 0, 0
    plus_match = re.search(r"(\d+)\s*\+", exp_text)
    if plus_match: return int(plus_match.group(1)), None
    range_match = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)", exp_text)
    if range_match: return int(range_match.group(1)), int(range_match.group(2))
    single_match = re.search(r"(\d+)", exp_text)
    if single_match:
        val = int(single_match.group(1))
        return val, val
    return None, None

def get_keka_jobs_from_supabase():
    print("Fetching Keka jobs from Supabase...")
    try:
        # --- FIX: Read from jobs_duplicate to match Stage 1 ---
        res = (
            supabase.table("jobs_duplicate")
            .select("id, job_url, scrapes_duplicate(ats_website(ats_name))")
            .is_("original_description", "null") # Only fetch pending ones
            .execute()
        )
    except Exception as e:
        print(f"Error fetching: {e}")
        return pd.DataFrame()

    if not res.data: return pd.DataFrame()

    df = pd.DataFrame(res.data)
    # Handle nested JSON
    df["ats_name"] = df["scrapes_duplicate"].apply(
        lambda x: x["ats_website"]["ats_name"] if x and x.get("ats_website") else None
    )
    
    keka_df = df[df["ats_name"].str.strip().str.lower().str.contains("keka", na=False)].copy()
    print(f"Total jobs pending: {len(df)} | Keka jobs to scrape: {len(keka_df)}")
    return keka_df

def update_supabase_record(job_id, payload):
    try:
        # --- FIX: Update jobs_duplicate ---
        supabase.table("jobs_duplicate").update(payload).eq("id", job_id).execute()
        print(f"   ✅ Updated Job ID {job_id}")
    except Exception as e:
        print(f"   ❌ Failed to update Job ID {job_id}: {e}")

async def scrape_job_async(browser, row, semaphore):
    job_url = row.get("job_url", "")
    job_id = row.get("id")
    if not job_url: return

    async with semaphore:
        page = await browser.new_page()
        try:
            await page.goto(job_url, timeout=20000)
            
            async def safe_text(selector):
                try: return clean_text(await page.locator(selector).inner_text())
                except: return ""

            experience_text = await safe_text("span.ki-user-tie >> xpath=../span[2]")
            location = await safe_text("span.ki-location >> xpath=../span[2]")
            job_type = await safe_text("span.ki-briefcase >> xpath=../span[2]")
            
            description = ""
            try:
                # Try container first
                if await page.locator(".job-description-container").count() > 0:
                    container = page.locator(".job-description-container")
                    li_texts = await container.locator("li").all_inner_texts()
                    if li_texts:
                        description = " | ".join(clean_text(t) for t in li_texts if t.strip())
                    else:
                        description = clean_text(await container.inner_text())
            except: pass

            min_exp, max_exp = parse_experience(experience_text)
            
            payload = {
                "min_exp": min_exp, "max_exp": max_exp,
                "location": location, "job_type": job_type,
                "original_description": description,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Remove empty fields
            payload = {k: v for k, v in payload.items() if v}

            await asyncio.to_thread(update_supabase_record, job_id, payload)

        except Exception as e:
            print(f"   ⚠️ Failed: {job_url[-30:]} | {e}")
        finally:
            await page.close()

async def run_stage_2_enrichment():
    print("\n--- STAGE 2: ENRICHMENT ---")
    df = get_keka_jobs_from_supabase()
    if df.empty:
        print("No pending Keka jobs found.")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
        
        tasks = [scrape_job_async(browser, row, semaphore) for _, row in df.iterrows()]
        await asyncio.gather(*tasks)
        await browser.close()
    print("Enrichment completed.")

# ---------------------------
# MAIN ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    # Run Discovery (Sync)
    run_stage_1_discovery()
    
    # Run Enrichment (Async)
    asyncio.run(run_stage_2_enrichment())
import time
import re
import pandas as pd
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, TimeoutError as SyncTimeoutError
from playwright.async_api import async_playwright

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OUTPUT_JOBS_FILE = "keka_jobs.csv"
OUTPUT_SCRAPES_FILE = "keka_scrapes.csv"

SCRAPES_TABLE = "scrapes"
JOBS_TABLE = "jobs"

BATCH_SIZE = 100
MAX_CONCURRENT_PAGES = 5

JOB_CARD_SELECTOR = (
    "a.kh-card, "
    "a.kh-job-card, "
    "a.card.job-card, "
    "a[href*='jobdetails']"
)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""


def normalize_posted_date(text: str) -> str:
    if not text:
        return None

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


def wait_for_html_ready_sync(page, timeout=20000):
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except:
        pass

    try:
        page.wait_for_function("() => document.readyState === 'complete'", timeout=timeout)
    except:
        pass

    try:
        page.wait_for_selector(JOB_CARD_SELECTOR, timeout=timeout)
    except:
        pass


def get_companies():
    try:
        res = supabase.table("ats_website").select("*").execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        print(f"Error fetching companies: {e}")
        return pd.DataFrame()


def save_to_supabase(scrapes_data, jobs_data):
    if scrapes_data:
        print(f"Uploading {len(scrapes_data)} scrape records...")
        try:
            supabase.table(SCRAPES_TABLE).upsert(scrapes_data).execute()
            print("Scrape records uploaded.")
        except Exception as e:
            print(f"Error uploading scrapes: {e}")
            return

    if not jobs_data:
        print("No jobs to process.")
        return

    unique_map = {j["job_url"]: j for j in jobs_data if j.get("job_url")}
    unique_jobs_list = list(unique_map.values())
    all_scraped_urls = list(unique_map.keys())

    print(f"Checking {len(all_scraped_urls)} jobs against database to skip duplicates...")
    existing_urls = set()

    CHECK_BATCH_SIZE = 40
    for i in range(0, len(all_scraped_urls), CHECK_BATCH_SIZE):
        batch_urls = all_scraped_urls[i : i + CHECK_BATCH_SIZE]
        try:
            response = (
                supabase.table(JOBS_TABLE)
                .select("job_url")
                .in_("job_url", batch_urls)
                .execute()
            )
            for row in response.data:
                existing_urls.add(row["job_url"])
        except Exception as e:
            print(f"Error checking duplicates (Batch {i}): {e}")

    new_jobs = [j for j in unique_jobs_list if j["job_url"] not in existing_urls]

    print(f"Skipped {len(unique_jobs_list) - len(new_jobs)} existing jobs.")
    print(f"Inserting {len(new_jobs)} new jobs...")

    if not new_jobs:
        print("No new jobs to insert.")
        return

    for i in range(0, len(new_jobs), BATCH_SIZE):
        batch = new_jobs[i : i + BATCH_SIZE]
        try:
            supabase.table(JOBS_TABLE).insert(batch).execute()
            print(f"Uploaded batch {i} - {i + len(batch)}")
        except Exception as e:
            print(f"Error uploading batch: {e}")


def scrape_keka_jobs(page, url, scrape_uuid):
    jobs = []
    print(f"Scanning: {url}")

    try:
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        wait_for_html_ready_sync(page, timeout=25000)
    except SyncTimeoutError:
        print(f"Timeout reaching: {url}")
        return []
    except Exception as e:
        print(f"Connection error: {e}")
        return []

    job_cards = page.query_selector_all(JOB_CARD_SELECTOR)
    if not job_cards:
        print(f"No job cards found on {url}")
        return []

    for card in job_cards:
        def safe_text(selector):
            el = card.query_selector(selector)
            return clean_text(el.inner_text()) if el else ""

        raw_url = card.get_attribute("href") or ""
        if raw_url and not raw_url.startswith("http"):
            job_url = urljoin(url, raw_url)
        else:
            job_url = raw_url

        title = safe_text("h4.kh-job-title, h3.job-title")
        posted_raw = safe_text("small, span.text-secondary")
        job_type = safe_text(".job-type, .type")

        # --- NEW: Location Extraction for Stage 1 ---
        # Based on snippet: span.font-large is used for Location AND Experience
        # Location is typically the first one in the container
        location = ""
        try:
            # Try specific selector based on the snippet provided
            # Locates the 'font-large' span that is likely the location
            loc_element = card.query_selector("span.font-large")
            if loc_element:
                location = clean_text(loc_element.inner_text())
        except:
            pass
        # --------------------------------------------

        now_iso = datetime.now(timezone.utc).isoformat()

        jobs.append({
            "created_at": now_iso,
            "updated_at": now_iso,
            "scrape_id": scrape_uuid,
            "title": title,
            "job_url": job_url,
            "is_active": True,
            "published_date": normalize_posted_date(posted_raw),
            "job_type": job_type if job_type else None,
            "original_description": None,
            "internal_slug": None,
            "min_exp": None,
            "max_exp": None,
            "location": location if location else None, # Store extracted location
        })

    return jobs


def run_stage_1_discovery():
    print("\n--- STAGE 1: DISCOVERY ---")
    start_time = time.time()

    df = get_companies()
    if df.empty:
        print("No companies found.")
        return

    if "ats_name" in df.columns:
        df = df[df["ats_name"].astype(str).str.lower().str.strip() == "keka"]

    print(f"Found {len(df)} Keka companies to scrape.")

    all_jobs, all_scrapes = [], []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()

        for _, row in df.iterrows():
            company_url = row.get("ats_url")
            if not company_url:
                continue

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
                company_jobs = scrape_keka_jobs(page, company_url, scrape_uuid)

                finish_iso = datetime.now(timezone.utc).isoformat()
                scrape_record["finished_at"] = finish_iso
                scrape_record["updated_at"] = finish_iso
                scrape_record["status"] = "success"

                if company_jobs:
                    all_jobs.extend(company_jobs)

            except Exception as e:
                print(f"Critical failure: {e}")
                finish_iso = datetime.now(timezone.utc).isoformat()
                scrape_record["finished_at"] = finish_iso
                scrape_record["updated_at"] = finish_iso
                scrape_record["status"] = "failed"

            all_scrapes.append(scrape_record)

        browser.close()

    clean_jobs = [j for j in all_jobs if j.get("job_url")]

    if all_scrapes:
        pd.DataFrame(all_scrapes).to_csv(OUTPUT_SCRAPES_FILE, index=False)
        print(f"\nBackup saved: {OUTPUT_SCRAPES_FILE}")

    if clean_jobs:
        pd.DataFrame(clean_jobs).to_csv(OUTPUT_JOBS_FILE, index=False)
        print(f"Backup saved (Stage 1 snapshot): {OUTPUT_JOBS_FILE}")

    save_to_supabase(all_scrapes, clean_jobs)
    print(f"Stage 1 Runtime: {time.time() - start_time:.2f}s")


def get_keka_jobs_from_supabase():
    print("Fetching pending Keka jobs from Supabase...")
    try:
        # Added 'location' to select query to check existing data in Stage 2
        res = (
            supabase.table(JOBS_TABLE)
            .select("id, job_url, location, scrapes(ats_website(ats_name))")
            .is_("original_description", "null")
            .execute()
        )
    except Exception as e:
        print(f"Error fetching: {e}")
        return pd.DataFrame()

    if not res.data:
        return pd.DataFrame()

    df = pd.DataFrame(res.data)

    df["ats_name"] = df["scrapes"].apply(
        lambda x: x["ats_website"]["ats_name"] if x and x.get("ats_website") else None
    )

    keka_df = df[df["ats_name"].str.strip().str.lower().str.contains("keka", na=False)].copy()
    print(f"Total pending jobs: {len(df)} | Keka jobs to scrape: {len(keka_df)}")
    return keka_df


def update_supabase_record(job_id, payload):
    try:
        supabase.table(JOBS_TABLE).update(payload).eq("id", job_id).execute()
        print(f"Updated Job ID {job_id}")
    except Exception as e:
        print(f"Failed to update Job ID {job_id}: {e}")


async def wait_for_html_ready_async(page, timeout=25000):
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except:
        pass

    try:
        await page.wait_for_function("() => document.readyState === 'complete'", timeout=timeout)
    except:
        pass

    try:
        await page.wait_for_selector(
            ".job-description-container, span.ki-user-tie, span.ki-location, span.ki-briefcase",
            timeout=timeout
        )
    except:
        pass


async def scrape_job_async(browser, row, semaphore):
    job_url = row.get("job_url", "")
    job_id = row.get("id")
    # Get existing location from DB row
    current_db_location = row.get("location")

    if not job_url:
        return

    async with semaphore:
        page = await browser.new_page()
        try:
            await page.goto(job_url, timeout=60000, wait_until="domcontentloaded")
            await wait_for_html_ready_async(page, timeout=25000)
            await page.wait_for_timeout(800)

            async def safe_text(selector):
                try:
                    loc = page.locator(selector)
                    if await loc.count() == 0:
                        return ""
                    return clean_text(await loc.first.inner_text())
                except:
                    return ""

            experience_text = await safe_text("span.ki-user-tie >> xpath=../span[2]")
            
            # Stage 2 Location Extraction
            extracted_location = await safe_text("span.ki-location >> xpath=../span[2]")
            
            job_type = await safe_text("span.ki-briefcase >> xpath=../span[2]")

            description = ""
            try:
                if await page.locator(".job-description-container").count() > 0:
                    container = page.locator(".job-description-container")
                    li_texts = await container.locator("li").all_inner_texts()
                    if li_texts:
                        description = " | ".join(clean_text(t) for t in li_texts if t.strip())
                    else:
                        description = clean_text(await container.inner_text())
            except:
                pass

            min_exp, max_exp = parse_experience(experience_text)

            payload = {
                "min_exp": min_exp,
                "max_exp": max_exp,
                # "location" is handled below with specific logic
                "job_type": job_type if job_type else None,
                "original_description": description if description else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # --- NEW: Stage 2 Logic for Location ---
            # If we found a location on the detailed page:
            if extracted_location:
                # If DB is null (None or empty string), update it.
                if not current_db_location: 
                    payload["location"] = extracted_location
                # Else: DB already has data, so we PASS (do not add to payload)
            # ---------------------------------------

            payload = {k: v for k, v in payload.items() if v is not None and v != ""}

            await asyncio.to_thread(update_supabase_record, job_id, payload)

        except Exception as e:
            print(f"Failed: {job_url[-40:]} | {e}")
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


def export_keka_jobs_backup_from_db():
    print("\nExporting final Keka jobs backup (Stage 1 + Stage 2) to CSV...")

    try:
        res = (
            supabase.table(JOBS_TABLE)
            .select("* , scrapes(ats_website(ats_name))")
            .execute()
        )
    except Exception as e:
        print(f"Error exporting jobs: {e}")
        return

    if not res.data:
        print("No jobs found to export.")
        return

    df = pd.DataFrame(res.data)

    if "scrapes" in df.columns:
        df["ats_name"] = df["scrapes"].apply(
            lambda x: x["ats_website"]["ats_name"] if x and x.get("ats_website") else None
        )

    df = df[df["ats_name"].str.lower().str.contains("keka", na=False)].copy()

    if "scrapes" in df.columns:
        df.drop(columns=["scrapes"], inplace=True)

    df.to_csv(OUTPUT_JOBS_FILE, index=False)
    print(f"Final backup saved: {OUTPUT_JOBS_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    run_stage_1_discovery()
    asyncio.run(run_stage_2_enrichment())
    export_keka_jobs_backup_from_db()
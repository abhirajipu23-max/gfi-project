import time
import re
import pandas as pd
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OUTPUT_JOBS_FILE = "darwinbox_jobs.csv"
OUTPUT_SCRAPES_FILE = "darwinbox_scrapes.csv"

BATCH_SIZE = 100
MAX_CONCURRENT_PAGES = 15

SCRAPES_TABLE = "scrapes_duplicate"
JOBS_TABLE = "jobs"


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_date(text: str) -> str:
    if not text or text == "NA":
        return None

    text = text.strip()
    lower = text.lower()
    today = datetime.now(timezone.utc)

    match = re.search(r"(\d+)\s+day[s]?\s+ago", lower)
    if match:
        dt = today - timedelta(days=int(match.group(1)))
        return dt.isoformat()

    if "yesterday" in lower:
        dt = today - timedelta(days=1)
        return dt.isoformat()

    if "today" in lower or "just now" in lower:
        return today.isoformat()

    try:
        dt = datetime.strptime(text, "%b %d, %Y")
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except:
        pass

    try:
        dt = datetime.strptime(text, "%d-%m-%Y")
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except:
        pass

    try:
        dt = datetime.strptime(text, "%d-%b-%Y")
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except:
        pass

    return None


def parse_experience_range(exp_text: str):
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


def wait_for_html_ready(page, timeout=30000):
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except:
        pass

    try:
        page.wait_for_function("() => document.readyState === 'complete'", timeout=timeout)
    except:
        pass

    wait_for_any_layout(page, timeout=timeout)


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

    if not jobs_data:
        print("No jobs to process.")
        return

    unique_map = {j["job_url"]: j for j in jobs_data if j.get("job_url") and j["job_url"] != "NA"}
    unique_jobs_list = list(unique_map.values())
    all_scraped_urls = list(unique_map.keys())

    print(f"Checking {len(all_scraped_urls)} jobs against database...")
    existing_urls = set()

    CHECK_BATCH_SIZE = 100
    for i in range(0, len(all_scraped_urls), CHECK_BATCH_SIZE):
        batch_urls = all_scraped_urls[i: i + CHECK_BATCH_SIZE]
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
        return

    for i in range(0, len(new_jobs), BATCH_SIZE):
        batch = new_jobs[i: i + BATCH_SIZE]
        try:
            supabase.table(JOBS_TABLE).insert(batch).execute()
            print(f"Uploaded batch {i} - {i + len(batch)}")
        except Exception as e:
            print(f"Error uploading batch: {e}")


def scrape_darwinbox_discovery(page, url, scrape_uuid):
    jobs_collected = []
    print(f"Scanning: {url}")

    try:
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        wait_for_html_ready(page, timeout=30000)
    except Exception as e:
        print(f"Connection failed: {e}")
        return []

    if page.locator("ui-job-tile").count() > 0:
        print("Detected CARD layout")

        while True:
            load_more = page.locator('xpath=//span[normalize-space()="Load More Jobs"]')
            if not click_if_exists(page, load_more, wait=1500):
                break

        cards = page.locator("ui-job-tile")
        for i in range(cards.count()):
            card = cards.nth(i)

            title = safe_text(card.locator(".job-title"))
            rel_link = safe_attr(card.locator("a.action-btn"), "href")
            dept = safe_text(card.locator('img[src*="department"] >> xpath=following-sibling::span'))

            job_url = urljoin(page.url, rel_link)
            if not rel_link or rel_link == "NA":
                job_url = "NA"

            now_iso = datetime.now(timezone.utc).isoformat()

            jobs_collected.append({
                "created_at": now_iso,
                "updated_at": now_iso,
                "scrape_id": scrape_uuid,
                "title": title,
                "job_url": job_url,
                "is_active": True,
                "department": dept if dept != "NA" else None,
                "location": None,
                "job_type": None,
                "published_date": None,
                "min_exp": None,
                "max_exp": None,
                "original_description": None
            })

    elif page.locator("table.db-table-one").count() > 0:
        print("Detected TABLE layout")
        try:
            page.wait_for_selector("table.db-table-one tbody tr", timeout=15000)
        except:
            pass

        while True:
            rows = page.locator("table.db-table-one tbody tr")
            for i in range(rows.count()):
                row = rows.nth(i)

                title_el = row.locator('td[data-th="Job title"] a')
                dept_el = row.locator('td[data-th="Home Team"] span')
                loc_el = row.locator('td[data-th="Location"] span')
                type_el = row.locator('td[data-th="Employee Type"] span')
                date_el = row.locator('td[data-th="Job posted on"] span')

                title = safe_text(title_el)
                rel_link = safe_attr(title_el, "href")

                dept = safe_text(dept_el)
                location = safe_text(loc_el)
                emp_type = safe_text(type_el)
                posted_raw = safe_text(date_el)

                job_url = urljoin(page.url, rel_link)
                if not rel_link or rel_link == "NA":
                    job_url = "NA"

                now_iso = datetime.now(timezone.utc).isoformat()

                jobs_collected.append({
                    "created_at": now_iso,
                    "updated_at": now_iso,
                    "scrape_id": scrape_uuid,
                    "title": title,
                    "job_url": job_url,
                    "is_active": True,
                    "published_date": normalize_date(posted_raw),
                    "job_type": emp_type if emp_type != "NA" else None,
                    "department": dept if dept != "NA" else None,
                    "location": location if location != "NA" else None,
                    "min_exp": None,
                    "max_exp": None,
                    "original_description": None
                })

            next_btn = page.locator("li.pagination-next:not(.disabled) a")
            if next_btn.count() > 0:
                next_btn.click(force=True)
                page.wait_for_timeout(2000)
            else:
                break
    else:
        print("No recognizable jobs found.")

    return jobs_collected


def run_stage_1_discovery():
    print("\n--- STAGE 1: DISCOVERY (SYNC) ---")
    start_time = time.time()

    df = get_companies()
    if df.empty:
        print("No companies found.")
        return

    if "ats_name" in df.columns:
        df = df[df["ats_name"].astype(str).str.lower().str.strip() == "darwinbox"]

    print(f"Found {len(df)} Darwinbox companies.")

    all_jobs = []
    all_scrapes = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
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
                scrape_record["updated_at"] = finish_iso
                scrape_record["status"] = "success"

                if company_jobs:
                    all_jobs.extend(company_jobs)

            except Exception as e:
                print(f"Failed: {e}")
                finish_iso = datetime.now(timezone.utc).isoformat()
                scrape_record["status"] = "failed"
                scrape_record["finished_at"] = finish_iso
                scrape_record["updated_at"] = finish_iso

            all_scrapes.append(scrape_record)

        browser.close()

    clean_jobs = [j for j in all_jobs if j.get("job_url") and j["job_url"] != "NA"]

    if all_scrapes:
        pd.DataFrame(all_scrapes).to_csv(OUTPUT_SCRAPES_FILE, index=False)
        print(f"Backup saved: {OUTPUT_SCRAPES_FILE}")

    if clean_jobs:
        pd.DataFrame(clean_jobs).to_csv(OUTPUT_JOBS_FILE, index=False)
        print(f"Backup saved (Stage 1 snapshot): {OUTPUT_JOBS_FILE}")

    save_to_supabase(all_scrapes, clean_jobs)

    print(f"Stage 1 Runtime: {time.time() - start_time:.2f}s")


def get_pending_darwinbox_jobs():
    print("Fetching pending Darwinbox jobs from Supabase...")
    try:
        res = (
            supabase.table(JOBS_TABLE)
            .select("id, job_url, location, job_type, department, published_date, scrapes_duplicate(ats_website(ats_name))")
            .is_("original_description", "null")
            .execute()
        )

        all_jobs = res.data or []
        darwin_jobs = [
            j for j in all_jobs
            if j.get("scrapes_duplicate")
            and j["scrapes_duplicate"].get("ats_website")
            and "darwinbox" in str(j["scrapes_duplicate"]["ats_website"].get("ats_name", "")).lower()
        ]

        print(f"Total jobs pending enrichment: {len(darwin_jobs)}")
        return darwin_jobs

    except Exception as e:
        print(f"Error fetching jobs: {e}")
        return []


def update_job_sync(job_id, payload):
    try:
        clean_payload = {k: v for k, v in payload.items() if v is not None and v != ""}
        if clean_payload:
            clean_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            supabase.table(JOBS_TABLE).update(clean_payload).eq("id", job_id).execute()
            print(f"[ID {job_id}] Updated")
    except Exception as e:
        print(f"[ID {job_id}] Update Failed: {e}")


async def extract_details_async(page):
    data = {
        "original_description": None,
        "location": None,
        "job_type": None,
        "department": None,
        "experience_text": None,
        "published_raw": None,
    }

    candidates = [".jd-container", ".job-summary", ".jd", ".job-description"]
    for selector in candidates:
        try:
            if await page.locator(selector).count() > 0:
                text = await page.locator(selector).first.inner_text()
                text = clean_text(text)
                if text and len(text) > 50:
                    data["original_description"] = text
                    break
        except:
            continue

    try:
        snapshot_items = page.locator(".section.mobile-section .grid-item")
        if await snapshot_items.count() > 0:
            for i in range(await snapshot_items.count()):
                try:
                    label = await snapshot_items.nth(i).locator(".label").inner_text()
                    label = clean_text(label).lower()

                    val_loc = snapshot_items.nth(i).locator(".value")
                    val = clean_text(await val_loc.first.inner_text()) if await val_loc.count() > 0 else ""

                    if label == "location":
                        data["location"] = val
                    elif "employee type" in label or label == "type":
                        data["job_type"] = val
                    elif "department" in label or "function" in label:
                        data["department"] = val
                    elif "experience" in label:
                        data["experience_text"] = val
                    elif "updated date" in label or "job posted on" in label:
                        data["published_raw"] = val
                except:
                    continue
    except:
        pass

    try:
        details = page.locator(".job-details .job-details-item")
        if await details.count() > 0:
            for i in range(await details.count()):
                try:
                    label = await details.nth(i).locator("label").first.inner_text()
                    val = await details.nth(i).locator("p").first.inner_text()
                    label, val = clean_text(label).lower(), clean_text(val)

                    if "location" in label:
                        data["location"] = val
                    elif "employee type" in label or "job type" in label or "type" in label:
                        data["job_type"] = val
                    elif "function" in label or "department" in label:
                        data["department"] = val
                    elif "experience range" in label or "experience" in label:
                        data["experience_text"] = val
                    elif "job posted on" in label or "updated date" in label:
                        data["published_raw"] = val
                except:
                    continue
    except:
        pass

    if not data.get("experience_text"):
        try:
            exp_item = page.locator(".job-details-item.experience-range p")
            if await exp_item.count() > 0:
                data["experience_text"] = clean_text(await exp_item.first.inner_text())
        except:
            pass

    return data


async def process_job(sem, context, job):
    async with sem:
        url = job.get("job_url")
        job_id = job.get("id")

        if not url or not str(url).startswith("http"):
            return

        page = await context.new_page()
        try:
            try:
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")
            except:
                pass

            try:
                await page.wait_for_function("() => document.readyState === 'complete'", timeout=30000)
            except:
                pass

            try:
                await page.wait_for_selector(
                    ".jd-container, .job-summary, .jd, .job-description, .job-details, .section.mobile-section",
                    timeout=20000
                )
            except:
                pass

            await asyncio.sleep(1.5)

            scraped = await extract_details_async(page)

            min_exp, max_exp = parse_experience_range(scraped.get("experience_text"))
            published_date = normalize_date(scraped.get("published_raw"))

            payload = {
                "original_description": scraped.get("original_description"),
                "location": scraped.get("location") or job.get("location"),
                "job_type": scraped.get("job_type") or job.get("job_type"),
                "department": scraped.get("department") or job.get("department"),
                "published_date": published_date or job.get("published_date"),
                "min_exp": min_exp,
                "max_exp": max_exp,
            }

            await asyncio.to_thread(update_job_sync, job_id, payload)

        except Exception as e:
            print(f"[ID {job_id}] Failed: {str(e)[:80]}")
        finally:
            await page.close()


async def run_stage_2_enrichment():
    print("\n--- STAGE 2: ENRICHMENT (ASYNC) ---")
    jobs = get_pending_darwinbox_jobs()
    if not jobs:
        print("No pending jobs found.")
        return

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


def export_darwinbox_jobs_backup_from_db():
    print("\nExporting FINAL Darwinbox jobs backup (Stage 1 + Stage 2) to CSV...")

    try:
        res = (
            supabase.table(JOBS_TABLE)
            .select("* , scrapes_duplicate(ats_website(ats_name))")
            .execute()
        )
    except Exception as e:
        print(f"Error exporting jobs: {e}")
        return

    if not res.data:
        print("No jobs found to export.")
        return

    df = pd.DataFrame(res.data)

    if "scrapes_duplicate" in df.columns:
        df["ats_name"] = df["scrapes_duplicate"].apply(
            lambda x: x["ats_website"]["ats_name"] if x and x.get("ats_website") else None
        )

    df = df[df["ats_name"].str.lower().str.contains("darwinbox", na=False)].copy()

    if "scrapes_duplicate" in df.columns:
        df.drop(columns=["scrapes_duplicate"], inplace=True)

    df.to_csv(OUTPUT_JOBS_FILE, index=False)
    print(f"Final backup saved: {OUTPUT_JOBS_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    run_stage_1_discovery()
    asyncio.run(run_stage_2_enrichment())
    export_darwinbox_jobs_backup_from_db()

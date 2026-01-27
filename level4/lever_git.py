import time
import re
import uuid
import asyncio
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = "https://ufnaxahhlblwpdomlybs.supabase.co"
SUPABASE_KEY = "sb_publishable_1d4J1Ll81KwhYPOS40U8mQ_qtCccNsa"

SCRAPES_TABLE = "scrapes_duplicate"
JOBS_TABLE = "jobs_duplicate"

OUTPUT_JOBS_FILE = "lever_jobs.csv"
OUTPUT_SCRAPES_FILE = "lever_scrapes.csv"
FINAL_BACKUP_FILE = "lever_jobs_backup.csv"

INSERT_BATCH_SIZE = 100
CHECK_BATCH_SIZE = 100


class ScraperUtils:
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\u200b", "")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def safe_text(locator) -> str:
        try:
            return locator.first.inner_text().strip() if locator.count() > 0 else "NA"
        except:
            return "NA"

    @staticmethod
    def safe_attr(locator, attr) -> str:
        try:
            val = locator.first.get_attribute(attr) if locator.count() > 0 else None
            return val.strip() if val else "NA"
        except:
            return "NA"

    @staticmethod
    def normalize_job_type(text: str):
        if not text:
            return None

        t = ScraperUtils.clean_text(text)
        t = t.replace("/", "").strip()
        t_low = t.lower()

        mapping = {
            "full time": "Full time",
            "full-time": "Full time",
            "part time": "Part time",
            "part-time": "Part time",
            "contract": "Contract",
            "internship": "Internship",
            "temporary": "Temporary",
            "freelance": "Freelance",
        }

        return mapping.get(t_low, t)

    @staticmethod
    def format_date_iso(date_str: str):
        if not date_str:
            return None

        clean_str = ScraperUtils.clean_text(date_str)

        formats = [
            "%B %d, %Y",
            "%b %d, %Y",
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%m/%d/%Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(clean_str, fmt).replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except:
                continue

        return None


class SupabaseManager:
    def __init__(self, url=SUPABASE_URL, key=SUPABASE_KEY):
        self.client: Client = create_client(url, key)

    def fetch_companies(self, ats_name="lever") -> pd.DataFrame:
        try:
            res = self.client.table("ats_website").select("*").execute()
            df = pd.DataFrame(res.data)
            if not df.empty and "ats_name" in df.columns:
                return df[df["ats_name"].astype(str).str.lower().str.strip() == ats_name]
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching companies: {e}")
            return pd.DataFrame()

    def fetch_pending_jobs(self, ats_filter="Lever"):
        try:
            print(f"Fetching pending {ats_filter} jobs...")
            res = (
                self.client.table(JOBS_TABLE)
                .select("id, job_url, location, job_type, department, scrapes_duplicate(ats_website(ats_name))")
                .is_("original_description", "null")
                .execute()
            )

            all_data = res.data or []
            return [
                j
                for j in all_data
                if j.get("scrapes_duplicate")
                and ats_filter.lower() in j["scrapes_duplicate"]["ats_website"]["ats_name"].lower()
            ]
        except Exception as e:
            print(f"Error fetching pending jobs: {e}")
            return []

    def save_scrapes(self, scrapes_data):
        if not scrapes_data:
            return
        try:
            print(f"Uploading {len(scrapes_data)} scrape records...")
            self.client.table(SCRAPES_TABLE).upsert(scrapes_data).execute()
            print("Scrape records uploaded.")
        except Exception as e:
            print(f"Error uploading scrapes: {e}")

    def save_jobs_deduplicated(self, jobs_data):
        if not jobs_data:
            print("No jobs to process.")
            return

        unique_map = {j["job_url"]: j for j in jobs_data if j.get("job_url") and j["job_url"] != "NA"}
        unique_jobs_list = list(unique_map.values())
        all_urls = list(unique_map.keys())

        print(f"Checking {len(all_urls)} jobs against database...")
        existing_urls = set()

        for i in range(0, len(all_urls), CHECK_BATCH_SIZE):
            batch = all_urls[i : i + CHECK_BATCH_SIZE]
            try:
                res = self.client.table(JOBS_TABLE).select("job_url").in_("job_url", batch).execute()
                for row in res.data:
                    existing_urls.add(row["job_url"])
            except Exception as e:
                print(f"Duplicate check error: {e}")

        new_jobs = [j for j in unique_jobs_list if j["job_url"] not in existing_urls]
        print(f"Inserting {len(new_jobs)} new jobs ({len(unique_jobs_list) - len(new_jobs)} skipped).")

        for i in range(0, len(new_jobs), INSERT_BATCH_SIZE):
            batch = new_jobs[i : i + INSERT_BATCH_SIZE]
            try:
                self.client.table(JOBS_TABLE).insert(batch).execute()
                print(f"Uploaded batch {i}-{i+len(batch)}")
            except Exception as e:
                print(f"Error inserting batch: {e}")

    def bulk_update_jobs(self, updates, batch_size=50):
        if not updates:
            return

        try:
            for i in range(0, len(updates), batch_size):
                batch = updates[i : i + batch_size]
                self.client.table(JOBS_TABLE).upsert(batch).execute()
                print(f"Bulk updated batch {i}-{i+len(batch)}")
        except Exception as e:
            print(f"Bulk update failed: {e}")


class LeverDiscovery:
    def __init__(self, db_manager: SupabaseManager):
        self.db = db_manager

    def scrape_site(self, page, url, scrape_uuid):
        jobs_collected = []
        print(f"Scanning: {url}")

        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")

            try:
                page.wait_for_selector('xpath=//div[contains(@class,"posting")]', timeout=15000)
            except:
                print("No job list found or structure changed.")
                return []

            jobs = page.locator('xpath=//div[contains(@class,"posting")]')
            count = jobs.count()
            print(f"Found {count} jobs")

            for i in range(count):
                job = jobs.nth(i)

                title_el = job.locator(
                    'xpath=./a[contains(@class,"posting-title")]//h5[@data-qa="posting-name"]'
                )
                link_el = job.locator('xpath=./a[contains(@class,"posting-title")]')
                type_el = job.locator('xpath=.//span[contains(@class,"commitment")]')
                loc_el = job.locator('xpath=.//span[contains(@class,"location")]')

                title = ScraperUtils.safe_text(title_el)
                raw_href = ScraperUtils.safe_attr(link_el, "href")
                job_type = ScraperUtils.safe_text(type_el)
                location = ScraperUtils.safe_text(loc_el)

                if title == "NA" or raw_href == "NA":
                    continue

                job_url = urljoin(url, raw_href)
                now_iso = datetime.now(timezone.utc).isoformat()

                jobs_collected.append(
                    {
                        "created_at": now_iso,
                        "updated_at": now_iso,
                        "scrape_id": scrape_uuid,
                        "title": title,
                        "job_url": job_url,
                        "is_active": True,
                        "published_date": None,
                        "job_type": ScraperUtils.normalize_job_type(job_type) if job_type != "NA" else None,
                        "department": None,
                        "location": location if location != "NA" else None,
                        "original_description": None,
                    }
                )

        except Exception as e:
            print(f"Error during scraping: {e}")
            return []

        return jobs_collected

    def run(self):
        print("\n--- STAGE 1: DISCOVERY (LEVER) ---")
        start_time = time.time()

        companies = self.db.fetch_companies(ats_name="lever")
        print(f"Found {len(companies)} Lever companies.")

        all_jobs = []
        all_scrapes = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            for _, row in companies.iterrows():
                scrape_uuid = str(uuid.uuid4())
                now_iso = datetime.now(timezone.utc).isoformat()

                scrape_record = {
                    "id": scrape_uuid,
                    "ats_website_id": row["id"],
                    "status": "pending",
                    "created_at": now_iso,
                    "updated_at": now_iso,
                    "finished_at": None,
                }

                try:
                    jobs = self.scrape_site(page, row["ats_url"], scrape_uuid)
                    scrape_record["status"] = "success" if jobs else "failed"
                    scrape_record["finished_at"] = datetime.now(timezone.utc).isoformat()
                    if jobs:
                        all_jobs.extend(jobs)
                except Exception as e:
                    print(f"Failed: {e}")
                    scrape_record["status"] = "failed"
                    scrape_record["finished_at"] = datetime.now(timezone.utc).isoformat()

                all_scrapes.append(scrape_record)

            browser.close()

        if all_scrapes:
            pd.DataFrame(all_scrapes).to_csv(OUTPUT_SCRAPES_FILE, index=False)
            print(f"Stage-1 scrapes snapshot saved: {OUTPUT_SCRAPES_FILE}")

        if all_jobs:
            pd.DataFrame(all_jobs).to_csv(OUTPUT_JOBS_FILE, index=False)
            print(f"Stage-1 jobs snapshot saved: {OUTPUT_JOBS_FILE}")

        clean_jobs = [j for j in all_jobs if j.get("job_url") and j["job_url"] != "NA"]
        self.db.save_scrapes(all_scrapes)
        self.db.save_jobs_deduplicated(clean_jobs)

        print(f"Stage 1 Runtime: {time.time() - start_time:.2f}s")
        print("--- STAGE 1 COMPLETE ---")


class LeverEnrichment:
    def __init__(self, db_manager: SupabaseManager, concurrency=8):
        self.db = db_manager
        self.concurrency = concurrency

    async def _extract_details(self, page):
        data = {
            "original_description": None,
            "location": None,
            "team": None,
            "employment_type": None,
            "workplace_type": None,
            "experience": "",
            "date_opened": "",
        }

        try:
            desc_el = page.locator('div.section.page-centered[data-qa="job-description"]')
            if await desc_el.count() > 0:
                data["original_description"] = ScraperUtils.clean_text(await desc_el.inner_text())
        except:
            pass

        try:
            loc_el = page.locator("div.posting-headline div.posting-category.location")
            if await loc_el.count() > 0:
                data["location"] = ScraperUtils.clean_text(await loc_el.first.inner_text())

            dept_el = page.locator("div.posting-headline div.posting-category.department")
            if await dept_el.count() > 0:
                data["team"] = ScraperUtils.clean_text(await dept_el.first.inner_text()).replace("/", "").strip()

            type_el = page.locator("div.posting-headline div.posting-category.commitment")
            if await type_el.count() > 0:
                raw = ScraperUtils.clean_text(await type_el.first.inner_text())
                data["employment_type"] = raw.replace("/", "").strip()

            work_el = page.locator("div.posting-headline div.posting-category.workplaceTypes")
            if await work_el.count() > 0:
                data["workplace_type"] = ScraperUtils.clean_text(await work_el.first.inner_text())

        except:
            pass

        try:
            info_items = page.locator('ul.posting-requirements li, [data-qa="job-info"] li')
            count = await info_items.count()

            for i in range(count):
                try:
                    item = info_items.nth(i)

                    label_loc = item.locator("h5, .posting-category-title")
                    value_loc = item.locator("span, .posting-category-value")

                    if await label_loc.count() == 0 or await value_loc.count() == 0:
                        continue

                    label = (await label_loc.first.inner_text()).strip().lower()
                    value = (await value_loc.first.inner_text()).strip()

                    if "location" in label and not data["location"]:
                        data["location"] = value
                    elif ("team" in label or "department" in label) and not data["team"]:
                        data["team"] = value
                    elif (
                        "employment" in label or "type" in label or "commitment" in label
                    ) and not data["employment_type"]:
                        data["employment_type"] = value
                    elif "experience" in label and not data["experience"]:
                        data["experience"] = value
                    elif ("date" in label or "opened" in label) and not data["date_opened"]:
                        data["date_opened"] = value

                except:
                    continue
        except:
            pass

        return data

    async def _process_job_with_page(self, page, job):
        url = job.get("job_url")
        job_id = job.get("id")
        if not url or not job_id:
            return None

        try:
            await page.goto(url, timeout=45000, wait_until="domcontentloaded")

            try:
                await page.wait_for_selector('div.section.page-centered[data-qa="job-description"]', timeout=8000)
            except:
                pass

            scraped = await self._extract_details(page)

            final_job_type = ScraperUtils.normalize_job_type(scraped.get("employment_type") or job.get("job_type"))
            final_location = scraped.get("location") or job.get("location") or None
            final_department = scraped.get("team") or job.get("department") or None

            payload = {
                "id": job_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "original_description": scraped.get("original_description"),
                "location": final_location,
                "job_type": final_job_type,
                "department": final_department,
            }

            if payload["original_description"]:
                return payload

            print(f"No description found ID {job_id}")
            return None

        except Exception as e:
            print(f"Failed ID {job_id}: {str(e)[:80]}")
            return None

    async def _worker(self, context, queue: asyncio.Queue, results: list):
        page = await context.new_page()
        try:
            while True:
                job = await queue.get()
                if job is None:
                    queue.task_done()
                    break

                payload = await self._process_job_with_page(page, job)
                if payload:
                    results.append(payload)

                queue.task_done()
        finally:
            await page.close()

    async def run_async(self):
        print("\n--- STAGE 2: ENRICHMENT (LEVER) ---")
        start_time = time.time()

        jobs = self.db.fetch_pending_jobs(ats_filter="Lever")
        if not jobs:
            print("No jobs pending enrichment.")
            return

        print(f"Enriching {len(jobs)} jobs with concurrency={self.concurrency} ...")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )

            async def block_resources(route):
                if route.request.resource_type in ["image", "media", "font"]:
                    await route.abort()
                else:
                    await route.continue_()

            await context.route("**/*", block_resources)

            queue = asyncio.Queue()
            for job in jobs:
                queue.put_nowait(job)

            for _ in range(self.concurrency):
                queue.put_nowait(None)

            results = []
            workers = [asyncio.create_task(self._worker(context, queue, results)) for _ in range(self.concurrency)]

            await queue.join()
            await asyncio.gather(*workers)
            await browser.close()

        print(f"\nUploading {len(results)} enriched jobs in bulk...")
        self.db.bulk_update_jobs(results, batch_size=50)

        print(f"Stage 2 Runtime: {time.time() - start_time:.2f}s")
        print("Enrichment completed.")
        print("--- STAGE 2 COMPLETE ---")

    def run(self):
        asyncio.run(self.run_async())


def export_lever_jobs_backup_from_db(db: SupabaseManager):
    print("\nExporting FINAL Lever jobs backup (Stage 1 + Stage 2 combined) to CSV...")

    try:
        res = db.client.table(JOBS_TABLE).select("* , scrapes_duplicate(ats_website(ats_name))").execute()
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

    df = df[df["ats_name"].astype(str).str.lower().str.contains("lever", na=False)].copy()

    if "scrapes_duplicate" in df.columns:
        df.drop(columns=["scrapes_duplicate"], inplace=True)

    df.to_csv(FINAL_BACKUP_FILE, index=False)
    print(f"Final backup saved: {FINAL_BACKUP_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    start = time.time()

    db_manager = SupabaseManager()

    discovery = LeverDiscovery(db_manager)
    discovery.run()

    enrichment = LeverEnrichment(db_manager, concurrency=8)
    enrichment.run()

    export_lever_jobs_backup_from_db(db_manager)

    print(f"\nTotal Runtime: {time.time() - start:.2f}s")

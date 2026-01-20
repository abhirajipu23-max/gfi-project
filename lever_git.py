import time
import re
import uuid
import asyncio
import pandas as pd
from datetime import datetime, timezone

from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = "https://ufnaxahhlblwpdomlybs.supabase.co"
SUPABASE_KEY = "sb_publishable_1d4J1Ll81KwhYPOS40U8mQ_qtCccNsa"


# ================= UTILS CLASS =================
class ScraperUtils:
    """Static helper methods for text processing, date parsing, and DOM safety."""

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
    def parse_experience(exp_text):
        """Parses experience text into (min_exp, max_exp)."""
        if not exp_text:
            return None, None
        exp_text = str(exp_text).lower().strip()

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

    @staticmethod
    def detect_work_mode(text):
        """Checks for Remote, Hybrid, or defaults to Onsite."""
        if not text:
            return "Onsite"
        text_lower = text.lower()
        if "remote" in text_lower:
            return "Remote"
        if "hybrid" in text_lower:
            return "Hybrid"
        return "Onsite"

    @staticmethod
    def format_date(date_str):
        """Attempts to standardize date for DB (YYYY-MM-DD)."""
        if not date_str:
            return None
        clean_str = date_str.strip()
        formats = ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d-%m-%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(clean_str, fmt).strftime("%Y-%m-%d")
            except:
                continue
        return None


# ================= DATABASE MANAGER =================
class SupabaseManager:
    """Handles all interactions with Supabase."""

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
        """Fetches jobs from jobs_duplicate that need enrichment."""
        try:
            print(f"Fetching pending {ats_filter} jobs...")
            res = (
                self.client.table("jobs_duplicate")
                .select(
                    "id, job_url, location, job_type, department, scrapes_duplicate(ats_website(ats_name))"
                )
                .is_("original_description", "null")
                .execute()
            )

            return [
                j
                for j in res.data
                if j.get("scrapes_duplicate")
                and ats_filter.lower()
                in j["scrapes_duplicate"]["ats_website"]["ats_name"].lower()
            ]
        except Exception as e:
            print(f"Error fetching pending jobs: {e}")
            return []

    def save_scrapes(self, scrapes_data):
        if not scrapes_data:
            return
        try:
            print(f"Uploading {len(scrapes_data)} scrape records...")
            self.client.table("scrapes_duplicate").upsert(scrapes_data).execute()
            print("✅ Scrape records uploaded.")
        except Exception as e:
            print(f"❌ Error uploading scrapes: {e}")

    def save_jobs_deduplicated(self, jobs_data):
        if not jobs_data:
            print("No jobs to process.")
            return

        unique_map = {j["job_url"]: j for j in jobs_data if j["job_url"] and j["job_url"] != "NA"}
        unique_jobs_list = list(unique_map.values())
        all_urls = list(unique_map.keys())

        print(f"Checking {len(all_urls)} jobs against database...")
        existing_urls = set()
        batch_size = 100

        for i in range(0, len(all_urls), batch_size):
            batch = all_urls[i : i + batch_size]
            try:
                res = self.client.table("jobs_duplicate").select("job_url").in_("job_url", batch).execute()
                for row in res.data:
                    existing_urls.add(row["job_url"])
            except Exception as e:
                print(f"   ⚠️ Duplicate check error: {e}")

        new_jobs = [j for j in unique_jobs_list if j["job_url"] not in existing_urls]
        print(f"   -> Inserting {len(new_jobs)} new jobs ({len(unique_jobs_list) - len(new_jobs)} skipped).")

        for i in range(0, len(new_jobs), batch_size):
            batch = new_jobs[i : i + batch_size]
            try:
                self.client.table("jobs_duplicate").insert(batch).execute()
                print(f"   ↳ Uploaded batch {i}-{i+len(batch)}")
            except Exception as e:
                print(f"   ❌ Error inserting batch: {e}")

    def bulk_update_jobs(self, updates, batch_size=50):
        """Fast bulk update using upsert on jobs_duplicate."""
        if not updates:
            return

        try:
            for i in range(0, len(updates), batch_size):
                batch = updates[i : i + batch_size]
                self.client.table("jobs_duplicate").upsert(batch).execute()
                print(f"   ✅ Bulk updated batch {i}-{i+len(batch)}")
        except Exception as e:
            print(f"Bulk update failed: {e}")


# ================= CLASS: STAGE 1 (DISCOVERY) =================
class LeverDiscovery:
    """Class responsible for scraping Lever job lists."""

    def __init__(self, db_manager: SupabaseManager):
        self.db = db_manager

    def scrape_site(self, page, url, scrape_uuid):
        jobs_collected = []
        print(f"   Scanning: {url}")

        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")

            try:
                page.wait_for_selector('xpath=//div[contains(@class,"posting")]', timeout=15000)
            except:
                print("     ⚠️ No job list found or structure changed.")
                return []

            jobs = page.locator('xpath=//div[contains(@class,"posting")]')
            count = jobs.count()
            print(f"     ↳ Found {count} jobs")

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

                if not title or not raw_href or raw_href == "NA":
                    continue

                jobs_collected.append(
                    {
                        "id": uuid.uuid4().int % (2**63 - 1),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "scrape_id": scrape_uuid,
                        "title": title,
                        "job_url": raw_href,
                        "is_active": True,
                        "published_date": None,
                        "job_type": job_type if job_type != "NA" else None,
                        "department": None,
                        "location": location if location != "NA" else None,
                        "original_description": None,
                    }
                )

        except Exception as e:
            print(f"     ❌ Error during scraping: {e}")
            return []

        return jobs_collected

    def run(self):
        print("\n--- STAGE 1: DISCOVERY (SYNC) ---")
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
                scrape_record = {
                    "id": scrape_uuid,
                    "ats_website_id": row["id"],
                    "status": "pending",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }

                try:
                    jobs = self.scrape_site(page, row["ats_url"], scrape_uuid)
                    scrape_record["status"] = "success"
                    scrape_record["finished_at"] = datetime.now(timezone.utc).isoformat()
                    if jobs:
                        all_jobs.extend(jobs)
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    scrape_record["status"] = "failed"
                    scrape_record["finished_at"] = datetime.now(timezone.utc).isoformat()

                all_scrapes.append(scrape_record)

            browser.close()

        clean_jobs = [j for j in all_jobs if j["job_url"] and j["job_url"] != "NA"]
        self.db.save_scrapes(all_scrapes)
        self.db.save_jobs_deduplicated(clean_jobs)
        print(f"Stage 1 Runtime: {time.time() - start_time:.2f}s")


# ================= CLASS: STAGE 2 (ENRICHMENT) =================
class LeverEnrichment:
    """Optimized enrichment using page worker pool + bulk DB update."""

    def __init__(self, db_manager: SupabaseManager, concurrency=8):
        self.db = db_manager
        self.concurrency = concurrency

    async def _extract_details(self, page):
        data = {
            "original_description": None,
            "location": "",
            "team": None,
            "employment_type": "",
            "experience": "",
            "date_opened": "",
        }

        # Description
        try:
            desc_el = page.locator('[data-qa="job-description"]')
            if await desc_el.count() > 0:
                data["original_description"] = ScraperUtils.clean_text(await desc_el.inner_text())
        except:
            pass

        # Meta info
        try:
            info_items = page.locator('ul.posting-requirements li, [data-qa="job-info"] li')
            count = await info_items.count()

            for i in range(count):
                try:
                    item = info_items.nth(i)
                    label = (
                        await item.locator("h5, .posting-category-title").first.inner_text()
                    ).strip().lower()
                    value = (await item.locator("span, .posting-category-value").first.inner_text()).strip()

                    if "location" in label:
                        data["location"] = value
                    elif "team" in label or "department" in label:
                        data["team"] = value
                    elif "employment" in label or "type" in label or "commitment" in label:
                        data["employment_type"] = value
                    elif "experience" in label:
                        data["experience"] = value
                    elif "date" in label or "opened" in label:
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

            # Wait only if needed (NO fixed sleep)
            try:
                await page.wait_for_selector('[data-qa="job-description"]', timeout=8000)
            except:
                pass

            scraped = await self._extract_details(page)

            min_exp, max_exp = ScraperUtils.parse_experience(scraped["experience"])
            published_date = ScraperUtils.format_date(scraped["date_opened"])

            raw_loc = scraped["location"] or job.get("location") or ""
            raw_type = scraped["employment_type"] or job.get("job_type") or ""
            work_mode = ScraperUtils.detect_work_mode(f"{raw_loc} {raw_type}")

            payload = {
                "id": job_id,
                "updated_at": datetime.utcnow().isoformat(),
                "original_description": scraped["original_description"],
                "location": raw_loc if raw_loc else None,
                "job_type": work_mode,
                "department": scraped["team"] or job.get("department"),
                "min_exp": min_exp,
                "max_exp": max_exp,
                "published_date": published_date,
            }

            if payload["original_description"]:
                return payload

            print(f"⚠️ [ID {job_id}] No description found.")
            return None

        except Exception as e:
            print(f"❌ [ID {job_id}] Failed: {str(e)[:80]}")
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
        print("\n--- STAGE 2: ENRICHMENT (ASYNC OPTIMIZED) ---")
        start_time = time.time()

        jobs = self.db.fetch_pending_jobs(ats_filter="Lever")
        if not jobs:
            print("No jobs pending enrichment.")
            return

        print(f"Enriching {len(jobs)} jobs with concurrency={self.concurrency} ...")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            # Block heavy resources for speed
            async def block_resources(route):
                if route.request.resource_type in ["image", "media", "font"]:
                    await route.abort()
                else:
                    await route.continue_()

            await context.route("**/*", block_resources)

            queue = asyncio.Queue()
            for job in jobs:
                queue.put_nowait(job)

            # Stop signals
            for _ in range(self.concurrency):
                queue.put_nowait(None)

            results = []
            workers = [
                asyncio.create_task(self._worker(context, queue, results))
                for _ in range(self.concurrency)
            ]

            await queue.join()
            await asyncio.gather(*workers)

            await browser.close()

        # Bulk update DB
        print(f"\nUploading {len(results)} enriched jobs in bulk...")
        self.db.bulk_update_jobs(results, batch_size=50)

        print(f"Stage 2 Runtime: {time.time() - start_time:.2f}s")
        print("✅ Enrichment completed.")

    def run(self):
        asyncio.run(self.run_async())


# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    db_manager = SupabaseManager()

    # Stage 1
    discovery = LeverDiscovery(db_manager)
    discovery.run()

    # Stage 2 (optimized)
    enrichment = LeverEnrichment(db_manager, concurrency=8)
    enrichment.run()

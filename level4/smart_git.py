import time
import re
import uuid
import asyncio
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urljoin

# Playwright
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

# Database
from dotenv import load_dotenv
from supabase import create_client, Client

# ================= CONFIGURATION =================
load_dotenv()
SUPABASE_URL = "https://ufnaxahhlblwpdomlybs.supabase.co"
SUPABASE_KEY = "sb_publishable_1d4J1Ll81KwhYPOS40U8mQ_qtCccNsa"

JOBS_TABLE = "jobs_duplicate"
SCRAPES_TABLE = "scrapes_duplicate"
FINAL_BACKUP_FILE = "smartrecruiters_jobs_backup.csv"


# ================= UTILS CLASS =================
class ScraperUtils:
    """Static helper methods for text processing and DOM safety."""

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


# ================= DATABASE MANAGER =================
class SupabaseManager:
    """Handles all interactions with Supabase."""

    def __init__(self, url=SUPABASE_URL, key=SUPABASE_KEY):
        self.client: Client = create_client(url, key)

    def fetch_companies(self, ats_name="smartrecruiters") -> pd.DataFrame:
        try:
            res = self.client.table("ats_website").select("*").execute()
            df = pd.DataFrame(res.data)
            if not df.empty and "ats_name" in df.columns:
                return df[df["ats_name"].astype(str).str.lower().str.strip() == ats_name]
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching companies: {e}")
            return pd.DataFrame()

    def fetch_pending_jobs(self, ats_filter="SmartRecruiters"):
        """Fetches jobs from jobs_duplicate that need enrichment."""
        try:
            print("Fetching pending SmartRecruiters jobs...")
            res = (
                self.client.table(JOBS_TABLE)
                .select("id, job_url, location, job_type, department, scrapes_duplicate(ats_website(ats_name))")
                .is_("original_description", "null")
                .execute()
            )
            return [
                j
                for j in res.data
                if j.get("scrapes_duplicate") and ats_filter in j["scrapes_duplicate"]["ats_website"]["ats_name"]
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
                res = self.client.table(JOBS_TABLE).select("job_url").in_("job_url", batch).execute()
                for row in res.data:
                    existing_urls.add(row["job_url"])
            except Exception as e:
                print(f"   ⚠️ Duplicate check error: {e}")

        new_jobs = [j for j in unique_jobs_list if j["job_url"] not in existing_urls]
        print(f"   -> Inserting {len(new_jobs)} new jobs ({len(unique_jobs_list) - len(new_jobs)} skipped).")

        for i in range(0, len(new_jobs), batch_size):
            batch = new_jobs[i : i + batch_size]
            try:
                self.client.table(JOBS_TABLE).insert(batch).execute()
                print(f"   ↳ Uploaded batch {i}-{i+len(batch)}")
            except Exception as e:
                print(f"   ❌ Error inserting batch: {e}")

    def update_job(self, job_id, payload):
        try:
            clean_payload = {k: v for k, v in payload.items() if v}
            if clean_payload:
                clean_payload["updated_at"] = datetime.utcnow().isoformat()
                self.client.table(JOBS_TABLE).update(clean_payload).eq("id", job_id).execute()
                print(f"   ✅ [ID {job_id}] Updated")
        except Exception as e:
            print(f"   ❌ [ID {job_id}] Update Failed: {e}")


# ================= BACKUP EXPORT (Stage-1 + Stage-2 combined) =================
def export_smartrecruiters_jobs_backup_from_db(db: SupabaseManager):
    """
    Final backup export from DB so it contains BOTH:
      - Stage-1 discovered jobs
      - Stage-2 enriched jobs
    """
    print("\n📦 Exporting SmartRecruiters FINAL backup CSV (Stage-1 + Stage-2 combined)...")

    try:
        res = db.client.table(JOBS_TABLE).select("* , scrapes_duplicate(ats_website(ats_name))").execute()
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return

    if not res.data:
        print("⚠️ No jobs found in DB to export.")
        return

    df = pd.DataFrame(res.data)

    # Extract ats_name from nested join
    if "scrapes_duplicate" in df.columns:
        df["ats_name"] = df["scrapes_duplicate"].apply(
            lambda x: x["ats_website"]["ats_name"] if x and x.get("ats_website") else None
        )

    # Filter only SmartRecruiters jobs
    df = df[df["ats_name"].astype(str).str.lower().str.contains("smartrecruiters", na=False)].copy()

    # Drop nested JSON column
    if "scrapes_duplicate" in df.columns:
        df.drop(columns=["scrapes_duplicate"], inplace=True)

    df.to_csv(FINAL_BACKUP_FILE, index=False)
    print(f"✅ Backup saved: {FINAL_BACKUP_FILE} | Rows: {len(df)}")


# ================= CLASS: STAGE 1 (DISCOVERY) =================
class SmartRecruitersDiscovery:
    """Class responsible for scraping SmartRecruiters job lists."""

    def __init__(self, db_manager: SupabaseManager):
        self.db = db_manager

    def scrape_site(self, page, url, scrape_uuid):
        jobs_collected = []
        print(f"   Scanning: {url}")

        try:
            page.goto(url, timeout=60000)

            # Wait for main container
            try:
                page.wait_for_selector('xpath=//div[contains(@class,"openings-body")]', timeout=15000)
            except:
                print("     ⚠️ No job list found (or layout diff).")
                return []

            # Sections (usually by Location)
            sections = page.locator('xpath=//section[contains(@class,"openings-section")]')
            if sections.count() == 0:
                print("     ⚠️ No sections found")
                return []

            print(f"     ↳ Found {sections.count()} location sections")

            for i in range(sections.count()):
                section = sections.nth(i)

                job_items = section.locator('xpath=.//li[contains(@class,"opening-job")]')

                for j in range(job_items.count()):
                    job = job_items.nth(j)

                    title = ScraperUtils.safe_text(job.locator('xpath=.//h4[contains(@class,"job-title")]'))
                    link = ScraperUtils.safe_attr(job.locator('xpath=.//a[contains(@class,"details")]'), "href")
                    jtype = ScraperUtils.safe_text(job.locator('xpath=.//p[contains(@class,"job-desc")]//span'))

                    if title == "NA" or link == "NA":
                        continue

                    jobs_collected.append(
                        {
                            "id": uuid.uuid4().int % (2**63 - 1),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "scrape_id": scrape_uuid,
                            "title": title,
                            "job_url": link,
                            "is_active": True,
                            "published_date": None,
                            "job_type": jtype if jtype != "NA" else None,
                            "department": None,
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
        companies = self.db.fetch_companies(ats_name="smartrecruiters")
        print(f"Found {len(companies)} SmartRecruiters companies.")

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
class SmartRecruitersEnrichment:
    """Class responsible for enriching SmartRecruiters job details."""

    def __init__(self, db_manager: SupabaseManager, concurrency=10):
        self.db = db_manager
        self.concurrency = concurrency

    async def _extract_details(self, page):
        data = {"original_description": None, "location": None, "job_type": None, "department": None}

        # 1. Location (Often loads dynamically via JS)
        try:
            await page.wait_for_selector(".c-spl-job-location__place", state="attached", timeout=4000)
            loc_el = page.locator(".c-spl-job-location__place").first
            if await loc_el.count() > 0:
                data["location"] = ScraperUtils.clean_text(await loc_el.inner_text())
        except:
            # Fallback: Meta tag
            try:
                meta_loc = page.locator('li[itemprop="jobLocation"]').first
                if await meta_loc.count() > 0:
                    raw = await meta_loc.inner_text()
                    data["location"] = ScraperUtils.clean_text(raw.replace("Location", ""))
            except:
                pass

        # 2. Description (Combine sections)
        try:
            desc_parts = []

            for sec_id, title in [
                ("#st-companyDescription .wysiwyg", "### Company Description"),
                ("#st-jobDescription .wysiwyg", "### Job Description"),
                ("#st-qualifications .wysiwyg", "### Qualifications"),
                ("#st-additionalInformation .wysiwyg", "### Additional Info"),
            ]:
                el = page.locator(sec_id)
                if await el.count() > 0:
                    text = ScraperUtils.clean_text(await el.inner_text())
                    if text:
                        desc_parts.append(f"{title}\n{text}")

            if desc_parts:
                data["original_description"] = "\n\n".join(desc_parts)
            else:
                el_gen = page.locator('[itemprop="description"]')
                if await el_gen.count() > 0:
                    data["original_description"] = ScraperUtils.clean_text(await el_gen.inner_text())
        except:
            pass

        # 3. Meta Data
        try:
            dept_meta = page.locator('meta[itemprop="industry"]').first
            if await dept_meta.count() > 0:
                data["department"] = await dept_meta.get_attribute("content")

            type_el = page.locator('[itemprop="employmentType"]').first
            if await type_el.count() > 0:
                data["job_type"] = ScraperUtils.clean_text(await type_el.inner_text())
        except:
            pass

        return data

    async def _process_job(self, sem, context, job):
        async with sem:
            url = job.get("job_url")
            job_id = job.get("id")
            if not url:
                return

            page = await context.new_page()
            try:
                await page.goto(url, timeout=45000, wait_until="domcontentloaded")

                scraped = await self._extract_details(page)

                payload = {
                    "original_description": scraped["original_description"],
                    "location": scraped["location"] or job.get("location"),
                    "job_type": scraped["job_type"] or job.get("job_type"),
                    "department": scraped["department"] or job.get("department"),
                }

                if payload["original_description"]:
                    await asyncio.to_thread(self.db.update_job, job_id, payload)
                else:
                    print(f"   ⚠️ [ID {job_id}] No description found.")

            except Exception as e:
                print(f"   ❌ [ID {job_id}] Failed: {str(e)[:50]}")
            finally:
                await page.close()

    async def run_async(self):
        print("\n--- STAGE 2: ENRICHMENT (ASYNC) ---")
        jobs = self.db.fetch_pending_jobs(ats_filter="SmartRecruiters")
        if not jobs:
            print("No jobs pending enrichment.")
            return

        print(f"Enriching {len(jobs)} jobs with concurrency {self.concurrency}...")
        sem = asyncio.Semaphore(self.concurrency)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            tasks = [self._process_job(sem, context, job) for job in jobs]
            await asyncio.gather(*tasks)

            await browser.close()

        print("Enrichment completed.")

    def run(self):
        asyncio.run(self.run_async())


# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    db_manager = SupabaseManager()

    # Run Stage 1
    discovery = SmartRecruitersDiscovery(db_manager)
    discovery.run()

    # Run Stage 2
    enrichment = SmartRecruitersEnrichment(db_manager, concurrency=10)
    enrichment.run()

    # ✅ Final CSV Backup (Stage-1 + Stage-2 combined)
    export_smartrecruiters_jobs_backup_from_db(db_manager)

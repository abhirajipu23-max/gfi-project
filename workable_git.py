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
from bs4 import BeautifulSoup

# Database
from dotenv import load_dotenv
from supabase import create_client, Client

# ================= CONFIGURATION =================
load_dotenv()
SUPABASE_URL = "https://ufnaxahhlblwpdomlybs.supabase.co"
SUPABASE_KEY = "sb_publishable_1d4J1Ll81KwhYPOS40U8mQ_qtCccNsa"

# ================= UTILS CLASS =================
class ScraperUtils:
    """Static helper methods for text processing and DOM safety."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        if not text: return ""
        text = text.replace('\u200b', '')
        return re.sub(r'\s+', ' ', text).strip()

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

    def fetch_companies(self, ats_name="workable") -> pd.DataFrame:
        try:
            res = self.client.table("ats_website").select("*").execute()
            df = pd.DataFrame(res.data)
            if not df.empty and "ats_name" in df.columns:
                return df[df["ats_name"].astype(str).str.lower().str.strip() == ats_name]
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching companies: {e}")
            return pd.DataFrame()

    def fetch_pending_jobs(self, ats_filter="Workable"):
        """Fetches jobs from jobs_duplicate that need enrichment."""
        try:
            print("Fetching pending Workable jobs...")
            res = (
                self.client.table("jobs_duplicate")
                .select("id, job_url, location, job_type, department, scrapes_duplicate(ats_website(ats_name))")
                .is_("original_description", "null")
                .execute()
            )
            return [
                j for j in res.data 
                if j.get("scrapes_duplicate") and ats_filter in j["scrapes_duplicate"]["ats_website"]["ats_name"]
            ]
        except Exception as e:
            print(f"Error fetching pending jobs: {e}")
            return []

    def save_scrapes(self, scrapes_data):
        if not scrapes_data: return
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

    def update_job(self, job_id, payload):
        try:
            clean_payload = {k: v for k, v in payload.items() if v}
            if clean_payload:
                clean_payload["updated_at"] = datetime.utcnow().isoformat()
                self.client.table("jobs_duplicate").update(clean_payload).eq("id", job_id).execute()
                print(f"   ✅ [ID {job_id}] Updated")
        except Exception as e:
            print(f"   ❌ [ID {job_id}] Update Failed: {e}")

# ================= CLASS: STAGE 1 (DISCOVERY) =================
class WorkableDiscovery:
    """Class responsible for scraping Workable job lists."""

    def __init__(self, db_manager: SupabaseManager):
        self.db = db_manager

    def scrape_site(self, page, url, scrape_uuid):
        jobs_collected = []
        print(f"   Scanning: {url}")

        try:
            page.goto(url, timeout=60000, wait_until="networkidle")
            page.wait_for_timeout(3000)

            # Cookie Banner Handling
            try:
                btn = page.locator('xpath=//button[contains(text(),"Accept") or contains(text(),"Agree")]')
                if btn.count() > 0:
                    btn.first.click(force=True)
                    page.wait_for_timeout(500)
            except: pass

            # Check for jobs
            jobs = page.locator('xpath=//li[@data-ui="job"]')
            page.wait_for_timeout(5000)

            if jobs.count() == 0:
                print("     ⚠️ No jobs found")
                return []

            # Load More Logic
            while True:
                more = page.locator('xpath=//button[@data-ui="load-more-button"]')
                if more.count() == 0: break
                try:
                    more.first.scroll_into_view_if_needed()
                    more.first.click(force=True)
                    page.wait_for_timeout(1200)
                except: break

            print(f"     ↳ Found {jobs.count()} jobs")

            for i in range(jobs.count()):
                job = jobs.nth(i)
                title = ScraperUtils.safe_text(job.locator('xpath=.//h3[@data-ui="job-title"]'))
                href = ScraperUtils.safe_attr(job.locator('xpath=.//a[1]'), "href")
                dept = ScraperUtils.safe_text(job.locator('xpath=.//span[@data-ui="job-department"]'))
                jtype = ScraperUtils.safe_text(job.locator('xpath=.//span[@data-ui="job-type"]'))

                if title == "NA" or href == "NA": continue
                
                job_url = urljoin(page.url, href)
                now_iso = datetime.now(timezone.utc).isoformat()

                jobs_collected.append({
                    "id": uuid.uuid4().int % (2**63 - 1),
                    "created_at": now_iso,
                    "updated_at": now_iso,
                    "scrape_id": scrape_uuid,
                    "title": title,
                    "job_url": job_url,
                    "is_active": True,
                    "published_date": None,
                    "job_type": jtype if jtype != "NA" else None,
                    "department": dept if dept != "NA" else None,
                    "original_description": None
                })

        except Exception as e:
            print(f"     ❌ Error during scraping: {e}")
            return []

        return jobs_collected

    def run(self):
        print("\n--- STAGE 1: DISCOVERY (SYNC) ---")
        start_time = time.time()
        companies = self.db.fetch_companies(ats_name="workable")
        print(f"Found {len(companies)} Workable companies.")

        all_jobs = []
        all_scrapes = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"
            )
            # Anti-detection script
            context.add_init_script("""
                document.addEventListener('DOMContentLoaded', () => {
                    [...document.querySelectorAll('button')]
                      .find(b => /accept|agree/i.test(b.innerText))
                      ?.click();
                });
            """)
            page = context.new_page()

            for _, row in companies.iterrows():
                scrape_uuid = str(uuid.uuid4())
                scrape_record = {
                    "id": scrape_uuid,
                    "ats_website_id": row["id"],
                    "status": "pending",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }

                try:
                    jobs = self.scrape_site(page, row["ats_url"], scrape_uuid)
                    scrape_record["status"] = "success"
                    scrape_record["finished_at"] = datetime.now(timezone.utc).isoformat()
                    if jobs: all_jobs.extend(jobs)
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
class WorkableEnrichment:
    """Class responsible for enriching Workable job details."""

    def __init__(self, db_manager: SupabaseManager, concurrency=10):
        self.db = db_manager
        self.concurrency = concurrency

    async def _extract_details(self, page, content):
        data = {"original_description": None, "location": None, "job_type": None, "department": None}
        soup = BeautifulSoup(content, "html.parser")
        
        # 1. Description
        desc_section = soup.find("section", {"data-ui": "job-description"})
        if desc_section:
            data["original_description"] = desc_section.get_text(separator="\n\n", strip=True)

        # 2. Metadata (UI tags)
        loc_tag = soup.find(attrs={"data-ui": "job-location"})
        if loc_tag: data["location"] = ScraperUtils.clean_text(loc_tag.get_text())

        type_tag = soup.find(attrs={"data-ui": "job-type"})
        if type_tag: data["job_type"] = ScraperUtils.clean_text(type_tag.get_text())

        dept_tag = soup.find(attrs={"data-ui": "job-department"})
        if dept_tag: data["department"] = ScraperUtils.clean_text(dept_tag.get_text())

        # 3. Fallback: Meta tags
        if not data["location"]:
            meta_loc = soup.find("meta", property="og:locality")
            if meta_loc: data["location"] = meta_loc.get("content")

        return data

    async def _process_job(self, sem, context, job):
        async with sem:
            url = job.get("job_url")
            job_id = job.get("id")
            if not url: return

            page = await context.new_page()
            try:
                # 1. Load Page
                await page.goto(url, timeout=45000, wait_until="domcontentloaded")
                
                # 2. Force Click (SPA Handling)
                try:
                    # Try to click the job link again if it appears in a list (Workable quirks)
                    await page.evaluate("""() => {
                        const link = document.querySelector(`a[href="${window.location.href}"]`);
                        if(link) link.click();
                    }""")
                    await asyncio.sleep(1)
                except: pass

                # 3. Wait for Description
                try:
                    await page.wait_for_selector('[data-ui="job-description"]', timeout=10000)
                except: pass

                # 4. Extract
                content = await page.content()
                scraped = await self._extract_details(page, content)

                payload = {
                    "original_description": scraped["original_description"],
                    "location": scraped["location"] or job.get("location"),
                    "job_type": scraped["job_type"] or job.get("job_type"),
                    "department": scraped["department"] or job.get("department")
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
        jobs = self.db.fetch_pending_jobs(ats_filter="Workable")
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
    # Initialize Manager
    db_manager = SupabaseManager()

    # Run Stage 1
    discovery = WorkableDiscovery(db_manager)
    discovery.run()

    # Run Stage 2
    enrichment = WorkableEnrichment(db_manager, concurrency=10)
    enrichment.run()
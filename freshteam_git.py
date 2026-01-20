import time
import re
import uuid
import json
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

# ================= UTILS CLASS =================
class ScraperUtils:
    """Static helper methods for text processing and DOM safety."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        if not text: return ""
        text = text.replace('\u200b', '')
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def remove_html_tags(text: str) -> str:
        if not text: return ""
        return re.sub(r'<[^>]+>', '\n', text).strip()

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

    def fetch_companies(self, ats_name="freshteam") -> pd.DataFrame:
        try:
            res = self.client.table("ats_website").select("*").execute()
            df = pd.DataFrame(res.data)
            if not df.empty and "ats_name" in df.columns:
                return df[df["ats_name"].astype(str).str.lower().str.strip() == ats_name]
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching companies: {e}")
            return pd.DataFrame()

    def fetch_pending_jobs(self, ats_filter="Freshteam"):
        """Fetches jobs from jobs_duplicate that need enrichment."""
        try:
            print(f"Fetching pending {ats_filter} jobs...")
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
class FreshteamDiscovery:
    """Class responsible for scraping Freshteam job lists."""

    def __init__(self, db_manager: SupabaseManager):
        self.db = db_manager

    def scrape_site(self, page, url, scrape_uuid):
        jobs_collected = []
        print(f"   Scanning: {url}")

        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            
            # Wait for main jobs list container
            try:
                page.wait_for_selector('xpath=//div[@data-portal-id="jobs_list"]', timeout=15000)
            except:
                print("     ⚠️ No job list found (structure may have changed).")
                return []

            # Iterate over "Roles" (Categories/Departments)
            roles = page.locator('xpath=//li[@data-portal-role]')
            
            if roles.count() == 0:
                print("     ⚠️ No job roles found.")
                return []

            print(f"     ↳ Found {roles.count()} job categories")

            for i in range(roles.count()):
                role = roles.nth(i)
                
                # Extract Department Name
                dept_el = role.locator('xpath=.//div[@class="role-title"]/h5')
                department = ScraperUtils.safe_text(dept_el)

                # Iterate over Jobs in this Role
                job_items = role.locator('xpath=.//a[contains(@class,"heading")]')

                for j in range(job_items.count()):
                    job = job_items.nth(j)

                    title_el = job.locator('xpath=.//div[@class="job-title"]')
                    loc_el = job.locator('xpath=.//div[@class="location-info"]')

                    title = ScraperUtils.safe_text(title_el)
                    raw_href = ScraperUtils.safe_attr(job, "href")
                    job_url = urljoin(url, raw_href)

                    # Parse Location and Type from the list items inside .location-info
                    location = "NA"
                    job_type = "NA"

                    if loc_el.count() > 0:
                        loc_text = loc_el.inner_text()
                        parts = [p.strip() for p in loc_text.split("\n") if p.strip()]
                        if len(parts) >= 1:
                            location = parts[0]
                        if len(parts) >= 2:
                            job_type = parts[-1]

                    if title == "NA" or job_url == "NA":
                        continue

                    # NOTE: We save 'location' here for potential debugging or CSV export,
                    # but pure deduplication logic might exclude it during DB insert if needed.
                    jobs_collected.append({
                        "id": uuid.uuid4().int % (2**63 - 1),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "scrape_id": scrape_uuid,
                        "title": title,
                        "job_url": job_url,
                        "is_active": True,
                        "published_date": None,
                        "job_type": job_type if job_type != "NA" else None,
                        "department": department if department != "NA" else None,
                        "location": location if location != "NA" else None,
                        "original_description": None
                    })

        except Exception as e:
            print(f"     ❌ Error during scraping: {e}")
            return []

        return jobs_collected

    def run(self):
        print("\n--- STAGE 1: DISCOVERY (SYNC) ---")
        start_time = time.time()
        companies = self.db.fetch_companies(ats_name="freshteam")
        print(f"Found {len(companies)} Freshteam companies.")

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
class FreshteamEnrichment:
    """Class responsible for enriching Freshteam job details."""

    def __init__(self, db_manager: SupabaseManager, concurrency=5):
        self.db = db_manager
        self.concurrency = concurrency

    async def _extract_details(self, page):
        data = {"original_description": None, "location": None, "job_type": None, "department": None}
        
        # 1. VISIBLE DESCRIPTION
        try:
            desc_el = page.locator(".job-details-content.content").first
            if await desc_el.count() > 0:
                data["original_description"] = ScraperUtils.clean_text(await desc_el.inner_text())
        except: pass

        # 2. JSON-LD FALLBACK (For better Location/Type or missing Description)
        try:
            # Locate all JSON-LD scripts
            scripts = page.locator('script[type="application/ld+json"]')
            count = await scripts.count()
            
            for i in range(count):
                try:
                    content = await scripts.nth(i).inner_text()
                    json_data = json.loads(content)

                    # Normalize to list to handle single objects
                    if isinstance(json_data, dict):
                        json_data = [json_data]
                    
                    # Find JobPosting
                    posting = next((item for item in json_data if item.get("@type") == "JobPosting"), None)
                    
                    if posting:
                        # A. Description Fallback
                        if not data["original_description"]:
                            raw_desc = posting.get("description", "")
                            data["original_description"] = ScraperUtils.remove_html_tags(raw_desc)

                        # B. Job Type
                        if posting.get("employmentType"):
                            raw_type = str(posting.get("employmentType")).replace("_", " ").title()
                            data["job_type"] = raw_type

                        # C. Location (Freshteam structure usually: address -> addressLocality/Region)
                        addr = posting.get("jobLocation", {}).get("address", {})
                        loc_parts = []
                        if addr.get("addressLocality"): loc_parts.append(addr.get("addressLocality"))
                        if addr.get("addressRegion"): loc_parts.append(addr.get("addressRegion"))
                        if addr.get("addressCountry"): loc_parts.append(addr.get("addressCountry"))
                        
                        if loc_parts:
                            data["location"] = ", ".join(loc_parts)
                        
                        break # Found it, stop looking
                except:
                    continue
        except: pass

        # 3. REGEX Fallback for Location if still missing
        if not data["location"] and data["original_description"]:
            match = re.search(r"Location\s*:\s*(.+)", data["original_description"], re.I)
            if match:
                data["location"] = match.group(1).strip()[:50]

        return data

    async def _process_job(self, sem, context, job):
        async with sem:
            url = job.get("job_url")
            job_id = job.get("id")
            if not url: return

            page = await context.new_page()
            try:
                await page.goto(url, timeout=45000, wait_until="domcontentloaded")
                # Small sleep for dynamic content stability
                await asyncio.sleep(1)
                
                scraped = await self._extract_details(page)

                payload = {
                    "original_description": scraped["original_description"],
                    "location": scraped["location"] or job.get("location"),
                    "job_type": scraped["job_type"] or job.get("job_type"),
                    # Keep existing department if scrape finds nothing (Freshteam detail pages often lack Dept info)
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
        jobs = self.db.fetch_pending_jobs(ats_filter="Freshteam")
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

    # Run Stage 1: Discovery
    discovery = FreshteamDiscovery(db_manager)
    discovery.run()

    # Run Stage 2: Enrichment
    enrichment = FreshteamEnrichment(db_manager, concurrency=5)
    enrichment.run()
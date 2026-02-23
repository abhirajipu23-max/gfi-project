import time
import re
import uuid
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SCRAPES_TABLE = "scrapes_duplicate"
JOBS_TABLE = "jobs"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

MAX_STAGE1_WORKERS = 1
STAGE2_CONCURRENCY = 10

OUTPUT_JOBS_FILE = "workable_jobs.csv"
OUTPUT_SCRAPES_FILE = "workable_scrapes.csv"


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
        if not text or text == "NA":
            return None

        t = ScraperUtils.clean_text(text).lower()

        mapping = {
            "full time": "Full Time",
            "full-time": "Full Time",
            "part time": "Part Time",
            "part-time": "Part Time",
            "contract": "Contract",
            "temporary": "Temporary",
            "internship": "Internship",
            "freelance": "Freelance",
        }

        return mapping.get(t, text.strip())

    @staticmethod
    def normalize_workable_posted_date(text: str):
        if not text or text == "NA":
            return None

        t = text.strip().lower()
        t = re.sub(r"\s+", " ", t)

        t = t.replace("posted", "").strip()
        t = t.replace("about", "").replace("approx", "").strip()

        now = datetime.now(timezone.utc)

        if "today" in t:
            return now.isoformat()

        if "yesterday" in t:
            return (now - timedelta(days=1)).isoformat()

        m = re.search(r"(\d+)\s*minute", t)
        if m:
            return (now - timedelta(minutes=int(m.group(1)))).isoformat()

        m = re.search(r"(\d+)\s*hour", t)
        if m:
            return (now - timedelta(hours=int(m.group(1)))).isoformat()

        m = re.search(r"(\d+)\s*day", t)
        if m:
            return (now - timedelta(days=int(m.group(1)))).isoformat()

        m = re.search(r"(\d+)\s*week", t)
        if m:
            return (now - timedelta(weeks=int(m.group(1)))).isoformat()

        m = re.search(r"(\d+)\s*month", t)
        if m:
            return (now - timedelta(days=int(m.group(1)) * 30)).isoformat()

        m = re.search(r"(\d+)\s*year", t)
        if m:
            return (now - timedelta(days=int(m.group(1)) * 365)).isoformat()

        return None


class SupabaseManager:
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
        try:
            print("Fetching pending Workable jobs...")
            res = (
                self.client.table(JOBS_TABLE)
                .select("id, job_url, location, job_type, department, scrapes_duplicate(ats_website(ats_name))")
                .is_("original_description", "null")
                .execute()
            )

            data = res.data or []
            pending = []
            for j in data:
                try:
                    ats_name = j["scrapes_duplicate"]["ats_website"]["ats_name"]
                    if ats_filter.lower() in ats_name.lower():
                        pending.append(j)
                except:
                    continue

            return pending
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
        batch_size = 100

        for i in range(0, len(all_urls), batch_size):
            batch = all_urls[i : i + batch_size]
            try:
                res = self.client.table(JOBS_TABLE).select("job_url").in_("job_url", batch).execute()
                for row in res.data:
                    existing_urls.add(row["job_url"])
            except Exception as e:
                print(f"Duplicate check error: {e}")

        new_jobs = [j for j in unique_jobs_list if j["job_url"] not in existing_urls]
        print(f"Inserting {len(new_jobs)} new jobs ({len(unique_jobs_list) - len(new_jobs)} skipped).")

        for i in range(0, len(new_jobs), batch_size):
            batch = new_jobs[i : i + batch_size]
            try:
                self.client.table(JOBS_TABLE).insert(batch).execute()
                print(f"Uploaded batch {i}-{i+len(batch)}")
            except Exception as e:
                print(f"Error inserting batch: {e}")

    def update_job(self, job_id, payload):
        try:
            clean_payload = {k: v for k, v in payload.items() if v is not None and v != ""}
            if clean_payload:
                clean_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.client.table(JOBS_TABLE).update(clean_payload).eq("id", job_id).execute()
                print(f"Updated ID {job_id}")
        except Exception as e:
            print(f"Update Failed ID {job_id}: {e}")


class WorkableDiscovery:
    def __init__(self, db_manager: SupabaseManager):
        self.db = db_manager

    def scrape_site(self, page, url, scrape_uuid):
        jobs_collected = []
        print(f"Scanning: {url}")

        try:
            page.goto(url, timeout=60000, wait_until="networkidle")
            page.wait_for_timeout(2500)

            try:
                btn = page.locator('xpath=//button[contains(text(),"Accept") or contains(text(),"Agree")]')
                if btn.count() > 0:
                    btn.first.click(force=True)
                    page.wait_for_timeout(500)
            except:
                pass

            jobs = page.locator('xpath=//li[@data-ui="job"]')
            page.wait_for_timeout(1500)

            if jobs.count() == 0:
                print("No jobs found")
                return []

            while True:
                more = page.locator('xpath=//button[@data-ui="load-more-button"]')
                if more.count() == 0:
                    break
                try:
                    more.first.scroll_into_view_if_needed()
                    more.first.click(force=True)
                    page.wait_for_timeout(1200)
                except:
                    break

            jobs = page.locator('xpath=//li[@data-ui="job"]')
            print(f"Found {jobs.count()} jobs")

            for i in range(jobs.count()):
                job = jobs.nth(i)

                title = ScraperUtils.safe_text(job.locator('xpath=.//h3[@data-ui="job-title"]'))
                href = ScraperUtils.safe_attr(job.locator('xpath=.//a[1]'), "href")

                dept = ScraperUtils.safe_text(job.locator('xpath=.//span[@data-ui="job-department"]'))
                jtype = ScraperUtils.safe_text(job.locator('xpath=.//span[@data-ui="job-type"]'))

                posted_text = ScraperUtils.safe_text(job.locator('xpath=.//small[@data-ui="job-posted"][1]'))
                published_date = ScraperUtils.normalize_workable_posted_date(posted_text)

                loc = ScraperUtils.safe_text(job.locator('xpath=.//*[@data-ui="job-location"]'))
                if loc == "NA":
                    loc = None

                if title == "NA" or href == "NA":
                    continue

                job_url = urljoin(page.url, href)
                now_iso = datetime.now(timezone.utc).isoformat()

                jobs_collected.append({
                    "created_at": now_iso,
                    "updated_at": now_iso,
                    "scrape_id": scrape_uuid,
                    "title": title,
                    "job_url": job_url,
                    "is_active": True,
                    "published_date": published_date,
                    "job_type": ScraperUtils.normalize_job_type(jtype),
                    "department": dept if dept != "NA" else None,
                    "location": loc,
                    "original_description": None
                })

        except Exception as e:
            print(f"Error during scraping: {e}")
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
            context = browser.new_context(user_agent=USER_AGENT)

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
                now_iso = datetime.now(timezone.utc).isoformat()

                scrape_record = {
                    "id": scrape_uuid,
                    "ats_website_id": row["id"],
                    "status": "pending",
                    "created_at": now_iso,
                    "updated_at": now_iso
                }

                try:
                    jobs = self.scrape_site(page, row["ats_url"], scrape_uuid)

                    scrape_record["status"] = "success"
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
            pd.DataFrame(all_jobs).to_csv("workable_jobs_stage1.csv", index=False)
            print("Stage-1 jobs snapshot saved: workable_jobs_stage1.csv")

        clean_jobs = [j for j in all_jobs if j.get("job_url") and j["job_url"] != "NA"]
        self.db.save_scrapes(all_scrapes)
        self.db.save_jobs_deduplicated(clean_jobs)

        print(f"Stage 1 Runtime: {time.time() - start_time:.2f}s")


class WorkableEnrichment:
    def __init__(self, db_manager: SupabaseManager, concurrency=10):
        self.db = db_manager
        self.concurrency = concurrency

    def _extract_details_from_html(self, html: str):
        data = {
            "original_description": None,
            "location": None,
            "job_type": None,
            "department": None
        }

        soup = BeautifulSoup(html, "html.parser")

        loc_block = soup.select_one('div[data-ui="job-location"]')
        if loc_block:
            data["location"] = ScraperUtils.clean_text(loc_block.get_text(" ", strip=True))

        type_tag = soup.find(attrs={"data-ui": "job-type"})
        if type_tag:
            data["job_type"] = ScraperUtils.normalize_job_type(type_tag.get_text(strip=True))

        dept_tag = soup.find(attrs={"data-ui": "job-department"})
        if dept_tag:
            data["department"] = ScraperUtils.clean_text(dept_tag.get_text(strip=True))

        parts = []

        desc_section = soup.find("section", {"data-ui": "job-description"})
        if desc_section:
            parts.append(desc_section.get_text(separator="\n", strip=True))

        req_section = soup.find("section", {"data-ui": "job-requirements"})
        if req_section:
            parts.append(req_section.get_text(separator="\n", strip=True))

        ben_section = soup.find("section", {"data-ui": "job-benefits"})
        if ben_section:
            parts.append(ben_section.get_text(separator="\n", strip=True))

        full_desc = "\n\n".join([p for p in parts if p]).strip()

        if full_desc:
            data["original_description"] = full_desc

        if not data["location"]:
            meta_loc = soup.find("meta", property="og:locality")
            if meta_loc and meta_loc.get("content"):
                data["location"] = meta_loc.get("content")

        return data

    async def _process_job(self, sem, context, job):
        async with sem:
            url = job.get("job_url")
            job_id = job.get("id")
            if not url or not job_id:
                return

            page = await context.new_page()

            try:
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")
                await asyncio.sleep(1)

                try:
                    await page.wait_for_selector('section[data-ui="job-description"]', timeout=12000)
                except:
                    pass

                html = await page.content()
                scraped = self._extract_details_from_html(html)

                payload = {
                    "original_description": scraped.get("original_description"),
                    "location": scraped.get("location") or job.get("location"),
                    "job_type": scraped.get("job_type") or job.get("job_type"),
                    "department": scraped.get("department") or job.get("department"),
                }

                if payload.get("original_description"):
                    await asyncio.to_thread(self.db.update_job, job_id, payload)
                else:
                    print(f"No description found ID {job_id}")

            except Exception as e:
                print(f"Failed ID {job_id}: {str(e)[:120]}")
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
            context = await browser.new_context(user_agent=USER_AGENT)

            tasks = [self._process_job(sem, context, job) for job in jobs]
            await asyncio.gather(*tasks)

            await browser.close()

        print("Enrichment completed.")

    def run(self):
        asyncio.run(self.run_async())


def export_workable_jobs_backup_from_db(db: SupabaseManager):
    print("\nExporting FINAL Workable jobs backup (Stage 1 + Stage 2 combined) to CSV...")

    try:
        res = (
            db.client.table(JOBS_TABLE)
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

    df = df[df["ats_name"].astype(str).str.lower().str.contains("workable", na=False)].copy()

    if "scrapes_duplicate" in df.columns:
        df.drop(columns=["scrapes_duplicate"], inplace=True)

    df.to_csv(OUTPUT_JOBS_FILE, index=False)
    print(f"Final backup saved: {OUTPUT_JOBS_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    start = time.time()

    db_manager = SupabaseManager()

    discovery = WorkableDiscovery(db_manager)
    discovery.run()

    enrichment = WorkableEnrichment(db_manager, concurrency=STAGE2_CONCURRENCY)
    enrichment.run()

    export_workable_jobs_backup_from_db(db_manager)

    print(f"\nTotal Runtime: {time.time() - start:.2f}s")

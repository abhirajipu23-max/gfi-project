import time
import re
import uuid
import json
import asyncio
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = "https://ufnaxahhlblwpdomlybs.supabase.co"
SUPABASE_KEY = "sb_publishable_1d4J1Ll81KwhYPOS40U8mQ_qtCccNsa"

SCRAPES_TABLE = "scrapes_duplicate"
JOBS_TABLE = "jobs_duplicate"

OUTPUT_JOBS_FILE = "freshteam_jobs_stage1.csv"
OUTPUT_SCRAPES_FILE = "freshteam_scrapes_stage1.csv"
FINAL_BACKUP_FILE = "freshteam_jobs_backup.csv"

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
    def remove_html_tags(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"</p\s*>", "\n\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return ScraperUtils.clean_text(text)

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

        t = ScraperUtils.clean_text(text).lower()

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

        t = t.replace("_", " ").strip()
        return mapping.get(t, ScraperUtils.clean_text(text))

    @staticmethod
    def format_date_iso(date_str: str):
        if not date_str:
            return None

        s = ScraperUtils.clean_text(str(date_str))

        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if not dt.tzinfo:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except:
            pass

        formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except:
                continue

        return None


class SupabaseManager:
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
                j for j in all_data
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

    def update_job(self, job_id, payload):
        try:
            clean_payload = {k: v for k, v in payload.items() if v is not None and v != ""}
            if clean_payload:
                clean_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.client.table(JOBS_TABLE).update(clean_payload).eq("id", job_id).execute()
                print(f"[ID {job_id}] Updated")
        except Exception as e:
            print(f"[ID {job_id}] Update Failed: {e}")


class FreshteamDiscovery:
    def __init__(self, db_manager: SupabaseManager):
        self.db = db_manager

    def scrape_site(self, page, url, scrape_uuid):
        jobs_collected = []
        print(f"Scanning: {url}")

        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")

            try:
                page.wait_for_selector('xpath=//div[@data-portal-id="jobs_list"]', timeout=15000)
            except:
                print("No job list found.")
                return []

            roles = page.locator('xpath=//li[@data-portal-role]')
            if roles.count() == 0:
                print("No job roles found.")
                return []

            print(f"Found {roles.count()} job categories")

            for i in range(roles.count()):
                role = roles.nth(i)

                dept_el = role.locator('xpath=.//div[@class="role-title"]/h5')
                department = ScraperUtils.safe_text(dept_el)

                job_items = role.locator('xpath=.//a[contains(@class,"heading")]')
                for j in range(job_items.count()):
                    job = job_items.nth(j)

                    title_el = job.locator('xpath=.//div[@class="job-title"]')
                    loc_el = job.locator('xpath=.//div[@class="location-info"]')

                    title = ScraperUtils.safe_text(title_el)
                    raw_href = ScraperUtils.safe_attr(job, "href")
                    job_url = urljoin(url, raw_href)

                    location = None
                    job_type = None

                    if loc_el.count() > 0:
                        try:
                            loc_text = loc_el.inner_text()
                            parts = [p.strip() for p in loc_text.split("\n") if p.strip()]
                            if len(parts) >= 1:
                                location = parts[0]
                            if len(parts) >= 2:
                                job_type = parts[-1]
                        except:
                            pass

                    if title == "NA" or not job_url or job_url == "NA":
                        continue

                    now_iso = datetime.now(timezone.utc).isoformat()

                    jobs_collected.append({
                        "created_at": now_iso,
                        "updated_at": now_iso,
                        "scrape_id": scrape_uuid,
                        "title": title,
                        "job_url": job_url,
                        "is_active": True,
                        "published_date": None,
                        "job_type": ScraperUtils.normalize_job_type(job_type) if job_type else None,
                        "department": department if department != "NA" else None,
                        "location": location,
                        "original_description": None
                    })

        except Exception as e:
            print(f"Error during scraping: {e}")
            return []

        return jobs_collected

    def run(self):
        print("\n--- STAGE 1: DISCOVERY (FRESHTEAM) ---")
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


class FreshteamEnrichment:
    def __init__(self, db_manager: SupabaseManager, concurrency=5):
        self.db = db_manager
        self.concurrency = concurrency

    async def _extract_details(self, page):
        data = {
            "original_description": None,
            "location": None,
            "job_type": None,
            "department": None,
            "published_date": None,
        }

        try:
            desc_el = page.locator(".job-details-content.content")
            if await desc_el.count() > 0:
                data["original_description"] = ScraperUtils.clean_text(await desc_el.first.inner_text())
        except:
            pass

        try:
            scripts = page.locator('script[type="application/ld+json"]')
            count = await scripts.count()

            for i in range(count):
                try:
                    content = await scripts.nth(i).inner_text()
                    json_data = json.loads(content)

                    if isinstance(json_data, dict):
                        json_data = [json_data]

                    posting = next((item for item in json_data if item.get("@type") == "JobPosting"), None)
                    if not posting:
                        continue

                    if not data["original_description"]:
                        raw_desc = posting.get("description", "")
                        data["original_description"] = ScraperUtils.remove_html_tags(raw_desc)

                    if posting.get("employmentType"):
                        data["job_type"] = ScraperUtils.normalize_job_type(str(posting.get("employmentType")))

                    if posting.get("datePosted"):
                        data["published_date"] = ScraperUtils.format_date_iso(posting.get("datePosted"))

                    job_loc = posting.get("jobLocation")
                    addr = None

                    if isinstance(job_loc, dict):
                        addr = job_loc.get("address")
                    elif isinstance(job_loc, list) and len(job_loc) > 0:
                        addr = job_loc[0].get("address")

                    loc_parts = []
                    if isinstance(addr, dict):
                        if addr.get("addressLocality"):
                            loc_parts.append(addr.get("addressLocality"))
                        if addr.get("addressRegion"):
                            loc_parts.append(addr.get("addressRegion"))
                        if addr.get("addressCountry"):
                            loc_parts.append(addr.get("addressCountry"))

                    if loc_parts:
                        data["location"] = ", ".join(loc_parts)

                    break

                except:
                    continue
        except:
            pass

        return data

    async def _process_job(self, sem, context, job):
        async with sem:
            url = job.get("job_url")
            job_id = job.get("id")
            if not url or not job_id:
                return

            page = await context.new_page()
            try:
                await page.goto(url, timeout=45000, wait_until="domcontentloaded")
                await asyncio.sleep(1)

                scraped = await self._extract_details(page)

                payload = {
                    "original_description": scraped["original_description"],
                    "location": scraped["location"] or job.get("location"),
                    "job_type": scraped["job_type"] or job.get("job_type"),
                    "department": scraped["department"] or job.get("department"),
                    "published_date": scraped["published_date"],
                }

                if payload["original_description"]:
                    await asyncio.to_thread(self.db.update_job, job_id, payload)
                else:
                    print(f"[ID {job_id}] No description found.")

            except Exception as e:
                print(f"[ID {job_id}] Failed: {str(e)[:80]}")
            finally:
                await page.close()

    async def run_async(self):
        print("\n--- STAGE 2: ENRICHMENT (FRESHTEAM) ---")
        jobs = self.db.fetch_pending_jobs(ats_filter="Freshteam")
        if not jobs:
            print("No jobs pending enrichment.")
            return

        print(f"Enriching {len(jobs)} jobs with concurrency {self.concurrency}...")

        sem = asyncio.Semaphore(self.concurrency)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )

            tasks = [self._process_job(sem, context, job) for job in jobs]
            await asyncio.gather(*tasks)

            await browser.close()

        print("Enrichment completed.")
        print("--- STAGE 2 COMPLETE ---")

    def run(self):
        asyncio.run(self.run_async())


def export_freshteam_jobs_backup_from_db(db: SupabaseManager):
    print("\nExporting FINAL Freshteam jobs backup (Stage 1 + Stage 2 combined) to CSV...")

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

    df = df[df["ats_name"].astype(str).str.lower().str.contains("freshteam", na=False)].copy()

    if "scrapes_duplicate" in df.columns:
        df.drop(columns=["scrapes_duplicate"], inplace=True)

    df.to_csv(FINAL_BACKUP_FILE, index=False)
    print(f"Final backup saved: {FINAL_BACKUP_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    start = time.time()

    db_manager = SupabaseManager()

    discovery = FreshteamDiscovery(db_manager)
    discovery.run()

    enrichment = FreshteamEnrichment(db_manager, concurrency=5)
    enrichment.run()

    export_freshteam_jobs_backup_from_db(db_manager)

    print(f"\nTotal Runtime: {time.time() - start:.2f}s")

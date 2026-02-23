import os
import time
import re
import atexit
import threading
import requests
import pandas as pd
import uuid

from dotenv import load_dotenv
from datetime import datetime, timezone
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from supabase import create_client, Client

load_dotenv()

import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

MAX_WORKERS = 5
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

SCRAPES_TABLE = "scrapes"
JOBS_TABLE = "jobs"

OUTPUT_JOBS_FILE = "zoho_jobs.csv"
OUTPUT_SCRAPES_FILE = "zoho_scrapes.csv"

INSERT_BATCH_SIZE = 100
CHECK_BATCH_SIZE = 50

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Failed to initialize Supabase client: {e}")
    raise SystemExit(1)


def format_date_iso(s):
    if not s:
        return None

    s = s.strip()
    s = re.sub(r"(?i)posted\s*on", "", s).strip()

    formats = (
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d-%b-%Y",
        "%d/%m/%Y",
    )

    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass

    return None


def parse_experience(exp_text):
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


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""


def dedupe_jobs(jobs):
    seen = {}
    for j in jobs:
        if j.get("job_url"):
            key = f"{j.get('job_url')}::{j.get('title','')}"
            seen[key] = j
    return list(seen.values())


def get_zoho_companies():
    try:
        res = supabase.table("ats_website").select("*").execute()
        df = pd.DataFrame(res.data)
        if df.empty:
            return df

        if "ats_name" in df.columns:
            df = df[df["ats_name"].astype(str).str.lower().str.strip() == "zoho recruit"]

        return df
    except Exception as e:
        print(f"Error fetching companies: {e}")
        return pd.DataFrame()


class ZohoDiscovery:
    def __init__(self):
        self._thread_local = threading.local()
        atexit.register(self.cleanup_playwright)

    def _get_thread_browser(self):
        if not hasattr(self._thread_local, "playwright"):
            self._thread_local.playwright = sync_playwright().start()
            self._thread_local.browser = self._thread_local.playwright.chromium.launch(headless=True)
        return self._thread_local.browser

    def cleanup_playwright(self):
        try:
            if hasattr(self._thread_local, "browser"):
                self._thread_local.browser.close()
                self._thread_local.playwright.stop()
        except:
            pass

    def _normalize_job(self, scrape_uuid, title="", job_type="", posted="", job_url="", location=""):
        now_iso = datetime.now(timezone.utc).isoformat()

        return {
            "created_at": now_iso,
            "updated_at": now_iso,
            "scrape_id": scrape_uuid,
            "title": title,
            "job_url": job_url,
            "is_active": True,
            "published_date": format_date_iso(posted),
            "job_type": job_type if job_type else None,
            "location": location if location else None,
            "original_description": None,
            "internal_slug": None,
            "min_exp": None,
            "max_exp": None,
            "department": None,
            "industry": None,
        }

    def _scrape_zoho_table(self, soup, url, scrape_uuid):
        jobs = []
        rows = soup.select("tr.jobDetailRow")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            link = cols[0].find("a", class_="jobdetail")
            job_type = link.get_text(strip=True) if link else ""
            job_url_parsed = urljoin(url, link["href"]) if link and link.get("href") else ""

            jobs.append(self._normalize_job(
                scrape_uuid=scrape_uuid,
                title=cols[1].get_text(strip=True),
                job_type=job_type,
                posted=cols[2].get_text(strip=True),
                location=cols[3].get_text(strip=True),
                job_url=job_url_parsed
            ))
        return jobs

    def _scrape_zoho_layout3_cards(self, soup, url, scrape_uuid):
        jobs = []
        cards = soup.select("div.cw-filter-joblist")
        if not cards:
            return []

        for card in cards:
            a = card.select_one("a.cw-3-title")
            if not a:
                continue

            title = a.get_text(strip=True)
            job_url_parsed = urljoin(url, a.get("href", ""))

            location = ""
            loc_el = card.select_one("p.filter-subhead")
            if loc_el:
                exp = loc_el.select_one("span.search-work-experience")
                if exp:
                    exp.extract()
                location = loc_el.get_text(" ", strip=True)

            job_type, posted = "", ""
            type_el = card.select_one("span.cw-full-time")
            if type_el:
                job_type = type_el.get_text(strip=True)
            posted_el = card.select_one("span.search-date-opened")
            if posted_el:
                posted = posted_el.get_text(strip=True)

            jobs.append(self._normalize_job(
                scrape_uuid=scrape_uuid,
                title=title,
                job_type=job_type,
                posted=posted,
                location=location,
                job_url=job_url_parsed
            ))

        return dedupe_jobs(jobs)

    def _scrape_bs4(self, url, scrape_uuid):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            if r.status_code != 200:
                return []

            soup = BeautifulSoup(r.text, "html.parser")

            jobs = self._scrape_zoho_table(soup, url, scrape_uuid)
            if jobs:
                return dedupe_jobs(jobs)

            jobs = self._scrape_zoho_layout3_cards(soup, url, scrape_uuid)
            if jobs:
                return dedupe_jobs(jobs)

            jobs = []
            anchors = soup.select(
                "a.cw-1-title,"
                "a.cw-3-title,"
                "a.zp_career_site_job_title,"
                "a.jobdetail,"
                "a[href*='/jobs/'],"
                "a[href*='JobOpenings'],"
                "a[href*='jobdetail'],"
                "a[href*='recruit']"
            )

            for a in anchors:
                href = a.get("href")
                title = a.get_text(" ", strip=True)

                if not href or not title:
                    continue

                if "visit website" in title.lower() or "zoho recruit" in title.lower():
                    continue

                jobs.append(self._normalize_job(
                    scrape_uuid=scrape_uuid,
                    title=title,
                    job_url=urljoin(url, href)
                ))

            return dedupe_jobs(jobs)

        except:
            return []

    def _scrape_playwright(self, url, scrape_uuid):
        try:
            browser = self._get_thread_browser()
            page = browser.new_page(user_agent=USER_AGENT)

            page.goto(url, timeout=45000, wait_until="domcontentloaded")
            page.wait_for_timeout(2500)

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            jobs = self._scrape_zoho_table(soup, url, scrape_uuid)
            if jobs:
                page.close()
                return dedupe_jobs(jobs)

            jobs = self._scrape_zoho_layout3_cards(soup, url, scrape_uuid)
            if jobs:
                page.close()
                return dedupe_jobs(jobs)

            jobs = []
            selectors = [
                "a.cw-1-title",
                "a.cw-3-title",
                "a.zp_career_site_job_title",
                "a.jobdetail",
                "a[href*='/jobs/']",
                "a[href*='JobOpenings']",
                "a[href*='jobdetail']",
            ]

            for sel in selectors:
                loc = page.locator(sel)
                if loc.count() > 0:
                    for i in range(loc.count()):
                        a = loc.nth(i)

                        try:
                            title = a.inner_text().strip()
                        except:
                            title = ""

                        try:
                            href = a.get_attribute("href") or ""
                        except:
                            href = ""

                        if not href or not title:
                            continue

                        if "zoho recruit" in title.lower() or "visit website" in title.lower():
                            continue

                        jobs.append(self._normalize_job(
                            scrape_uuid=scrape_uuid,
                            title=title,
                            job_url=urljoin(url, href)
                        ))

                    page.close()
                    return dedupe_jobs(jobs)

            page.close()
            return []

        except:
            return []

    def process_company(self, row):
        if not row.get("ats_url"):
            return None

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

        jobs = self._scrape_bs4(row["ats_url"], scrape_uuid)
        if not jobs:
            jobs = self._scrape_playwright(row["ats_url"], scrape_uuid)

        finish_iso = datetime.now(timezone.utc).isoformat()
        scrape_record["finished_at"] = finish_iso
        scrape_record["updated_at"] = finish_iso
        scrape_record["status"] = "success" if jobs else "failed"

        return scrape_record, jobs

    def save_to_db(self, scrapes, jobs):
        if scrapes:
            try:
                supabase.table(SCRAPES_TABLE).upsert(scrapes).execute()
                print("Scrapes synced.")
            except Exception as e:
                print(f"Error saving scrapes: {e}")

        if not jobs:
            return

        urls = [j["job_url"] for j in jobs if j.get("job_url")]
        existing = set()

        print(f"Checking {len(urls)} jobs for duplicates...")

        for i in range(0, len(urls), CHECK_BATCH_SIZE):
            batch = urls[i:i + CHECK_BATCH_SIZE]
            try:
                res = supabase.table(JOBS_TABLE).select("job_url").in_("job_url", batch).execute()
                for r in res.data:
                    existing.add(r["job_url"])
            except Exception as e:
                print(f"Duplicate check error (batch {i}): {e}")

        new_jobs = [j for j in jobs if j.get("job_url") and j["job_url"] not in existing]
        print(f"Found {len(new_jobs)} new jobs to insert.")

        if not new_jobs:
            return

        for i in range(0, len(new_jobs), INSERT_BATCH_SIZE):
            batch = new_jobs[i:i + INSERT_BATCH_SIZE]
            try:
                supabase.table(JOBS_TABLE).insert(batch).execute()
                print(f"Inserted batch {i} - {i + len(batch)}")
            except Exception as e:
                print(f"Insert error: {e}")

    def run(self, df):
        print(f"\n--- STAGE 1: DISCOVERY ({len(df)} companies) ---")
        all_scrapes, all_jobs = [], []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = [exe.submit(self.process_company, row) for _, row in df.iterrows()]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    all_scrapes.append(res[0])
                    all_jobs.extend(res[1])

        if all_scrapes:
            pd.DataFrame(all_scrapes).to_csv(OUTPUT_SCRAPES_FILE, index=False)
            print(f"Stage-1 scrapes snapshot saved: {OUTPUT_SCRAPES_FILE}")

        if all_jobs:
            pd.DataFrame(all_jobs).to_csv(OUTPUT_JOBS_FILE, index=False)
            print(f"Stage-1 jobs snapshot saved: {OUTPUT_JOBS_FILE}")

        self.save_to_db(all_scrapes, all_jobs)
        print("--- STAGE 1 COMPLETE ---")


class ZohoEnrichment:
    def get_pending_jobs(self):
        print("\n--- STAGE 2: ENRICHMENT (Fetching pending jobs) ---")
        try:
            res = (
                supabase.table(JOBS_TABLE)
                .select("""
                    id, job_url,
                    scrapes!inner (
                        ats_website!inner ( ats_name )
                    )
                """)
                .is_("original_description", "null")
                .execute()
            )

            all_data = res.data or []
            zoho_jobs = [
                j for j in all_data
                if j["scrapes"]["ats_website"]["ats_name"].strip().lower() == "zoho recruit"
            ]

            print(f"Found {len(zoho_jobs)} Zoho jobs waiting for details.")
            return zoho_jobs

        except Exception as e:
            print(f"Error fetching pending jobs: {e}")
            return []

    def parse_job_details_from_soup(self, soup):
        data = {
            "title": "",
            "posted": "",
            "experience": "",
            "job_type": "",
            "raw_location": "",
            "description": "",
            "department": "",
            "industry": "",
        }

        for sel in [
            "div.tem3-titleBlock h1",
            "div.tem4-col2_hdr h2",
            "div.cw-jobheader-information h1",
            "h1",
            "h2",
        ]:
            el = soup.select_one(sel)
            if el:
                data["title"] = el.get_text(" ", strip=True).split("\n")[0].strip()
                break

        for sel in [
            "div.job-dec-block_new",
            "div.cw-jobdescription",
            "div.temp4-content",
            "span#spandesc",
            "div.tem3ContentBlock",
        ]:
            el = soup.select_one(sel)
            if el:
                data["description"] = el.get_text("\n", strip=True)
                break

        def apply_label_value(label, value):
            if not label or not value:
                return

            label = label.lower().strip()
            value = value.strip()

            if any(x in label for x in ["date opened", "posted", "date"]):
                if not data["posted"]:
                    data["posted"] = value

            elif "job type" in label:
                if not data["job_type"]:
                    data["job_type"] = value

            elif any(x in label for x in ["work experience", "required experience", "experience"]):
                if not data["experience"]:
                    data["experience"] = value

            elif any(x in label for x in ["department name", "department", "function"]):
                if not data["department"]:
                    data["department"] = value

            elif "industry" in label:
                if not data["industry"]:
                    data["industry"] = value

            elif any(x in label for x in ["location", "city", "state", "province", "country", "region"]):
                if not data["raw_location"]:
                    data["raw_location"] = value
                else:
                    data["raw_location"] += f", {value}"

        summary_items = soup.select("ul.cw-summary-list li")
        for li in summary_items:
            spans = li.find_all("span")
            if len(spans) < 2:
                continue
            apply_label_value(spans[0].get_text(strip=True), spans[1].get_text(strip=True))

        tem3_blocks = soup.select("div.tem3ContentBlock div.tem3-TextCont")
        for block in tem3_blocks:
            label_el = block.find("span")
            value_el = block.select_one("div.tem3-RightSideText")
            if label_el and value_el:
                apply_label_value(
                    label_el.get_text(" ", strip=True),
                    value_el.get_text(" ", strip=True)
                )

        return data if data["title"] else None

    def scrape_single_job(self, job_row):
        url = job_row["job_url"]
        job_id = job_row["id"]

        details = None

        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            if r.status_code == 200:
                details = self.parse_job_details_from_soup(BeautifulSoup(r.text, "html.parser"))
        except:
            pass

        if not details:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page(user_agent=USER_AGENT)
                    page.goto(url, timeout=60000, wait_until="domcontentloaded")
                    page.wait_for_timeout(1500)
                    html = page.content()
                    browser.close()

                details = self.parse_job_details_from_soup(BeautifulSoup(html, "html.parser"))
            except Exception as e:
                print(f"Playwright failed ID {job_id}: {e}")

        if not details:
            print(f"Failed to parse details ID {job_id}")
            return None

        min_exp, max_exp = parse_experience(details["experience"])
        posted_date = format_date_iso(details["posted"])

        payload = {
            "title": details["title"] or None,
            "original_description": details["description"] or None,
            "job_type": details["job_type"] or None,
            "published_date": posted_date,
            "min_exp": min_exp,
            "max_exp": max_exp,
            "location": details["raw_location"] or None,
            "department": details["department"] or None,
            "industry": details["industry"] or None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        return job_id, payload

    def run(self):
        jobs = self.get_pending_jobs()
        if not jobs:
            print("No pending Zoho jobs found.")
            return

        print(f"Enriching {len(jobs)} jobs with {MAX_WORKERS} workers...")

        updates = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = [exe.submit(self.scrape_single_job, job) for job in jobs]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    updates.append(res)

        print(f"Applying {len(updates)} updates to {JOBS_TABLE}...")
        for job_id, payload in updates:
            try:
                supabase.table(JOBS_TABLE).update(payload).eq("id", job_id).execute()
                print(f"Updated ID {job_id}")
            except Exception as e:
                print(f"DB Error ID {job_id}: {e}")

        print("--- STAGE 2 COMPLETE ---")


def export_zoho_jobs_backup_from_db():
    print("\nExporting FINAL Zoho jobs backup (Stage 1 + Stage 2) to CSV...")

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

    df = df[df["ats_name"].astype(str).str.lower().str.contains("zoho recruit", na=False)].copy()

    if "scrapes" in df.columns:
        df.drop(columns=["scrapes"], inplace=True)

    df.to_csv(OUTPUT_JOBS_FILE, index=False)
    print(f"Final backup saved: {OUTPUT_JOBS_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    start = time.time()

    print("Fetching Zoho Recruit companies...")
    companies_df = get_zoho_companies()

    if companies_df.empty:
        print("No Zoho Recruit companies found in ats_website.")
    else:
        discovery = ZohoDiscovery()
        discovery.run(companies_df)

        enrichment = ZohoEnrichment()
        enrichment.run()

        export_zoho_jobs_backup_from_db()

    print(f"\nTotal Runtime: {time.time() - start:.2f}s")

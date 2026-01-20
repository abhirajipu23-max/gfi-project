import os
import time
import re
import atexit
import threading
import requests
import pandas as pd
import uuid
import random

from dotenv import load_dotenv
from datetime import datetime, timezone
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from supabase import create_client, Client

load_dotenv()

# ---------------------------
# Configuration
# ---------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ufnaxahhlblwpdomlybs.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_1d4J1Ll81KwhYPOS40U8mQ_qtCccNsa")
MAX_WORKERS = 5  # Adjust based on system capacity
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Initialize Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Failed to initialize Supabase client: {e}")
    exit(1)

# ---------------------------
# Shared Helpers
# ---------------------------
def format_date_iso(s):
    """Parses date string and returns ISO 8601 format."""
    if not s: return None
    s = s.strip()
    # Cleanup "Posted on" text
    s = re.sub(r"(?i)posted\s*on", "", s).strip()
    
    formats = (
        "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", 
        "%b %d, %Y", "%B %d, %Y", "%d-%b-%Y"
    )
    for fmt in formats:
        try:
            # Return standard DB Date format (YYYY-MM-DD) or ISO
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    return None

def parse_experience(exp_text):
    """Extracts min and max experience from text."""
    if not exp_text: return None, None
    exp_text = exp_text.lower().strip()

    if "fresher" in exp_text or re.fullmatch(r"0\s*(years|yrs)?", exp_text):
        return 0, 0

    # Handle "5+ years"
    plus_match = re.search(r"(\d+)\s*\+", exp_text)
    if plus_match:
        return int(plus_match.group(1)), None

    # Handle "2-5 years" or "2 to 5"
    range_match = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)", exp_text)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    # Handle single number "2 years"
    single_match = re.search(r"(\d+)", exp_text)
    if single_match:
        val = int(single_match.group(1))
        return val, val

    return None, None

def detect_work_mode(full_text):
    """Determines if Remote or Onsite."""
    if not full_text: return "Onsite"
    text_lower = full_text.lower()
    if "remote" in text_lower: return "Remote"
    return "Onsite"

# ==============================================================================
# STAGE 1: DISCOVERY (Find Jobs & Insert to DB)
# ==============================================================================
class ZohoDiscovery:
    def __init__(self):
        self.INSERT_BATCH_SIZE = 100
        self.CHECK_BATCH_SIZE = 50  # Reduced to avoid 400 Bad Request
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
        except: pass

    def _normalize_job(self, scrape_uuid, title="", job_type="", posted="", job_url="", source_url="", company_url="", location="", code_name=""):
        now_iso = datetime.now(timezone.utc).isoformat()
        job_id = uuid.uuid4().int % (2**63 - 1) # BigInt safe

        return {
            "id": job_id,
            "created_at": now_iso,
            "updated_at": now_iso,
            "scrape_id": scrape_uuid,
            "title": title,
            "job_url": job_url,
            "is_active": True,
            "published_date": format_date_iso(posted),
            "job_type": job_type,
            "location": location,
            "original_description": None, # Intentionally None, to be filled by Stage 2
            "internal_slug": None
        }

    def _scrape_zoho_table(self, soup, url, company_url, code_name, scrape_uuid):
        jobs = []
        rows = soup.select("tr.jobDetailRow")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4: continue
            link = cols[0].find("a", class_="jobdetail")
            job_type = link.get_text(strip=True) if link else ""
            job_url_parsed = urljoin(url, link["href"]) if link and link.get("href") else ""
            jobs.append(self._normalize_job(
                scrape_uuid=scrape_uuid, source_url=url, company_url=company_url,
                title=cols[1].get_text(strip=True), job_type=job_type,
                posted=cols[2].get_text(strip=True), location=cols[3].get_text(strip=True),
                job_url=job_url_parsed, code_name=code_name
            ))
        return jobs

    def _scrape_zoho_layout3_cards(self, soup, url, company_url, code_name, scrape_uuid):
        jobs = []
        cards = soup.select("div.cw-filter-joblist")
        if not cards: return []
        for card in cards:
            a = card.select_one("a.cw-3-title")
            if not a: continue
            title = a.get_text(strip=True)
            job_url_parsed = urljoin(url, a.get("href", ""))
            
            location = ""
            loc_el = card.select_one("p.filter-subhead")
            if loc_el:
                exp = loc_el.select_one("span.search-work-experience")
                if exp: exp.extract()
                location = loc_el.get_text(" ", strip=True)

            job_type, posted = "", ""
            type_el = card.select_one("span.cw-full-time")
            if type_el: job_type = type_el.get_text(strip=True)
            posted_el = card.select_one("span.search-date-opened")
            if posted_el: posted = posted_el.get_text(strip=True)

            jobs.append(self._normalize_job(
                scrape_uuid=scrape_uuid, source_url=url, company_url=company_url,
                title=title, job_type=job_type, posted=posted, location=location,
                job_url=job_url_parsed, code_name=code_name
            ))
        return jobs

    def _scrape_bs4(self, url, company_url, code_name, scrape_uuid):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            if r.status_code != 200: return []
            soup = BeautifulSoup(r.text, "html.parser")
            
            jobs = self._scrape_zoho_table(soup, url, company_url, code_name, scrape_uuid)
            if jobs: return jobs
            
            jobs = self._scrape_zoho_layout3_cards(soup, url, company_url, code_name, scrape_uuid)
            if jobs: return jobs
            
            # Fallback
            jobs = []
            anchors = soup.select("a.cw-1-title, a.cw-3-title, a.zp_career_site_job_title, a.jobdetail")
            for a in anchors:
                href = a.get("href")
                if href:
                    jobs.append(self._normalize_job(
                        scrape_uuid=scrape_uuid, source_url=url, company_url=company_url,
                        title=a.get_text(strip=True), job_url=urljoin(url, href), code_name=code_name
                    ))
            return jobs
        except: return []

    def _scrape_playwright(self, url, company_url, code_name, scrape_uuid):
        try:
            browser = self._get_thread_browser()
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, timeout=30000)
            page.wait_for_timeout(2000)
            
            # Simplified Playwright logic for brevity (Table + Generic)
            jobs = []
            rows = page.locator("tr.jobDetailRow")
            if rows.count() > 0:
                for i in range(rows.count()):
                    cols = rows.nth(i).locator("td")
                    if cols.count() >= 4:
                        link = cols.nth(0).locator("a.jobdetail")
                        jobs.append(self._normalize_job(
                            scrape_uuid=scrape_uuid, source_url=url, company_url=company_url,
                            title=cols.nth(1).inner_text().strip(),
                            job_url=urljoin(url, link.get_attribute("href") or ""),
                            posted=cols.nth(2).inner_text().strip(),
                            location=cols.nth(3).inner_text().strip(),
                            code_name=code_name
                        ))
                return jobs
                
            # Generic
            links = page.locator("a.cw-1-title, a.cw-3-title")
            for i in range(links.count()):
                jobs.append(self._normalize_job(
                    scrape_uuid=scrape_uuid, source_url=url, company_url=company_url,
                    title=links.nth(i).inner_text().strip(),
                    job_url=urljoin(url, links.nth(i).get_attribute("href") or ""),
                    code_name=code_name
                ))
            page.close()
            return jobs
        except: 
            return []

    def process_company(self, row):
        if not row.get("ats_url"): return None
        scrape_uuid = str(uuid.uuid4())
        record = {
            "id": scrape_uuid, "ats_website_id": row["id"], "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Try BS4 then Playwright
        jobs = self._scrape_bs4(row["ats_url"], row.get("company_name"), row.get("code_name"), scrape_uuid)
        if not jobs:
            jobs = self._scrape_playwright(row["ats_url"], row.get("company_name"), row.get("code_name"), scrape_uuid)

        # Update record status
        record["status"] = "success" if jobs else "failed"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        
        # Dedupe within this run
        seen = {}
        unique_jobs = []
        for j in jobs:
            if j["job_url"] and j["job_url"] not in seen:
                seen[j["job_url"]] = True
                unique_jobs.append(j)

        return (record, unique_jobs)

    def save_to_db(self, scrapes, jobs):
        if scrapes:
            try:
                supabase.table("scrapes_duplicate").upsert(scrapes).execute()
                print("✅ Scrape records synced.")
            except Exception as e: print(f"Error saving scrapes: {e}")

        if not jobs: return
        
        # --- BATCH CHECKING ---
        urls = [j["job_url"] for j in jobs if j["job_url"]]
        existing = set()
        print(f"Checking {len(urls)} jobs for duplicates...")

        # Process in smaller batches to avoid 400 Bad Request (URL too long)
        for i in range(0, len(urls), self.CHECK_BATCH_SIZE):
            batch = urls[i : i + self.CHECK_BATCH_SIZE]
            try:
                res = supabase.table("jobs_duplicate").select("job_url").in_("job_url", batch).execute()
                for r in res.data:
                    existing.add(r["job_url"])
            except Exception as e:
                print(f"⚠️ Error checking duplicates (Batch {i}): {e}")

        new_jobs = [j for j in jobs if j["job_url"] not in existing]
        print(f" -> Found {len(new_jobs)} NEW jobs to insert into jobs_duplicate.")

        if new_jobs:
            for i in range(0, len(new_jobs), self.INSERT_BATCH_SIZE):
                try:
                    supabase.table("jobs_duplicate").insert(new_jobs[i:i+self.INSERT_BATCH_SIZE]).execute()
                    print(f"    ↳ Inserted batch {i} to {i+self.INSERT_BATCH_SIZE}")
                except Exception as e: 
                    print(f"Error inserting jobs: {e}")

    def run(self, df):
        print(f"--- STAGE 1: DISCOVERY ({len(df)} companies) ---")
        all_scrapes, all_jobs = [], []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = [exe.submit(self.process_company, row) for _, row in df.iterrows()]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    all_scrapes.append(res[0])
                    all_jobs.extend(res[1])
        
        self.save_to_db(all_scrapes, all_jobs)
        print("--- STAGE 1 COMPLETE ---")

# ==============================================================================
# STAGE 2: ENRICHMENT (Scrape Details for Jobs in DB)
# ==============================================================================
class ZohoEnrichment:
    def get_pending_jobs(self):
        """Fetches jobs from jobs_duplicate that are Zoho Recruit and missing descriptions."""
        print("--- STAGE 2: ENRICHMENT (Fetching pending jobs) ---")
        try:
            # Join with scrapes_duplicate -> ats_website to ensure it's Zoho
            # And original_description is null (needs scraping)
            res = (
                supabase.table("jobs_duplicate")
                .select("""
                    id, job_url,
                    scrapes_duplicate!inner (
                        ats_website!inner ( ats_name )
                    )
                """)
                .is_("original_description", "null") # Only scrape if empty
                .execute()
            )
            
            all_data = res.data
            zoho_jobs = [
                j for j in all_data 
                if j["scrapes_duplicate"]["ats_website"]["ats_name"].strip().lower() == "zoho recruit"
            ]
            print(f"Found {len(zoho_jobs)} Zoho jobs waiting for details.")
            return zoho_jobs
        except Exception as e:
            print(f"Error fetching pending jobs: {e}")
            return []

    def parse_job_details_from_soup(self, soup):
        data = {
            "title": "", "posted": "", "experience": "", "job_type": "",
            "raw_location": "", "description": ""
        }

        # Extract Title
        for sel in ["div.tem3-titleBlock h1", "div.tem4-col2_hdr h2", 
                    "div.cw-jobheader-information h1", "h1", "h2"]:
            el = soup.select_one(sel)
            if el:
                data["title"] = el.get_text(" ", strip=True).split("\n")[0].strip()
                break

        # Extract Posted Date
        post_el = soup.select_one("div.tem3-postDate")
        if post_el and not data["posted"]:
            post_text = post_el.get_text(" ", strip=True)
            m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", post_text)
            if m: data["posted"] = m.group(1)

        # Extract Description
        for sel in ["div.job-dec-block_new", "div.cw-jobdescription", 
                    "div.temp4-content", "span#spandesc", "div.tem3ContentBlock"]:
            el = soup.select_one(sel)
            if el:
                data["description"] = el.get_text("\n", strip=True)
                break

        # Helper to apply labels found in lists/blocks
        def apply_label_value(label, value):
            if not label or not value: return
            label = label.lower().strip()
            value = value.strip()

            if any(x in label for x in ["date opened", "posted", "date"]):
                if not data["posted"]: data["posted"] = value
            elif "job type" in label:
                if not data["job_type"]: data["job_type"] = value
            elif any(x in label for x in ["work experience", "required experience", "experience"]):
                if not data["experience"]: data["experience"] = value
            elif any(x in label for x in ["location", "city", "state", "region"]):
                if not data["raw_location"]: 
                    data["raw_location"] = value
                else:
                    data["raw_location"] += f", {value}"

        # Parse Summary Lists
        summary_items = soup.select("div.cw-summary ul.cw-summary-list li")
        for li in summary_items:
            spans = li.find_all("span")
            if len(spans) < 2: continue
            apply_label_value(spans[0].get_text(strip=True), spans[1].get_text(strip=True))

        # Parse Tem3 Blocks
        tem3_blocks = soup.select("div.tem3ContentBlock div.tem3-TextCont")
        for block in tem3_blocks:
            label_el = block.find("span")
            value_el = block.select_one("div.tem3-RightSideText")
            if label_el and value_el:
                apply_label_value(label_el.get_text(" ", strip=True), value_el.get_text(" ", strip=True))

        return data if data["title"] else None

    def scrape_single_job(self, job_row):
        url = job_row["job_url"]
        job_id = job_row["id"]
        
        details = None
        
        # 1. Try BS4
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            if r.status_code == 200:
                details = self.parse_job_details_from_soup(BeautifulSoup(r.text, "html.parser"))
        except: pass

        # 2. Try Playwright if BS4 failed
        if not details:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page(user_agent=USER_AGENT)
                    page.goto(url, timeout=45000)
                    content = page.content()
                    browser.close()
                    details = self.parse_job_details_from_soup(BeautifulSoup(content, "html.parser"))
            except Exception as e:
                print(f"   [ID {job_id}] Playwright failed: {e}")

        if not details:
            print(f"   ❌ [ID {job_id}] Failed to parse details.")
            return None

        # Process Extracted Data
        min_exp, max_exp = parse_experience(details["experience"])
        posted_date = format_date_iso(details["posted"])
        location = details["raw_location"]
        
        # Logic: Combine fields to check for "Remote"
        combined_text = f"{location} {details['description']} {details['job_type']}"
        work_type = detect_work_mode(combined_text)

        payload = {
            "title": details["title"],
            "original_description": details["description"],
            "job_type": work_type,
            "published_date": posted_date,
            "min_exp": min_exp,
            "max_exp": max_exp,
            "location": location,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Remove None values to avoid overwriting existing data with nulls (optional, but safer)
        clean_payload = {k: v for k, v in payload.items() if v is not None}
        return (job_id, clean_payload)

    def run(self):
        jobs = self.get_pending_jobs()
        if not jobs: return

        print(f"Enriching {len(jobs)} jobs with {MAX_WORKERS} workers...")
        
        updates = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = [exe.submit(self.scrape_single_job, job) for job in jobs]
            for f in as_completed(futures):
                res = f.result()
                if res: updates.append(res)

        print(f"Applying {len(updates)} updates to jobs_duplicate...")
        for job_id, payload in updates:
            try:
                supabase.table("jobs_duplicate").update(payload).eq("id", job_id).execute()
                print(f"   ✅ [ID {job_id}] Updated")
            except Exception as e:
                print(f"   ❌ [ID {job_id}] DB Error: {e}")
        print("--- STAGE 2 COMPLETE ---")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # 1. Fetch Company List
    def get_zoho_companies():
        try:
            res = supabase.table("ats_website").select("*").execute()
            df = pd.DataFrame(res.data)
            if "ats_name" in df.columns:
                return df[df["ats_name"].str.lower().str.strip() == "zoho recruit"]
            return df
        except: return pd.DataFrame()

    print("Fetching company list...")
    companies_df = get_zoho_companies()
    
    if not companies_df.empty:
        discovery = ZohoDiscovery()
        discovery.run(companies_df)
        
        enrichment = ZohoEnrichment()
        enrichment.run()
    else:
        print("No Zoho Recruit companies found in ats_website.")
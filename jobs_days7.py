import pandas as pd
import asyncio
import nest_asyncio
import time
import uuid
import re
from datetime import datetime, timezone

# Playwright
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

# Database (Global)
from zoho_git import ZohoDiscovery, ZohoEnrichment, supabase

# --- IMPORTS FROM KEKA ---
from keka_git import (
    scrape_keka_jobs,
    save_to_supabase as save_keka_db,
    get_companies as get_keka_companies_base,
    clean_text as keka_clean_text,
    update_supabase_record,
    MAX_CONCURRENT_PAGES as KEKA_MAX_PAGES
)

# --- IMPORTS FROM DARWINBOX ---
from darwinbox_git import (
    scrape_darwinbox_discovery,
    save_to_supabase as save_darwin_db,
    process_job as process_darwin_job,
    get_companies as get_darwin_companies_base,
    MAX_CONCURRENT_PAGES as DARWIN_MAX_PAGES
)

# --- IMPORTS FROM WORKABLE ---
from workable_git import (
    WorkableDiscovery,
    WorkableEnrichment,
    SupabaseManager as WorkableDBManager
)

# --- IMPORTS FROM FRESHTEAM ---
from freshteam_git import (
    FreshteamDiscovery,
    FreshteamEnrichment,
    SupabaseManager as FreshteamDBManager
)

# --- IMPORTS FROM SMARTRECRUITERS ---
from smart_git import (
    SmartRecruitersDiscovery,
    SmartRecruitersEnrichment,
    SupabaseManager as SmartRecruitersDBManager
)

# --- IMPORTS FROM LEVER ---
from lever_git import (
    LeverDiscovery,
    LeverEnrichment,
    SupabaseManager as LeverDBManager
)

# Apply nest_asyncio to allow nested event loops
try:
    nest_asyncio.apply()
except:
    pass


# =========================================================
# 1. STATS FUNCTION (Identify High Volume URLs)
# =========================================================
def get_filtered_ats_stats():
    """Returns a list of ATS URLs with > 15 jobs."""
    print("--- Calculating Job Stats ---")

    jobs_res = supabase.table("jobs_duplicate").select("scrape_id").execute()
    jobs_df = pd.DataFrame(jobs_res.data)

    scrapes_res = supabase.table("scrapes_duplicate").select("id, ats_website_id").execute()
    scrapes_df = pd.DataFrame(scrapes_res.data)

    ats_res = supabase.table("ats_website").select("id, ats_url").execute()
    ats_df = pd.DataFrame(ats_res.data)

    if jobs_df.empty or scrapes_df.empty or ats_df.empty:
        return pd.DataFrame()

    merged = pd.merge(jobs_df, scrapes_df, left_on='scrape_id', right_on='id', how='inner')
    final_df = pd.merge(merged, ats_df, left_on='ats_website_id', right_on='id', how='inner')

    ats_counts = final_df.groupby('ats_url').size().reset_index(name='job_count')

    # Filter > 15
    filtered_results = ats_counts[ats_counts['job_count'] > 15].sort_values(by='job_count', ascending=False)
    return filtered_results


# =========================================================
# 2. ZOHO RUNNER
# =========================================================
def run_zoho_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING ZOHO SCRAPER")
    print("==================================")

    # Helper to get filtered Zoho companies
    def get_zoho_companies():
        res = supabase.table("ats_website").select("*").execute()
        df = pd.DataFrame(res.data)
        if "ats_name" in df.columns:
            df = df[df["ats_name"].str.lower().str.strip() == "zoho recruit"]
        if restrict_to_urls is not None and not df.empty:
            df = df[df["ats_url"].isin(restrict_to_urls)]
        return df

    companies_df = get_zoho_companies()

    if not companies_df.empty:
        print(f"   -> Targeting {len(companies_df)} Zoho Companies")
        ZohoDiscovery().run(companies_df)
        ZohoEnrichment().run()
    else:
        print("   -> No Zoho Recruit companies matched the filter.")


# =========================================================
# 3. KEKA RUNNER (Custom Filtered Stages)
# =========================================================
async def run_keka_stage_2_custom(target_urls=None):
    print(">> Keka Stage 2: Enrichment (Custom Filtered)")

    # Fetch pending jobs
    res = supabase.table("jobs_duplicate").select("id, job_url").is_("original_description", "null").execute()
    if not res.data:
        return

    df = pd.DataFrame(res.data)

    # Filter by URL
    if target_urls:
        pattern = '|'.join([re.escape(url) for url in target_urls])
        keka_df = df[df["job_url"].str.contains(pattern, case=False, na=False)].copy()
    else:
        keka_df = df[df["job_url"].str.contains("keka.com", case=False, na=False)].copy()

    if keka_df.empty:
        print("   -> No pending Keka jobs found for target URLs.")
        return

    print(f"   -> Enriching {len(keka_df)} Keka jobs...")

    async def keka_worker(browser, row, sem):
        async with sem:
            page = await browser.new_page()
            try:
                await page.goto(row["job_url"], timeout=20000)
                try:
                    desc = await page.locator(".job-description-container").first.inner_text()
                    desc = keka_clean_text(desc)
                except:
                    desc = ""

                if desc:
                    await asyncio.to_thread(update_supabase_record, row["id"], {"original_description": desc})
            except:
                pass
            finally:
                await page.close()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        sem = asyncio.Semaphore(KEKA_MAX_PAGES)
        tasks = [keka_worker(browser, row, sem) for _, row in keka_df.iterrows()]
        await asyncio.gather(*tasks)
        await browser.close()


def run_keka_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING KEKA SCRAPER")
    print("==================================")

    # --- Stage 1 ---
    print(">> Keka Stage 1: Discovery")
    df = get_keka_companies_base()
    if "ats_name" in df.columns:
        df = df[df["ats_name"].astype(str).str.lower().str.strip() == "keka"]

    if restrict_to_urls:
        df = df[df["ats_url"].isin(restrict_to_urls)]

    if not df.empty:
        print(f"   -> Targeting {len(df)} Keka Companies")
        all_jobs, all_scrapes = [], []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            for _, row in df.iterrows():
                suid = str(uuid.uuid4())
                s_rec = {
                    "id": suid,
                    "ats_website_id": row["id"],
                    "status": "pending",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                try:
                    jobs = scrape_keka_jobs(page, row["ats_url"], suid)
                    if jobs:
                        all_jobs.extend(jobs)
                    s_rec["status"] = "success"
                except:
                    s_rec["status"] = "failed"
                all_scrapes.append(s_rec)
            browser.close()
        save_keka_db(all_scrapes, [j for j in all_jobs if j["job_url"]])
    else:
        print("   -> No Keka companies matched.")

    # --- Stage 2 ---
    try:
        asyncio.run(run_keka_stage_2_custom(target_urls=restrict_to_urls))
    except Exception as e:
        print(f"Keka Stage 2 Error: {e}")


# =========================================================
# 4. DARWINBOX RUNNER (Integrated)
# =========================================================
async def run_darwinbox_stage_2_custom(target_urls=None):
    print(">> Darwinbox Stage 2: Enrichment (Custom Filtered)")

    try:
        res = supabase.table("jobs_duplicate").select(
            "id, job_url, location, job_type, department"
        ).is_("original_description", "null").execute()
    except Exception as e:
        print(f"   Error fetching DB jobs: {e}")
        return

    if not res.data:
        print("   -> No pending jobs in DB.")
        return

    df = pd.DataFrame(res.data)

    if target_urls:
        clean_targets = [u.rstrip('/') for u in target_urls]
        pattern = '|'.join([re.escape(u) for u in clean_targets])
        darwin_df = df[df["job_url"].str.contains(pattern, case=False, na=False)].copy()
    else:
        darwin_df = df[df["job_url"].str.contains("darwinbox", case=False, na=False)].copy()

    if darwin_df.empty:
        print("   -> No pending Darwinbox jobs found for target URLs.")
        return

    print(f"   -> Enriching {len(darwin_df)} Darwinbox jobs...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        sem = asyncio.Semaphore(DARWIN_MAX_PAGES)
        jobs_list = darwin_df.to_dict('records')

        tasks = [process_darwin_job(sem, context, job) for job in jobs_list]
        await asyncio.gather(*tasks)

        await browser.close()
    print("   ✅ Darwinbox Enrichment Complete.")


def run_darwinbox_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING DARWINBOX SCRAPER")
    print("==================================")

    print(">> Darwinbox Stage 1: Discovery")
    df = get_darwin_companies_base()

    if "ats_name" in df.columns:
        df = df[df["ats_name"].astype(str).str.lower().str.strip() == "darwinbox"]

    if restrict_to_urls:
        print(f"   (Filtering List to match {len(restrict_to_urls)} URLs...)")
        df = df[df["ats_url"].isin(restrict_to_urls)]

    if not df.empty:
        print(f"   -> Targeting {len(df)} Darwinbox Companies")
        all_jobs, all_scrapes = [], []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            for _, row in df.iterrows():
                scrape_uuid = str(uuid.uuid4())
                scrape_record = {
                    "id": scrape_uuid,
                    "ats_website_id": row["id"],
                    "status": "pending",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }

                try:
                    jobs = scrape_darwinbox_discovery(page, row["ats_url"], scrape_uuid)
                    if jobs:
                        all_jobs.extend(jobs)
                    scrape_record["status"] = "success"
                    scrape_record["finished_at"] = datetime.now(timezone.utc).isoformat()
                except Exception as e:
                    print(f"   ❌ Error on {row['ats_url']}: {e}")
                    scrape_record["status"] = "failed"

                all_scrapes.append(scrape_record)
            browser.close()

        clean_jobs = [j for j in all_jobs if j["job_url"] and j["job_url"] != "NA"]
        save_darwin_db(all_scrapes, clean_jobs)
    else:
        print("   -> No Darwinbox companies matched.")

    try:
        asyncio.run(run_darwinbox_stage_2_custom(target_urls=restrict_to_urls))
    except Exception as e:
        print(f"Darwinbox Stage 2 Error: {e}")


# =========================================================
# 5. WORKABLE RUNNER
# =========================================================
async def run_workable_stage_2_custom(db_manager, target_urls=None):
    print(">> Workable Stage 2: Enrichment (Custom Filtered)")

    all_pending = db_manager.fetch_pending_jobs(ats_filter="Workable")

    if not all_pending:
        print("   -> No pending Workable jobs.")
        return

    filtered_jobs = []
    if target_urls:
        clean_targets = [u.rstrip('/') for u in target_urls]
        for job in all_pending:
            j_url = job.get("job_url", "")
            if any(t in j_url for t in clean_targets):
                filtered_jobs.append(job)
    else:
        filtered_jobs = all_pending

    if not filtered_jobs:
        print("   -> No Workable jobs matched the target URLs.")
        return

    print(f"   -> Enriching {len(filtered_jobs)} Workable jobs...")

    enricher = WorkableEnrichment(db_manager, concurrency=8)
    sem = asyncio.Semaphore(enricher.concurrency)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        tasks = [enricher._process_job(sem, context, job) for job in filtered_jobs]
        await asyncio.gather(*tasks)

        await browser.close()
    print("   ✅ Workable Enrichment Complete.")


def run_workable_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING WORKABLE SCRAPER")
    print("==================================")

    w_db = WorkableDBManager()

    print(">> Workable Stage 1: Discovery")
    companies = w_db.fetch_companies(ats_name="workable")

    if restrict_to_urls:
        companies = companies[companies["ats_url"].isin(restrict_to_urls)]

    if not companies.empty:
        print(f"   -> Targeting {len(companies)} Workable Companies")

        discovery = WorkableDiscovery(w_db)
        all_jobs, all_scrapes = [], []

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
                    jobs = discovery.scrape_site(page, row["ats_url"], scrape_uuid)
                    if jobs:
                        all_jobs.extend(jobs)
                    scrape_record["status"] = "success"
                    scrape_record["finished_at"] = datetime.now(timezone.utc).isoformat()
                except Exception as e:
                    print(f"   ❌ Failed on {row['ats_url']}: {e}")
                    scrape_record["status"] = "failed"
                    scrape_record["finished_at"] = datetime.now(timezone.utc).isoformat()

                all_scrapes.append(scrape_record)
            browser.close()

        clean_jobs = [j for j in all_jobs if j["job_url"] and j["job_url"] != "NA"]
        w_db.save_scrapes(all_scrapes)
        w_db.save_jobs_deduplicated(clean_jobs)
    else:
        print("   -> No Workable companies matched.")

    try:
        asyncio.run(run_workable_stage_2_custom(w_db, target_urls=restrict_to_urls))
    except Exception as e:
        print(f"Workable Stage 2 Error: {e}")


# =========================================================
# 6. SMARTRECRUITERS RUNNER
# =========================================================
async def run_smartrecruiters_stage_2_custom(db_manager, target_urls=None):
    print(">> SmartRecruiters Stage 2: Enrichment (Custom Filtered)")

    all_pending = db_manager.fetch_pending_jobs(ats_filter="SmartRecruiters")

    if not all_pending:
        print("   -> No pending SmartRecruiters jobs.")
        return

    filtered_jobs = []
    if target_urls:
        clean_targets = [u.rstrip('/') for u in target_urls]
        for job in all_pending:
            j_url = job.get("job_url", "")
            if any(t in j_url for t in clean_targets):
                filtered_jobs.append(job)
    else:
        filtered_jobs = all_pending

    if not filtered_jobs:
        print("   -> No SmartRecruiters jobs matched the target URLs.")
        return

    print(f"   -> Enriching {len(filtered_jobs)} SmartRecruiters jobs...")

    enricher = SmartRecruitersEnrichment(db_manager, concurrency=10)
    sem = asyncio.Semaphore(enricher.concurrency)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        tasks = [enricher._process_job(sem, context, job) for job in filtered_jobs]
        await asyncio.gather(*tasks)

        await browser.close()
    print("   ✅ SmartRecruiters Enrichment Complete.")


def run_smartrecruiters_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING SMARTRECRUITERS SCRAPER")
    print("==================================")

    sr_db = SmartRecruitersDBManager()

    print(">> SmartRecruiters Stage 1: Discovery")
    companies = sr_db.fetch_companies(ats_name="smartrecruiters")

    if restrict_to_urls:
        companies = companies[companies["ats_url"].isin(restrict_to_urls)]

    if not companies.empty:
        print(f"   -> Targeting {len(companies)} SmartRecruiters Companies")

        discovery = SmartRecruitersDiscovery(sr_db)
        all_jobs, all_scrapes = [], []

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
                    jobs = discovery.scrape_site(page, row["ats_url"], scrape_uuid)
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
        sr_db.save_scrapes(all_scrapes)
        sr_db.save_jobs_deduplicated(clean_jobs)
    else:
        print("   -> No SmartRecruiters companies matched.")

    try:
        asyncio.run(run_smartrecruiters_stage_2_custom(sr_db, target_urls=restrict_to_urls))
    except Exception as e:
        print(f"SmartRecruiters Stage 2 Error: {e}")


# =========================================================
# 7. FRESHTEAM RUNNER
# =========================================================
async def run_freshteam_stage_2_custom(db_manager, target_urls=None):
    print(">> Freshteam Stage 2: Enrichment (Custom Filtered)")

    all_pending = db_manager.fetch_pending_jobs(ats_filter="Freshteam")

    if not all_pending:
        print("   -> No pending Freshteam jobs.")
        return

    filtered_jobs = []
    if target_urls:
        clean_targets = [u.rstrip('/') for u in target_urls]
        for job in all_pending:
            j_url = job.get("job_url", "")
            if any(t in j_url for t in clean_targets):
                filtered_jobs.append(job)
    else:
        filtered_jobs = all_pending

    if not filtered_jobs:
        print("   -> No Freshteam jobs matched the target URLs.")
        return

    print(f"   -> Enriching {len(filtered_jobs)} Freshteam jobs...")

    enricher = FreshteamEnrichment(db_manager, concurrency=8)
    sem = asyncio.Semaphore(enricher.concurrency)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        tasks = [enricher._process_job(sem, context, job) for job in filtered_jobs]
        await asyncio.gather(*tasks)

        await browser.close()
    print("   ✅ Freshteam Enrichment Complete.")


def run_freshteam_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING FRESHTEAM SCRAPER")
    print("==================================")

    ft_db = FreshteamDBManager()

    print(">> Freshteam Stage 1: Discovery")
    companies = ft_db.fetch_companies(ats_name="freshteam")

    if restrict_to_urls:
        companies = companies[companies["ats_url"].isin(restrict_to_urls)]

    if not companies.empty:
        print(f"   -> Targeting {len(companies)} Freshteam Companies")
        discovery = FreshteamDiscovery(ft_db)

        all_jobs, all_scrapes = [], []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            for _, row in companies.iterrows():
                scrape_uuid = str(uuid.uuid4())
                scrape_record = {
                    "id": scrape_uuid,
                    "ats_website_id": row["id"],
                    "status": "pending",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }

                try:
                    jobs = discovery.scrape_site(page, row["ats_url"], scrape_uuid)
                    if jobs:
                        all_jobs.extend(jobs)
                    scrape_record["status"] = "success"
                    scrape_record["finished_at"] = datetime.now(timezone.utc).isoformat()
                except Exception as e:
                    print(f"   ❌ Failed {row['ats_url']}: {e}")
                    scrape_record["status"] = "failed"

                all_scrapes.append(scrape_record)
            browser.close()

        clean_jobs = [j for j in all_jobs if j["job_url"] and j["job_url"] != "NA"]
        ft_db.save_scrapes(all_scrapes)
        ft_db.save_jobs_deduplicated(clean_jobs)
    else:
        print("   -> No Freshteam companies matched.")

    try:
        asyncio.run(run_freshteam_stage_2_custom(ft_db, target_urls=restrict_to_urls))
    except Exception as e:
        print(f"Freshteam Stage 2 Error: {e}")


# =========================================================
# 8. LEVER RUNNER (NEW)
# =========================================================
async def run_lever_stage_2_custom(db_manager, target_urls=None):
    print(">> Lever Stage 2: Enrichment (Custom Filtered)")

    all_pending = db_manager.fetch_pending_jobs(ats_filter="Lever")

    if not all_pending:
        print("   -> No pending Lever jobs.")
        return

    filtered_jobs = []
    if target_urls:
        clean_targets = [u.rstrip('/') for u in target_urls]
        for job in all_pending:
            j_url = job.get("job_url", "")
            if any(t in j_url for t in clean_targets):
                filtered_jobs.append(job)
    else:
        filtered_jobs = all_pending

    if not filtered_jobs:
        print("   -> No Lever jobs matched the target URLs.")
        return

    print(f"   -> Enriching {len(filtered_jobs)} Lever jobs...")

    enricher = LeverEnrichment(db_manager, concurrency=8)
    sem = asyncio.Semaphore(enricher.concurrency)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        tasks = [enricher._process_job(sem, context, job) for job in filtered_jobs]
        await asyncio.gather(*tasks)

        await browser.close()
    print("   ✅ Lever Enrichment Complete.")


def run_lever_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING LEVER SCRAPER")
    print("==================================")

    lever_db = LeverDBManager()

    print(">> Lever Stage 1: Discovery")
    companies = lever_db.fetch_companies(ats_name="lever")

    if restrict_to_urls:
        companies = companies[companies["ats_url"].isin(restrict_to_urls)]

    if not companies.empty:
        print(f"   -> Targeting {len(companies)} Lever Companies")

        discovery = LeverDiscovery(lever_db)
        discovery.run()
    else:
        print("   -> No Lever companies matched.")

    try:
        asyncio.run(run_lever_stage_2_custom(lever_db, target_urls=restrict_to_urls))
    except Exception as e:
        print(f"Lever Stage 2 Error: {e}")


# =========================================================
# MAIN EXECUTION BLOCK
# =========================================================
if __name__ == "__main__":
    print("Starting Job Stats Analysis...")

    stats_df = get_filtered_ats_stats()

    zoho_targets = []
    keka_targets = []
    darwin_targets = []
    workable_targets = []
    freshteam_targets = []
    smartrecruiters_targets = []
    lever_targets = []

    if not stats_df.empty:
        print("\n--- High Volume ATS Sites (>15 jobs) ---")
        print(stats_df)

        high_vol_urls = stats_df['ats_url'].tolist()

        for url in high_vol_urls:
            u_lower = url.lower()

            if "zoho" in u_lower:
                zoho_targets.append(url)
            elif "keka" in u_lower:
                keka_targets.append(url)
            elif "darwinbox" in u_lower:
                darwin_targets.append(url)
            elif "workable" in u_lower:
                workable_targets.append(url)
            elif "freshteam" in u_lower:
                freshteam_targets.append(url)
            elif "smartrecruiters" in u_lower:
                smartrecruiters_targets.append(url)
            elif "lever" in u_lower:
                lever_targets.append(url)

    else:
        print("\nNo high volume sites found. (Or DB connection issue).")

    # 2. Run Scrapers (Only for High Volume targets)
    if zoho_targets:
        run_zoho_scraper(restrict_to_urls=zoho_targets)

    if keka_targets:
        run_keka_scraper(restrict_to_urls=keka_targets)

    if darwin_targets:
        run_darwinbox_scraper(restrict_to_urls=darwin_targets)

    if workable_targets:
        run_workable_scraper(restrict_to_urls=workable_targets)

    if freshteam_targets:
        run_freshteam_scraper(restrict_to_urls=freshteam_targets)

    if smartrecruiters_targets:
        run_smartrecruiters_scraper(restrict_to_urls=smartrecruiters_targets)

    if lever_targets:
        run_lever_scraper(restrict_to_urls=lever_targets)

    print("\n\n✅ ALL JOBS PROCESSED.")

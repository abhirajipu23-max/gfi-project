import pandas as pd
import asyncio
import nest_asyncio

from zoho_git import ZohoDiscovery, ZohoEnrichment, supabase

from keka_git import (
    scrape_keka_jobs,
    save_to_supabase as save_keka_db,
    get_companies as get_keka_companies_base,
    clean_text as keka_clean_text,
    update_supabase_record,
    MAX_CONCURRENT_PAGES as KEKA_MAX_PAGES
)

from darwinbox_git import (
    scrape_darwinbox_discovery,
    save_to_supabase as save_darwin_db,
    process_job as process_darwin_job,
    get_companies as get_darwin_companies_base,
    MAX_CONCURRENT_PAGES as DARWIN_MAX_PAGES
)

from workable_git import (
    WorkableDiscovery,
    WorkableEnrichment,
    SupabaseManager as WorkableDBManager
)

from freshteam_git import (
    FreshteamDiscovery,
    FreshteamEnrichment,
    SupabaseManager as FreshteamDBManager
)

from smart_git import (
    SmartRecruitersDiscovery,
    SmartRecruitersEnrichment,
    SupabaseManager as SmartRecruitersDBManager
)

from lever_git import (
    LeverDiscovery,
    LeverEnrichment,
    SupabaseManager as LeverDBManager
)

from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
import uuid
import re
from datetime import datetime, timezone

try:
    nest_asyncio.apply()
except:
    pass


# -------------------------------
# Fetch ALL ats_website rows
# -------------------------------
def get_all_ats_websites():
    res = supabase.table("ats_website").select("*").execute()
    return pd.DataFrame(res.data)


# -------------------------------
# ZOHO
# -------------------------------
def run_zoho_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING ZOHO SCRAPER")
    print("==================================")

    res = supabase.table("ats_website").select("*").execute()
    df = pd.DataFrame(res.data)

    if df.empty:
        print("   -> No ATS websites found.")
        return

    df = df[df["ats_name"].astype(str).str.lower().str.strip() == "zoho recruit"]

    if restrict_to_urls is not None:
        df = df[df["ats_url"].isin(restrict_to_urls)]

    if df.empty:
        print("   -> No Zoho Recruit companies matched.")
        return

    print(f"   -> Targeting {len(df)} Zoho Companies")
    ZohoDiscovery().run(df)
    ZohoEnrichment().run()


# -------------------------------
# KEKA
# -------------------------------
async def run_keka_stage_2_custom(target_urls=None):
    print(">> Keka Stage 2: Enrichment (Custom Filtered)")

    res = supabase.table("jobs_duplicate").select("id, job_url").is_("original_description", "null").execute()
    if not res.data:
        return

    df = pd.DataFrame(res.data)

    if target_urls:
        pattern = '|'.join([re.escape(url) for url in target_urls])
        keka_df = df[df["job_url"].str.contains(pattern, case=False, na=False)].copy()
    else:
        keka_df = df[df["job_url"].str.contains("keka.com", case=False, na=False)].copy()

    if keka_df.empty:
        print("   -> No pending Keka jobs found.")
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

    df = get_keka_companies_base()
    df = df[df["ats_name"].astype(str).str.lower().str.strip() == "keka"]

    if restrict_to_urls:
        df = df[df["ats_url"].isin(restrict_to_urls)]

    if df.empty:
        print("   -> No Keka companies matched.")
        return

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

    try:
        asyncio.run(run_keka_stage_2_custom(target_urls=restrict_to_urls))
    except Exception as e:
        print(f"Keka Stage 2 Error: {e}")


# -------------------------------
# DARWINBOX
# -------------------------------
async def run_darwinbox_stage_2_custom(target_urls=None):
    print(">> Darwinbox Stage 2: Enrichment (Custom Filtered)")

    res = supabase.table("jobs_duplicate").select(
        "id, job_url, location, job_type, department"
    ).is_("original_description", "null").execute()

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
        print("   -> No pending Darwinbox jobs found.")
        return

    print(f"   -> Enriching {len(darwin_df)} Darwinbox jobs...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"
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

    df = get_darwin_companies_base()
    df = df[df["ats_name"].astype(str).str.lower().str.strip() == "darwinbox"]

    if restrict_to_urls:
        df = df[df["ats_url"].isin(restrict_to_urls)]

    if df.empty:
        print("   -> No Darwinbox companies matched.")
        return

    print(f"   -> Targeting {len(df)} Darwinbox Companies")

    all_jobs, all_scrapes = [], []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"
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

    try:
        asyncio.run(run_darwinbox_stage_2_custom(target_urls=restrict_to_urls))
    except Exception as e:
        print(f"Darwinbox Stage 2 Error: {e}")


# -------------------------------
# WORKABLE / FRESHTEAM / SMARTRECRUITERS / LEVER
# (kept same structure)
# -------------------------------
def run_workable_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING WORKABLE SCRAPER")
    print("==================================")

    w_db = WorkableDBManager()
    companies = w_db.fetch_companies(ats_name="workable")

    if restrict_to_urls:
        companies = companies[companies["ats_url"].isin(restrict_to_urls)]

    if companies.empty:
        print("   -> No Workable companies matched.")
        return

    print(f"   -> Targeting {len(companies)} Workable Companies")
    discovery = WorkableDiscovery(w_db)
    discovery.run()


def run_freshteam_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING FRESHTEAM SCRAPER")
    print("==================================")

    ft_db = FreshteamDBManager()
    companies = ft_db.fetch_companies(ats_name="freshteam")

    if restrict_to_urls:
        companies = companies[companies["ats_url"].isin(restrict_to_urls)]

    if companies.empty:
        print("   -> No Freshteam companies matched.")
        return

    print(f"   -> Targeting {len(companies)} Freshteam Companies")
    discovery = FreshteamDiscovery(ft_db)
    discovery.run()


def run_smartrecruiters_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING SMARTRECRUITERS SCRAPER")
    print("==================================")

    sr_db = SmartRecruitersDBManager()
    companies = sr_db.fetch_companies(ats_name="smartrecruiters")

    if restrict_to_urls:
        companies = companies[companies["ats_url"].isin(restrict_to_urls)]

    if companies.empty:
        print("   -> No SmartRecruiters companies matched.")
        return

    print(f"   -> Targeting {len(companies)} SmartRecruiters Companies")
    discovery = SmartRecruitersDiscovery(sr_db)
    discovery.run()


def run_lever_scraper(restrict_to_urls=None):
    print("\n==================================")
    print("   STARTING LEVER SCRAPER")
    print("==================================")

    lever_db = LeverDBManager()
    companies = lever_db.fetch_companies(ats_name="lever")

    if restrict_to_urls:
        companies = companies[companies["ats_url"].isin(restrict_to_urls)]

    if companies.empty:
        print("   -> No Lever companies matched.")
        return

    print(f"   -> Targeting {len(companies)} Lever Companies")
    discovery = LeverDiscovery(lever_db)
    discovery.run()


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("🚀 Running ATS Scrapers directly from ats_website table...")

    companies_df = get_all_ats_websites()

    if companies_df.empty:
        print("❌ No records found in ats_website.")
        exit()

    # Normalize
    companies_df["ats_name_clean"] = companies_df["ats_name"].astype(str).str.lower().str.strip()

    zoho_targets = companies_df[companies_df["ats_name_clean"] == "zoho recruit"]["ats_url"].tolist()
    keka_targets = companies_df[companies_df["ats_name_clean"] == "keka"]["ats_url"].tolist()
    darwin_targets = companies_df[companies_df["ats_name_clean"] == "darwinbox"]["ats_url"].tolist()
    workable_targets = companies_df[companies_df["ats_name_clean"] == "workable"]["ats_url"].tolist()
    freshteam_targets = companies_df[companies_df["ats_name_clean"] == "freshteam"]["ats_url"].tolist()
    smart_targets = companies_df[companies_df["ats_name_clean"] == "smartrecruiters"]["ats_url"].tolist()
    lever_targets = companies_df[companies_df["ats_name_clean"] == "lever"]["ats_url"].tolist()

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

    if smart_targets:
        run_smartrecruiters_scraper(restrict_to_urls=smart_targets)

    if lever_targets:
        run_lever_scraper(restrict_to_urls=lever_targets)

    print("\nALL ATS WEBSITES PROCESSED.")

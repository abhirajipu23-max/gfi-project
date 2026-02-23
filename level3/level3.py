from __future__ import annotations

import os
import re
import logging
from typing import Dict, Optional, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright, Page

from dotenv import load_dotenv
from supabase import create_client, Client

OUTPUT_FILE = "cleaned_job_urls13.csv"

MAX_PAGES = int(os.getenv("MAX_PAGES", "20"))
NAV_TIMEOUT = int(os.getenv("NAV_TIMEOUT", "15000"))
DOM_TIMEOUT = int(os.getenv("DOM_TIMEOUT", "3000"))

STATUS_SCANNED = "scanned"
STATUS_NOT_FOUND = "not ats found"

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ATS_CONFIG: Dict[str, Dict[str, object]] = {
    "iCIMS": {
        "domains": ["icims.com"],
        "regex": r"https?://([^.]+)\.icims\.com",
        "template": "https://{}.icims.com/jobs/",
    },
    "SuccessFactors": {
        "domains": ["successfactors.com"],
        "regex": r"successfactors\.com/([^/]+)",
        "template": "https://career5.successfactors.com/career?company={}",
    },
    "Taleo": {
        "domains": ["taleo.net"],
        "regex": r"taleo\.net/([^/]+)",
        "template": "https://{}.taleo.net/careersection/2/jobsearch.ftl",
    },
    "Freshteam": {
        "domains": ["freshteam.com"],
        "regex": r"https?://([^.]+)-team\.freshteam\.com",
        "template": "https://{}-team.freshteam.com/jobs",
    },
    "Recruiterbox": {
        "domains": ["recruiterbox.com"],
        "regex": r"recruiterbox\.com/([^/]+)",
        "template": "https://{}.recruiterbox.com/jobs",
    },
    "JazzHR": {
        "domains": ["applytojob.com"],
        "regex": r"https?://([^.]+)\.applytojob\.com",
        "template": "https://{}.applytojob.com/apply",
    },
    "Breezy HR": {
        "domains": ["breezy.hr"],
        "regex": r"breezy\.hr/([^/]+)",
        "template": "https://{}.breezy.hr/",
    },
    "FactoHR": {
        "domains": ["factohr.com"],
        "regex": r"factohr\.com/([^/]+)",
        "template": "https://{}.factohr.com/careers",
    },
    "Ashby": {
        "domains": ["ashbyhq.com", "jobs.ashbyhq.com"],
        "regex": r"ashbyhq\.com/([^/]+)",
        "template": "https://jobs.ashbyhq.com/{}",
    },
    "RecruitCRM": {
        "domains": ["recruitcrm.io"],
        "regex": r"recruitcrm\.io/([^/]+)",
        "template": "https://{}.recruitcrm.io/jobs",
    },
    "Rippling": {
        "domains": ["rippling.com"],
        "regex": r"rippling\.com/([^/]+)",
        "template": "https://www.rippling.com/careers/{}",
    },
    "PyjamaHR": {
        "domains": ["pyjamahr.com"],
        "regex": r"https?://([^.]+)\.pyjamahr\.com",
        "template": "https://{}.pyjamahr.com/careers",
    },
    "ZipRecruiter": {
        "domains": ["ziprecruiter.com"],
        "regex": r"ziprecruiter\.com/([^/]+)",
        "template": "https://www.ziprecruiter.com/candidate/search?company={}",
    },
    "Lever": {
        "domains": ["jobs.lever.co", "lever.co"],
        "regex": r"lever\.co/([^/]+)",
        "template": "https://jobs.lever.co/{}",
    },
    "Zoho Recruit": {
        "domains": ["zohorecruit.com", "zohorecruit.in", "zohorecruit.eu"],
        "regex": r"https?://([^.]+)\.zohorecruit\.(com|in|eu)",
        "template": "https://{}.zohorecruit.{}/careers",
    },
    "Keka": {
        "domains": ["keka.com"],
        "regex": r"https?://([^.]+)\.keka\.com",
        "template": "https://{}.keka.com/careers/",
    },
    "Workable": {
        "domains": ["apply.workable.com"],
        "regex": r"workable\.com/([^/]+)",
        "template": "https://apply.workable.com/{}/",
    },
    "Workday": {
        "domains": ["myworkdayjobs.com"],
        "regex": r"https?://([^.]+)\.myworkdayjobs\.com",
        "template": "https://{}.myworkdayjobs.com/en-US/Careers",
    },
    "Darwinbox": {
        "domains": ["darwinbox.in"],
        "regex": r"https?://([^.]+)\.darwinbox\.in",
        "template": "https://{}.darwinbox.in/ms/candidatev2/main/careers/allJobs",
    },
    "Greenhouse": {
        "domains": ["greenhouse.io", "boards.greenhouse.io"],
        "regex": r"greenhouse\.io/([^/]+)",
        "template": "https://boards.greenhouse.io/{}",
    },
    "Teamtailor": {
        "domains": ["teamtailor.com"],
        "regex": r"https?://([^.]+)\.teamtailor\.com",
        "template": "https://{}.teamtailor.com/jobs",
    },
    "Jobvite": {
        "domains": ["jobvite.com"],
        "regex": r"jobvite\.com/([^/]+)",
        "template": "https://jobs.jobvite.com/{}/jobs",
    },
    "SmartRecruiters": {
        "domains": ["smartrecruiters.com"],
        "regex": r"smartrecruiters\.com/([^/]+)",
        "template": "https://careers.smartrecruiters.com/{}",
    },
    "BambooHR": {
        "domains": ["bamboohr.com"],
        "regex": r"https?://([^.]+)\.bamboohr\.com",
        "template": "https://{}.bamboohr.com/careers",
    },
}

def get_companies_from_supabase() -> pd.DataFrame:
    res = supabase.table("companies").select("id,name,homepage_url,careers_url,status").execute()
    return pd.DataFrame(res.data)

def get_ats_website_from_supabase() -> pd.DataFrame:
    res = supabase.table("ats_website").select("company_id").execute()
    return pd.DataFrame(res.data)

def insert_into_ats_website(company_id: str, ats_name: str, ats_url: str, code_name: str) -> None:
    payload = {
        "company_id": company_id,
        "ats_name": ats_name,
        "ats_url": ats_url,
        "code_name": code_name,
    }

    try:
        supabase.table("ats_website").insert(payload).execute()
        logger.info("Inserted into ats_website: %s", payload)
    except Exception as e:
        logger.error("Failed inserting into ats_website: %s", e)

def update_company_status(company_id: str, status: str) -> None:
    try:
        supabase.table("companies").update({"status": status}).eq("id", company_id).execute()
        logger.info("Updated company status: %s -> %s", company_id, status)
    except Exception as e:
        logger.error("Failed updating company status (%s): %s", company_id, e)

def is_valid_url(value: object) -> bool:
    return isinstance(value, str) and value.startswith("http")

def extract_urls_from_html(html: str) -> list[str]:
    return re.findall(r'https?://[^\s"\'<>]+', html)

def reconstruct_url(text: str, ats: str) -> Optional[str]:
    cfg = ATS_CONFIG[ats]
    match = re.search(cfg["regex"], text, re.IGNORECASE)
    if not match:
        return None

    if ats == "Zoho Recruit":
        return cfg["template"].format(match.group(1), match.group(2))

    return cfg["template"].format(match.group(1))

def match_ats(text: str) -> Tuple[Optional[str], Optional[str]]:
    lower = text.lower()
    for ats, cfg in ATS_CONFIG.items():
        if any(domain in lower for domain in cfg["domains"]):
            clean = reconstruct_url(text, ats)
            if clean:
                return ats, clean
    return None, None

def append_to_csv(row: dict) -> None:
    write_header = not os.path.exists(OUTPUT_FILE)
    pd.DataFrame([row]).to_csv(
        OUTPUT_FILE,
        mode="a",
        index=False,
        header=write_header,
    )

def to_snake_case(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def is_status_already_set(status: object) -> bool:
    if status is None:
        return False
    s = str(status).strip().lower()
    return s != ""

def scrape_page(page: Page, url: str) -> Tuple[str, str, Optional[str]]:
    network_hits: set[str] = set()
    page.on("request", lambda r: network_hits.add(r.url))

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    except Exception:
        return url, "Error", None

    ats, clean = match_ats(page.url)
    if ats:
        return url, ats, clean

    for hit in network_hits:
        ats, clean = match_ats(hit)
        if ats:
            return url, ats, clean

    try:
        page.wait_for_load_state("domcontentloaded", timeout=DOM_TIMEOUT)
        for found in extract_urls_from_html(page.content()):
            ats, clean = match_ats(found)
            if ats:
                return url, ats, clean
    except Exception:
        pass

    return url, "Unknown", None

def main() -> None:
    logger.info("Starting ATS detection job (skip by status + skip homepage already in ats_website)")

    companies_df = get_companies_from_supabase()
    if companies_df.empty:
        logger.error("No companies found in companies table.")
        return

    ats_df = get_ats_website_from_supabase()

    existing_homepages: set[str] = set()

    if not ats_df.empty and "company_id" in ats_df.columns:
        existing_ids = set(str(x) for x in ats_df["company_id"].dropna().tolist())

        existing_homepages = set(
            str(x).strip().lower()
            for x in companies_df[companies_df["id"].astype(str).isin(existing_ids)]["homepage_url"]
            .dropna()
            .tolist()
        )

    logger.info("Existing homepage_urls found via ats_website join: %s", len(existing_homepages))

    companies_df["Company Name"] = companies_df["name"]
    companies_df["domain"] = companies_df["homepage_url"]
    companies_df["job_url"] = companies_df["careers_url"]

    rows = (
        companies_df[companies_df["job_url"].apply(is_valid_url)][
            ["id", "Company Name", "domain", "job_url", "status"]
        ]
        .to_dict("records")
    )

    filtered_rows = []
    skipped_homepage = 0
    skipped_status = 0

    for r in rows:
        if is_status_already_set(r.get("status")):
            skipped_status += 1
            continue

        homepage = str(r.get("domain") or "").strip().lower()
        if homepage and homepage in existing_homepages:
            skipped_homepage += 1
            continue

        filtered_rows.append(r)

    logger.info("Total companies with valid careers_url: %s", len(rows))
    logger.info("Skipped (status already set): %s", skipped_status)
    logger.info("Skipped (homepage already in ats_website): %s", skipped_homepage)
    logger.info("Remaining to check: %s", len(filtered_rows))

    if not filtered_rows:
        logger.info("Nothing to check. Exiting.")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(ignore_https_errors=True)

        context.route(
            "**/*",
            lambda route, request: route.abort()
            if request.resource_type in {"image", "font", "media"}
            else route.continue_(),
        )

        pages = [context.new_page() for _ in range(MAX_PAGES)]

        for index, row in enumerate(filtered_rows, start=1):
            page = pages[index % MAX_PAGES]
            job_url = row["job_url"]
            company_id = str(row["id"])

            company_url, ats, clean = scrape_page(page, job_url)

            if ats not in {"Unknown", "Error"} and clean:
                logger.info("MATCH [%s] %s → %s", index, row["Company Name"], ats)

                code_name = f"{to_snake_case(row['Company Name'])}_{to_snake_case(ats)}"

                insert_into_ats_website(
                    company_id=company_id,
                    ats_name=ats,
                    ats_url=clean,
                    code_name=code_name,
                )

                update_company_status(company_id, STATUS_SCANNED)

                append_to_csv(
                    {
                        "Company Name": row["Company Name"],
                        "Domain": row["domain"],
                        "Company URL": company_url,
                        "ATS_Detected": ats,
                        "Jobs_Page_URL": clean,
                    }
                )
            else:
                logger.info("NO MATCH [%s] %s", index, row["Company Name"])
                update_company_status(company_id, STATUS_NOT_FOUND)

        context.close()
        browser.close()

    logger.info("Finished. CSV written to %s", OUTPUT_FILE)

if __name__ == "__main__":
    main()

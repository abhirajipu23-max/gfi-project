import asyncio
import uuid
import re
import os
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv

from scrapling.fetchers import AsyncFetcher
from supabase import create_client, Client

# =========================
# 🔇 DISABLE SCRAPLING/CURL LOG SPAM
# =========================
logging.getLogger("scrapling").setLevel(logging.CRITICAL)
logging.getLogger("curl_cffi").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

# --- CONFIGURATION & SETUP ---

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Scraper Settings
BASE_PATHS = [
    "/careers", "/jobs", "/join-us", "/join", "/opportunities",
    "/openings", "/vacancies", "/work-with-us",
    "/about/careers", "/company/careers", "/en/careers"
]

VARIANTS = [
    ("career", "careers"),
    ("job", "jobs"),
    ("opening", "openings"),
    ("vacancy", "vacancies")
]

ATS_HINTS = [
    "greenhouse.io", "boards.greenhouse.io", "lever.co", "workable.com",
    "ashbyhq.com", "myworkdayjobs.com", "icims.com", "recruitee.com",
    "smartrecruiters.com", "teamtailor.com", "bamboohr.com", "naukri.com",
    "darwinbox.in", "keka.com", "wellfound.com", "zohorecruit.com",
    "personio.com", "jobs.jobvite.com", "applytojob.com"
]

CONTENT_HINTS = [
    "openings", "positions", "roles", "departments", "location",
    "apply", "search jobs", "job listings", "join our team"
]

# --- HELPER FUNCTIONS ---

def normalize_base(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
    return url.rstrip("/")

def abs_url(base: str, href: str) -> str:
    return urljoin(base + "/", href)

def google_favicon_v2(url, size=64):
    norm_url = normalize_base(url)
    if not norm_url:
        return ""
    return (
        "https://t1.gstatic.com/faviconV2"
        f"?client=SOCIAL&type=FAVICON"
        f"&fallback_opts=TYPE,SIZE,URL"
        f"&url={norm_url}"
        f"&size={size}"
    )

def looks_like_job_url(href: str, text: str = "") -> bool:
    h = (href or "").lower()
    t = (text or "").lower()
    keybits = [
        "career", "job", "join-us", "joinus",
        "work-with-us", "opportunities", "openings", "vacancies"
    ]
    return any(k in h for k in keybits) or any(k in t for k in keybits)

def score_page_text(txt: str) -> int:
    score = 0
    low = (txt or "").lower()
    for w in CONTENT_HINTS:
        if w in low:
            score += 1
    if "jobposting" in low:
        score += 2
    return score

def contains_ats(url: str) -> bool:
    u = (url or "").lower()
    return any(host in u for host in ATS_HINTS)

async def fetch_static(url: str, timeout_ms: int = 20000):
    try:
        return await AsyncFetcher.get(url, timeout=timeout_ms, follow_redirects=True)
    except Exception:
        return None

# --- VERIFY HOMEPAGE URL (FOLLOW REDIRECTS) ---

async def verify_homepage_url(domain: str) -> str:
    base = normalize_base(domain)
    if not base:
        return ""

    page = await fetch_static(base)
    if not page:
        return base

    final_url = getattr(page, "url", None) or base
    return normalize_base(final_url)

# --- SCRAPING LOGIC ---

def generate_path_guesses():
    guesses = set(BASE_PATHS)
    for singular, plural in VARIANTS:
        guesses.add("/" + singular)
        guesses.add("/" + plural)
    expanded = set()
    for g in guesses:
        expanded.add(g)
        expanded.add(g.capitalize())
    return sorted(expanded)

def extract_candidate_links(page, base_url: str):
    links = []
    for a in page.css("a"):
        href = a.attrib.get("href")
        text = a.text
        if not href:
            continue
        links.append((abs_url(base_url, href), text))
    return links

def evaluate_as_jobs_page(page, url: str):
    if contains_ats(url):
        return 0.95, "ats-url"

    txt = page.get_all_text(ignore_tags=("script", "style"))
    content_score = score_page_text(txt)
    links = extract_candidate_links(page, url)

    jobish_links = [u for (u, t) in links if looks_like_job_url(u, t)]
    ats_links = [u for (u, _) in links if contains_ats(u)]

    confidence = 0.0
    reason = "low"

    if ats_links:
        confidence = 0.9
        reason = "links-to-ats"
    if len(jobish_links) >= 5:
        confidence = max(confidence, 0.8)
        reason = "many-jobish-links"
    if content_score >= 3 and len(txt) > 500:
        confidence = max(confidence, 0.75)
        reason = "content-hints"
    if looks_like_job_url(url):
        confidence = max(confidence, 0.7)
        reason = "url-hint"

    return confidence, reason

async def try_direct_paths(base: str):
    for path in generate_path_guesses():
        url = base + path
        page = await fetch_static(url)
        if not page or page.status != 200:
            continue
        conf, why = evaluate_as_jobs_page(page, url)
        if conf >= 0.7:
            return {"job_url": url, "confidence": conf, "method": f"direct:{why}"}
    return None

async def crawl_with_depth(base: str, max_depth: int = 2):
    visited, queue = set(), [(base, 0)]

    while queue:
        url, depth = queue.pop(0)
        if url in visited or depth > max_depth:
            continue

        visited.add(url)
        page = await fetch_static(url)
        if not page or page.status != 200:
            continue

        conf, why = evaluate_as_jobs_page(page, url)
        if conf >= 0.7:
            return {"job_url": url, "confidence": conf, "method": f"depth-{depth}:{why}"}

        if depth < max_depth:
            for (u, t) in extract_candidate_links(page, url):
                if looks_like_job_url(u, t) and u not in visited:
                    queue.append((u, depth + 1))
    return None

async def find_job_page_for_domain(domain: str):
    base = normalize_base(domain)
    if not base:
        return {"domain": domain, "job_url": ""}

    hit = await try_direct_paths(base)
    if hit:
        return {"domain": domain, **hit}

    hit = await crawl_with_depth(base, max_depth=2)
    if hit:
        return {"domain": domain, **hit}

    careers_sub = re.sub(r"^(https?://)(www\.)?", r"\1", base)
    careers_sub = careers_sub.replace("://", "://careers.")
    hit = await try_direct_paths(careers_sub)
    if hit:
        return {"domain": domain, **hit}

    return {"domain": domain, "job_url": ""}

# --- SUPABASE DB HELPERS ---

def update_input_list_status(list_id: int, status: str):
    try:
        supabase.table("input_list").update({"status": status}).eq("id", list_id).execute()
        print(f"📝 STATUS UPDATED: id={list_id} -> {status}")
    except Exception as e:
        print(f"❌ Failed updating status for id={list_id}: {e}")

def check_homepage_exists_in_companies(homepage_url: str) -> bool:
    try:
        homepage_url = normalize_base(homepage_url)
        if not homepage_url:
            return False

        response = (
            supabase.table("companies")
            .select("id")
            .eq("homepage_url", homepage_url)
            .limit(1)
            .execute()
        )
        return len(response.data) > 0
    except Exception as e:
        print(f"Error checking homepage_url in companies table for {homepage_url}: {e}")
        return False

def check_name_exists_in_companies(name: str) -> bool:
    try:
        name = (name or "").strip()
        if not name:
            return False

        response = (
            supabase.table("companies")
            .select("id")
            .eq("name", name)
            .limit(1)
            .execute()
        )
        return len(response.data) > 0
    except Exception as e:
        print(f"❌ Error checking name in companies table for {name}: {e}")
        return False

def insert_into_companies(data: dict):
    try:
        supabase.table("companies").insert(data).execute()
        print(f"✅ INSERTED: {data['name']}")
        return True
    except Exception as e:
        print(f"❌ DB ERROR for {data['name']}: {e}")
        return False

def get_input_list_from_supabase():
    try:
        # 🔥 Only fetch rows where status is NULL (not processed yet)
        res = (
            supabase.table("input_list")
            .select("id,name,domain,description,status")
            .is_("status", "null")
            .execute()
        )
        return res.data or []
    except Exception as e:
        print(f"❌ Failed fetching input_list: {e}")
        return []

# --- MAIN WORKER ---

async def process_supabase_companies(concurrency: int = 10):
    rows = get_input_list_from_supabase()

    if not rows:
        print("✅ No pending companies found (all have status).")
        return

    sem = asyncio.Semaphore(concurrency)

    async def worker(row):
        async with sem:
            list_id = row.get("id")
            company_name = (row.get("name") or "").strip()
            domain = (row.get("domain") or "").strip()
            list_description = (row.get("description") or "").strip()

            if not list_id or not company_name or not domain:
                return

            # ✅ Verify homepage_url by following redirects
            verified_homepage_url = await verify_homepage_url(domain)

            # ✅ Skip if already exists by homepage_url OR name
            if (
                check_homepage_exists_in_companies(verified_homepage_url)
                or check_name_exists_in_companies(company_name)
            ):
                update_input_list_status(list_id, "Already Exist")
                return

            # Scrape career page
            try:
                res = await find_job_page_for_domain(domain)
            except Exception:
                res = {"job_url": ""}

            # Insert + update status
            if res.get("job_url"):
                now_ts = datetime.now(timezone.utc).isoformat()

                db_row = {
                    "id": str(uuid.uuid4()),
                    "created_at": now_ts,
                    "updated_at": now_ts,
                    "name": company_name,
                    "homepage_url": verified_homepage_url,
                    "careers_url": res["job_url"],
                    "logo_file_name": google_favicon_v2(verified_homepage_url),
                    "description": list_description,
                    "list_id": list_id,
                }

                inserted = insert_into_companies(db_row)

                if inserted:
                    update_input_list_status(list_id, "Scanned")
                else:
                    update_input_list_status(list_id, "Insert Failed")
            else:
                update_input_list_status(list_id, "No Career Page")

    await asyncio.gather(*(worker(row) for row in rows))

if __name__ == "__main__":
    try:
        print("Starting Supabase Pipeline (Input: input_list)...")
        asyncio.run(process_supabase_companies(concurrency=20))
        print("Pipeline Complete.")
    except KeyboardInterrupt:
        print("\nStopped by user.")

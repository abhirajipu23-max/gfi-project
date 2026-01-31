import os
import time
import re
import random
import string
import pandas as pd
import torch
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from urllib.parse import urlparse, urlunparse
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
from groq import Groq, RateLimitError

load_dotenv()

TARGET_TABLE = "jobs_uploadable_duplicate"
SOURCE_TABLE = "jobs_duplicate"

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ufnaxahhlblwpdomlybs.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_1d4J1Ll81KwhYPOS40U8mQ_qtCccNsa")

GROQ_API_KEYS = [
    os.getenv("API_KEY1"),
    os.getenv("API_KEY2"),
    os.getenv("API_KEY3"),
    os.getenv("API_KEY4"),
    os.getenv("API_KEY5"),
    os.getenv("API_KEY6"),
]

GROQ_API_KEYS = [k for k in GROQ_API_KEYS if k]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

current_key_index = 0
current_model_index = 0

BATCH_SIZE = 50
Rate_Limit_Sleep = 1

UTM_QUERY_PARAM = "ref=growthforimpact.co&utm_source=growthforimpact.co&utm_medium=referral"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Loading Department Embedding Models...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

CANONICAL_DEPARTMENTS = ["Accounting & Taxation","Administration","Administration & Staff","Advertising & Creative","After Sales Service & Repair","Airline Services",
                         "Animation / Effects","Architecture & Interior Design","Artists","Assessment / Advisory","Audit & Control","Aviation Engineering","Back Office",
                         "Banking Operations","BD / Pre Sales","Beauty & Personal Care","BFSI, Investments & Trading","Business","Business Intelligence & Analytics","Business Process Quality",
                         "Category Management & Operations","Climate Change","Community Health & Safety","Compensation & Benefits","Conservation","Construction / Manufacturing",
                         "Construction Engineering","Content Management- Print / Online / Electronic","Content, Editorial & Journalism","Corporate Affairs","Corporate Communication",
                         "Corporate Training","CSR & Sustainability","Customer Success","Customer Success, Service & Operations","Data Science & Analytics","Data Science & Machine Learning",
                         "DBA / Data warehousing","DevOps","Digital Marketing","Direction","Downstream","Ecology","eCommerce Operations","Editing","Educators & Teachers","EHS",
                         "Employee Relations","Energy Efficiency","Engineering","Engineering & Manufacturing","Enterprise & B2B Sales","Environment Health and Safety","Environmental","F&B Service",
                         "Facility Management","Farming","Fashion & Accessories","Finance","Finance & Accounting",
                        "Food, Beverage & Hospitality","Front Office & Guest Services","General Insurance","Geologist","Hardware","Hardware and Networks","Health & Fitness","Healthcare & Life Sciences","Housekeeping & Laundry",
                        "HR Business Advisory","HR Operations","Human Resources","HVAC","Imaging & Diagnostics","Import & Export","Investment Banking, Private Equity & VC","IT & Information Security","IT Consulting","IT Infrastructure Services",
                        "IT Network","IT Security","IT Support","Journalism","Kitchen / F&B Production","Language Teacher","LEED","Legal & Regulatory","Legal Operations","Lending","Life Skills / ECA Teacher","Management","Management Consulting",
                        "Market Research & Insights","Marketing","Marketing and Communication","Media Production & Entertainment","Merchandising & Planning","Merchandising, Retail & eCommerce","Mining","Naturalist","Nursing","Occupational Health & Safety",
                        "Operations","Operations / Strategy","Operations Support","Operations, Maintenance & Support","Other Consulting","Other Hospital Staff","Payroll & Transactions","Pharmaceutical & Biotechnology","Port & Maritime Operations",
                        "Power Generation","Power Supply and Distribution","Preschool & Primary Education","Procurement & Purchase","Procurement & Supply Chain","Product Management","Product Management - Technology","Production","Production & Manufacturing",
                        "Program / Project Management","Quality Assurance","Quality Assurance and Testing","Recruitment & Talent Acquisition","Recruitment Marketing & Branding","Recycling","Renewable Energy","Research & Development","Retail & B2C Sales",
                        "Retail Store Operations","Risk Management & Compliance","Sales Support & Operations","SCM & Logistics","Security / Fraud","Security Officer","Security Services","Service Delivery","Shipping & Maritime","Shipping Engineering & Technical",
                        "Smart Grid","Social & Public Service","Software Development","Solar","Sound / Light / Technical Support","Sports Staff and Management","Sports, Fitness & Personal Care","Stores & Material Management","Strategic & Top Management","Strategic Management",
                        "Subject / Specialization Teacher","Surveying","Sustainability","Sustainable Ag.","Teaching & Training","Technology / IT","Telecom","Top Management","Tourism Services","Trading, Asset & Wealth Management",
                        "Treasury & Forex","UI / UX","Water","Weatherization","Wildlife","Wind Power"]
CANONICAL_EMBEDDINGS = embedder.encode(CANONICAL_DEPARTMENTS, convert_to_tensor=True)

DEPT_RULE_MAP = {
    r"\b(fpga|embedded|vlsi|hardware|circuit|chip|asic|board)\b": "Hardware",
    r"\b(java|python|developer|coder|software|frontend|backend|fullstack)\b": "Software Development",
    r"\b(engineer|engineering)\b": "Engineering",
    r"\b(sales|bd|business development|pre[- ]?sales|marketing|specialist)\b": "Sales & Marketing",
    r"\b(hr|human resources|talent|recruitment)\b": "Human Resources",
    r"\b(finance|accounting|audit|tax|treasury)\b": "Finance",
    r"\b(solar|renewable)\b": "Solar",
    r"\b(wind)\b": "Wind Power",
    r"\b(healthcare|nurse|doctor|clinical)\b": "Healthcare",
    r"\b(pharma|biotech)\b": "Pharmaceutical & Biotechnology",
    r"\b(ops|operations|supply chain|logistics|procurement)\b": "Operations",
    r"\b(legal|law|compliance)\b": "Legal",
    r"\b(admin|office)\b": "Administration",
    r"\b(content|editor|journalist|writer)\b": "Content, Editorial & Journalism",
    r"\b(customer|support|success|service)\b": "Customer Success"
}


def supabase_execute_with_retry(query_builder, retries=5):
    for attempt in range(retries):
        try:
            return query_builder.execute()
        except Exception as e:
            wait_time = (attempt + 1) * 2
            print(f"Network Error (Attempt {attempt+1}): {e}")
            time.sleep(wait_time)
    print("Critical: Supabase request failed.")
    return None

def generate_slug(job_title, company_name):
    def slugify(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        return text.strip("-")

    if pd.isna(job_title):
        return None

    comp = company_name if pd.notna(company_name) else "unknown"
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{slugify(job_title)}-at-{slugify(comp)}-{suffix}"

PIXEL_EXCEPTION_DOMAIN = "pixxel.darwinbox.in"

def clean_utm_url(url):
    if pd.isna(url) or not str(url).startswith("http"):
        return url

    try:
        parsed = urlparse(url)

        if parsed.netloc == PIXEL_EXCEPTION_DOMAIN:
            return urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                "", "", ""
            ))

        current_query = parsed.query

        if current_query:
            new_query = f"{current_query}&{UTM_QUERY_PARAM}"
        else:
            new_query = UTM_QUERY_PARAM

        cleaned_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
        return cleaned_url

    except Exception:
        return url



def cleanup_old_jobs():
    # print("Cleaning up jobs older than 6 months from target table...")

    cutoff_date = (pd.Timestamp.utcnow() - pd.DateOffset(months=6)).strftime("%Y-%m-%d")

    try:
        res = (
            supabase.table(TARGET_TABLE)
            .delete()
            .lt("published_at", cutoff_date)
            .execute()
        )

        deleted_count = len(res.data) if res and res.data else 0
        print(f"🗑️ Deleted {deleted_count} old jobs from {TARGET_TABLE}")

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")





def calculate_experience_string(min_exp, max_exp):
    try:
        mn = int(min_exp) if pd.notna(min_exp) else 0
        mx = int(max_exp) if pd.notna(max_exp) else 0

        if mn < 1 and mx < 1:
            return None 

        if mx > mn:
            return f"{mn}-{mx} years"
        else:
            return f"{mn}+ years"
    except:
        return None

def map_department(text):
    if not text or not isinstance(text, str):
        return "Other"

    norm_text = text.lower()
    for pattern, dept in DEPT_RULE_MAP.items():
        if re.search(pattern, norm_text):
            return dept

    match, score, _ = process.extractOne(text, CANONICAL_DEPARTMENTS, scorer=fuzz.WRatio)
    if score >= 85:
        return match

    test_emb = embedder.encode(text, convert_to_tensor=True)
    cos_scores = util.cos_sim(test_emb, CANONICAL_EMBEDDINGS)[0]
    best_idx = torch.argmax(cos_scores).item()

    if cos_scores[best_idx].item() >= 0.5:
        return CANONICAL_DEPARTMENTS[best_idx]

    return "Other"

def call_groq_with_retry(prompt):
    global current_key_index, current_model_index
    total_keys = len(GROQ_API_KEYS)
    total_models = len(GROQ_MODELS)
    max_attempts = total_keys * total_models
    attempts = 0

    while attempts < max_attempts:
        active_key = GROQ_API_KEYS[current_key_index]
        active_model = GROQ_MODELS[current_model_index]
        client = Groq(api_key=active_key)

        try:
            response = client.chat.completions.create(
                model=active_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            print(f"Rate Limit: Key ...{active_key[-6:]} | Model {active_model}")
            current_key_index = (current_key_index + 1) % total_keys
            if current_key_index == 0:
                print("Switching Model to fallback...")
                current_model_index = (current_model_index + 1) % total_models
                if current_model_index == 0:
                    print("ALL KEYS EXHAUSTED. Pause for 2 mins...")
                    time.sleep(120)
            attempts += 1

        except Exception as e:
            print(f"Groq Error: {e}")
            return None
    return None


def normalize_job_type(job_type):
    jt = job_type.lower() if isinstance(job_type, str) and job_type.strip() else ""

    if "intern" in jt:
        return "Internship"

    if any(w in jt for w in [
        "freelance",
        "contract",
        "part",
        "fraction",
        "training",
        "part-time",
    ]):
        return "Contractor"

    return "Full Time"





def clean_description_groq(text):
    if not text or len(str(text)) < 50:
        return ""
    
    prompt = f"""
Clean the following job description into structured sections.

CRITICAL RULES (must follow exactly):
- Return ONLY plain text.
- Do NOT use markdown, symbols, explanations, or extra text.
- Do NOT invent, infer, summarize, or guess information.
- Extract only what is explicitly stated in the input.

SECTION RULES:
- ALWAYS include ROLE and REQUIREMENTS.
- ONLY include a BENEFITS section if the input explicitly lists real benefits.
- If no benefits are explicitly listed, OMIT the BENEFITS section entirely.
- If benefits are omitted, the word "BENEFITS" must NOT appear anywhere in the output.
- NEVER write placeholders or explanations such as:
  "No benefits mentioned"
  "None"
  "N/A"
  "Not provided"

FORMAT (follow exactly; omit BENEFITS section if not applicable):

ROLE:
- point
- point

REQUIREMENTS:
- point
- point

BENEFITS:
- point
- point

INPUT:
<<<{str(text)[:3000]}>>>
"""


    return call_groq_with_retry(prompt)

def clean_val(val):
    if pd.isna(val):
        return None
    return val

def _first_if_list(x):
    if isinstance(x, list):
        return x[0] if x else None
    return x

def extract_company_data(row):
    defaults = {
        "name": "Unknown",
        "homepage_url": None,
        "logo_file_name": None,
        "description": None
    }

    try:
        scrapes = _first_if_list(row.get("scrapes_duplicate"))
        if not scrapes:
            return defaults

        ats = _first_if_list(scrapes.get("ats_website"))
        if not ats:
            return defaults

        comp = _first_if_list(ats.get("companies"))
        if not comp:
            return defaults

        return {
            "name": comp.get("name", "Unknown"),
            "homepage_url": comp.get("homepage_url"),
            "logo_file_name": comp.get("logo_file_name"),
            "description": comp.get("description"),
        }
    except Exception:
        return defaults


def run_pipeline():
    cleanup_old_jobs()


def run_pipeline():
    print(f"Starting Optimized Pipeline -> Target: {TARGET_TABLE}")
    print("   - Filter: Job ID Exists in Target")
    print("   - Filter: Description words <= 20")
    print("   - Filter: Published Date > 6 months ago")

    offset = 0
    total_processed = 0

    cutoff_date = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=6)

    while True:
        print(f"\nFetching batch: Rows {offset} to {offset + BATCH_SIZE}...")

        query = "*, scrapes_duplicate(ats_website(companies(name, homepage_url, logo_file_name, description)))"

        query_builder = (
            supabase.table(SOURCE_TABLE)
            .select(query)
            .eq("is_active", True)
            .range(offset, offset + BATCH_SIZE - 1)
        )

        res = supabase_execute_with_retry(query_builder)

        if not res or not res.data:
            print("Source exhausted. Pipeline Finished.")
            break

        df = pd.DataFrame(res.data)

        batch_ids = df["id"].tolist()

        check_query = (
            supabase.table(TARGET_TABLE)
            .select("job_id")
            .in_("job_id", batch_ids)
        )

        check_res = supabase_execute_with_retry(check_query)

        existing_ids = set()
        if check_res and check_res.data:
            existing_ids = {item["job_id"] for item in check_res.data}

        mask_new_id = ~df["id"].isin(existing_ids)

        df["word_count"] = df["original_description"].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        mask_long_desc = df["word_count"] > 20

        df["published_date_dt"] = pd.to_datetime(
            df["published_date"], errors="coerce", utc=True
        )
        mask_fresh_date = (df["published_date_dt"].notna()) & (df["published_date_dt"] >= cutoff_date)

        final_mask = mask_new_id & mask_long_desc & mask_fresh_date
        df_clean = df[final_mask].copy()

        skipped_exists = (~mask_new_id).sum()
        skipped_short = (mask_new_id & ~mask_long_desc).sum()
        skipped_old = (mask_new_id & mask_long_desc & ~mask_fresh_date).sum()

        if skipped_exists > 0:
            print(f"Skipped {skipped_exists} jobs (Already Exist)")
        if skipped_short > 0:
            print(f"Skipped {skipped_short} jobs (Desc <= 20 words)")
        if skipped_old > 0:
            print(f"Skipped {skipped_old} jobs (Older than 6 months)")

        if df_clean.empty:
            print("Batch fully filtered out. Next...")
            offset += BATCH_SIZE
            continue

        print(f"Processing {len(df_clean)} valid jobs...")
        processed_rows = []

        for _, row in df_clean.iterrows():
            job_id = row["id"]

            raw_desc = row.get("original_description", "")
            generated_description = clean_description_groq(raw_desc)

            if generated_description is None:
                print("Stopping batch processing due to API Exhaustion.")
                break

            source_dept_text = row.get("department") or row.get("title")
            mapped_department = map_department(source_dept_text)

            comp_data = extract_company_data(row)
            company_name = comp_data["name"]

            job_url_cleaned = clean_utm_url(row.get("job_url"))
            company_website_cleaned = clean_utm_url(comp_data["homepage_url"])

            internal_slug = generate_slug(row.get("title"), company_name)
            
            published_at_str = None
            if pd.notna(row.get("published_date")):
                 try:
                     published_at_str = pd.to_datetime(row["published_date"]).strftime("%Y-%m-%d")
                 except:
                     published_at_str = None

            min_e = row.get("min_exp")
            max_e = row.get("max_exp")
            exp_range = calculate_experience_string(min_e, max_e)

            final_obj = {
                "job_id": int(clean_val(row["id"])),
                
                "company_name": company_name,
                "company_website": company_website_cleaned,

                "job_url": job_url_cleaned,
                "title": clean_val(row.get("title")),
                "published_at": published_at_str,
                
                "min_ex": int(min_e) if pd.notna(min_e) else None,
                "max_ex": int(max_e) if pd.notna(max_e) else None,
                "experience_range": exp_range,

                "location": clean_val(row.get("location")),
                "job_type": normalize_job_type(row.get("job_type")),

                "logo_file": clean_val(comp_data["logo_file_name"]),
                "internal_slug": internal_slug,
                "generated_description": generated_description,
                "company_description": clean_val(comp_data["description"]),

                "department": mapped_department,
                "industry": clean_val(row.get("industry")),
            }

            processed_rows.append(final_obj)

        if processed_rows:
            print(f"Upserting {len(processed_rows)} rows to {TARGET_TABLE}...")
            upsert_query = supabase.table(TARGET_TABLE).upsert(processed_rows)
            upsert_res = supabase_execute_with_retry(upsert_query)
            if upsert_res:
                total_processed += len(processed_rows)
                print(f"Success. Total: {total_processed}")

        if len(processed_rows) < len(df_clean):
            print("Script stopping gracefully due to API limits.")
            break

        offset += BATCH_SIZE
        time.sleep(Rate_Limit_Sleep)

if __name__ == "__main__":
    run_pipeline()
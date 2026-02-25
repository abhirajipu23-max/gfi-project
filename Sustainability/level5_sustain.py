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

# ==================================================
# CONFIGURATION
# ==================================================

TARGET_TABLE = "jobs_uploadable_wp"
SOURCE_TABLE = "jobs_sustain"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

GROQ_API_KEYS = [
    os.getenv("API_KEY1"),
    os.getenv("API_KEY2"),
    os.getenv("API_KEY3"),
    os.getenv("API_KEY4"),
    os.getenv("API_KEY5"),
    os.getenv("API_KEY6"),
]

GROQ_API_KEYS = [k for k in GROQ_API_KEYS if k]

for i, key in enumerate(GROQ_API_KEYS, start=1):
    print(f"API_KEY{i}: {key}")

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
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==================================================
# CATEGORIES & MAPPING
# ==================================================

KEYWORD_RULES = [
    (r"\b(ehs|hse|she|environmental\s+health|occupational\s+health|safety\s+(?:officer|engineer|manager)|health\s+and\s+safety)\b", "EHS / HSE (Environmental Health & Safety)"),
    (r"\b(reporting|gri|sasb|tcfd|csrd|esg\s+reporting|sustainability\s+reporting|audit|lca|life\s+cycle\s+assessment|esg\s+compliance)\b", "ESG Compliance & Sustainability Reporting"),
    (r"\b(climate\s+data|esg\s+data|esg\s+(?:analyst|analytics)|sustainability\s+(?:analyst|data)|climate\s+analytics|gis|geospatial)\b", "Climate Data & ESG Analytics"),
    (r"\b(solar|pv|photovoltaic|rooftop|distributed\s+energy|microgrid)\b", "Solar & Distributed Energy"),
    (r"\b(renewable|clean\s*tech|wind|geothermal|biomass|bess|battery\s+storage|ev\s+|electric\s+vehicle)\b", "Renewable Energy & Clean Tech"),
    (r"\b(climate\s+finance|impact\s+investing|green\s+bonds|esg\s+investing|sustainable\s+finance|investment\s+analyst)\b", "Climate Finance & Impact Investing"),
    (r"\b(supply\s+chain|procurement|logistics|scope\s+3|sustainable\s+sourcing)\b", "Sustainable Supply Chain"),
    (r"\b(green\s+building|leed|breeam|sustainable\s+infrastructure|civil\s+engineer|mep|energy\s+efficiency|hvac)\b", "Green Engineering & Sustainable Infrastructure"),
    (r"\b(water|wastewater|hydrology|hydrogeology|desalination|effluent)\b", "Water & Wastewater Management"),
    (r"\b(ecology|ecological|biodiversity|conservation|wildlife|forestry|marine\s+biology|zoology)\b", "Ecology & Biodiversity"),
    (r"\b(carbon|ghg|greenhouse\s+gas|net\s*zero|decarbonization|emissions|climate\s+action)\b", "Climate & Carbon Management"),
    (r"\b(education|skilling|training|capacity\s+building|climate\s+literacy|curriculum)\b", "Climate Education & Skilling"),
    (r"\b(waste\s+management|circular\s+economy|pollution|remediation|environmental\s+(?:scientist|consultant|manager|planner))\b", "Environmental Management"),
    (r"\b(sustainability|esg|environmental\s+social|corporate\s+responsibility|csr)\b", "Sustainability & ESG"),
]

CATEGORIES = [
    "Sustainability & ESG",
    "Climate & Carbon Management",
    "Renewable Energy & Clean Tech",
    "Environmental Management",
    "EHS / HSE (Environmental Health & Safety)",
    "Ecology & Biodiversity",
    "Water & Wastewater Management",
    "Climate Data & ESG Analytics",
    "ESG Compliance & Sustainability Reporting",
    "Green Engineering & Sustainable Infrastructure",
    "Solar & Distributed Energy",
    "Climate Finance & Impact Investing",
    "Sustainable Supply Chain",
    "Climate Education & Skilling"
]

print("Encoding categories...")
CATEGORY_EMBEDDINGS = embedder.encode(CATEGORIES, convert_to_tensor=True)

def map_department(text):
    if pd.isna(text) or not str(text).strip():
        return "Sustainability & ESG"

    text_clean = str(text).strip().lower()

    for pattern, category in KEYWORD_RULES:
        if re.search(pattern, text_clean, re.IGNORECASE):
            return category

    emb = embedder.encode(text_clean, convert_to_tensor=True)
    scores = util.cos_sim(emb, CATEGORY_EMBEDDINGS)[0]
    idx = torch.argmax(scores).item()

    if scores[idx].item() >= 0.30:
        return CATEGORIES[idx]

    return "Sustainability & ESG"


# ==================================================
# LOCATION STANDARDIZATION & FILTERING
# ==================================================

LOCATION_CACHE = {}
INDIA_LOCATION_CACHE = {}

def clean_location(text: str) -> str:
    if not text or pd.isna(text) or str(text).lower() == 'nan':
        return ""
    t = str(text).lower()
    t = re.sub(r"\.\.\+\s*\d+", "", t)              
    t = re.sub(r"\(.*?\)", "", t)                   
    t = re.sub(r"\d+\/\d+|\d+\s*mw.*", "", t)       
    t = re.sub(r"sector[-\s]*\d+[a-z]*", "", t)     
    t = re.sub(r"\s+", " ", t)
    return t.strip()

LOCATION_PROMPT_TEMPLATE = """
You are a location classifier. Determine if the given location should be kept for an India-focused job board.

RETURN TRUE FOR:
- Any location in India (e.g., Mumbai, Bangalore, Delhi, etc.)
- "Remote" or any variation (remote, work from home, wfh, hybrid)
- Locations that include India (e.g., "Mumbai, India", "Bangalore, Karnataka, India")

RETURN FALSE FOR:
- "Indianapolis" (this is in USA, NOT India)
- "Indiana" (this is a US state, NOT India)
- Any location outside India (USA, UK, Europe, Australia, Singapore, etc.)
- Locations that mention other countries

IMPORTANT NOTES:
- "Remote" should ALWAYS return true - we keep all remote jobs regardless of location
- Do not confuse words containing "ind" with India (like Indiana, Indianapolis)

Return ONLY "true" or "false" - no other text or punctuation.

Location: {location}
Output:"""

def post_process_location(llm_output: str) -> str:
    if not llm_output or ";" in llm_output:
        return llm_output

    parts = [p.strip() for p in llm_output.split(',')]
    
    if len(parts) > 3:
        country = parts[-1]
        cities = []
        for i in range(0, len(parts)-1):
            cities.append(parts[i])
        
        unique_cities = list(dict.fromkeys(cities))
        if country in unique_cities: unique_cities.remove(country)
        
        return f"{', '.join(unique_cities[:2])}, {country}"

    return llm_output

def standardize_single_location(raw_location: str) -> str:
    if not raw_location:
        return ""

    cleaned_loc = clean_location(raw_location)
    if not cleaned_loc:
        return ""
    
    lower = cleaned_loc.lower()
    if "remote" in lower: return "Remote"
    if "head office" in lower: return "Head Office"
    if lower in LOCATION_CACHE: return LOCATION_CACHE[lower]

    try:
        active_key = GROQ_API_KEYS[current_key_index] if GROQ_API_KEYS else None
        if not active_key:
            return cleaned_loc.title()
        
        location_client = Groq(api_key=active_key)
        response = location_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": LOCATION_PROMPT_TEMPLATE.format(location=cleaned_loc)}],
            temperature=0,
            max_tokens=50,
        )

        result = response.choices[0].message.content.strip().split('\n')[0]
        result = result.split("Example")[0].rstrip('.: ')
        result = post_process_location(result)
        
        LOCATION_CACHE[lower] = result
        return result

    except RateLimitError:
        print(f"Rate Limit hit during location standardization for '{cleaned_loc}'")
        return cleaned_loc.title()
    except Exception as e:
        print(f"Error standardizing location '{cleaned_loc}': {e}")
        return cleaned_loc.title()

def standardize_location(raw_location: str) -> str:
    if pd.isna(raw_location) or not str(raw_location).strip():
        return ""
    return standardize_single_location(str(raw_location))

def is_india_location(location_str: str) -> bool:
    global current_key_index
    
    if not location_str or pd.isna(location_str):
        return False
    
    location_lower = str(location_str).lower().strip()
    
    if "india" in location_lower:
        return True
        
    if location_lower in INDIA_LOCATION_CACHE:
        return INDIA_LOCATION_CACHE[location_lower]
        
    prompt = f"Is the location '{location_str}' situated in India? Reply ONLY with 'True' or 'False'. No other text."
    
    total_keys = len(GROQ_API_KEYS)
    attempts = 0
    
    while attempts < total_keys:
        try:
            active_key = GROQ_API_KEYS[current_key_index] if GROQ_API_KEYS else None
            if not active_key:
                return False
                
            client = Groq(api_key=active_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a precise geographic API. You only output 'True' or 'False'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=5,
            )
            
            result = response.choices[0].message.content.strip().lower()
            is_india = "true" in result
            
            INDIA_LOCATION_CACHE[location_lower] = is_india
            return is_india
            
        except RateLimitError:
            current_key_index = (current_key_index + 1) % total_keys
            attempts += 1
            if attempts >= total_keys:
                print("All Groq keys rate-limited during India location check.")
                time.sleep(2)
        except Exception as e:
            print(f"Groq API Error checking if location '{location_str}' is in India: {e}")
            break
            
    # Fallback if API fails completely
    INDIA_LOCATION_CACHE[location_lower] = False
    return False

# ==================================================
# UTILS & HELPERS
# ==================================================

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
    cutoff_date = (pd.Timestamp.utcnow() - pd.DateOffset(days=90)).strftime("%Y-%m-%d")
    try:
        res = (
            supabase.table(TARGET_TABLE)
            .delete()
            .lt("published_at", cutoff_date)
            .execute()
        )
        deleted_count = len(res.data) if res and res.data else 0
        print(f"Deleted {deleted_count} jobs published more than 90 days ago from {TARGET_TABLE}")
    except Exception as e:
        print(f"Cleanup failed: {e}")

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
                messages=[
                    {"role": "system", "content": "You are a strict job description cleaner. You never explain, infer, or add content."},
                    {"role": "user", "content": prompt}
                ],
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

# ==================================================
# DESCRIPTION CLEANING LOGIC 
# ==================================================

def enforce_structural_rules(text):
    if not text:
        return ""

    sections = {"ROLE": [], "REQUIREMENTS": [], "BENEFITS": []}
    current_section = None

    placeholder_phrases = (
        "not mentioned", "not provided", "not listed", "not specified",
        "no specific", "no particular", "no information", "none", "n/a", "(", ")"
    )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        heading = line.upper()

        if heading in ("ROLE", "RESPONSIBILITIES"):
            current_section = "ROLE"
            continue

        if heading in ("REQUIREMENTS", "QUALIFICATIONS"):
            current_section = "REQUIREMENTS"
            continue

        if heading in ("BENEFITS", "PERKS"):
            current_section = "BENEFITS"
            continue

        if current_section and line.startswith("-"):
            if not any(p in line.lower() for p in placeholder_phrases):
                sections[current_section].append(line)

    if not sections["ROLE"]:
        return ""

    output = []
    output.append("ROLE")
    output.extend(sections["ROLE"])

    if sections["REQUIREMENTS"]:
        output.append("")
        output.append("REQUIREMENTS")
        output.extend(sections["REQUIREMENTS"])

    if sections["BENEFITS"]:
        output.append("")
        output.append("BENEFITS")
        output.extend(sections["BENEFITS"])

    return "\n".join(output)

def clean_description_groq(text):
    if not text or len(str(text)) < 50:
        return ""
    
    prompt = f"""
You are a job description cleaner and formatter.

Task:
Extract and structure the job description into these three sections:

ROLE
- List all the key tasks and responsibilities using bullets that begin with '-'.

REQUIREMENTS
- List all skills, experience, and qualifications needed using bullets with '-'.

BENEFITS
- List benefits or perks offered using bullets with '-'.

Formatting rules:
- Use the headings exactly as: ROLE, REQUIREMENTS, BENEFITS
- In case the job description doesn't list any benefits or perks, have only 2 headings- ROLE and REQUIREMENTS. In such cases don't have a separate section for BENEFITS.
- Leave a blank line between each section.
- Do not include company introductions, marketing fluff, culture statements, About company stuff or legal disclaimers.
- Output only plain text (no commentary or emoticons)

Input job posting:
<<<{text}>>>
"""
    raw_response = call_groq_with_retry(prompt)
    if not raw_response:
        return ""
    
    return enforce_structural_rules(raw_response)

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
        # Read directly from the joined companies_sustain object
        comp = row.get("companies_sustain")
        
        # Supabase joins can sometimes return lists depending on relationships
        if isinstance(comp, list):
            comp = comp[0] if comp else None
            
        if not comp:
            return defaults

        return {
            "name": comp.get("company_name", "Unknown"),
            "homepage_url": comp.get("company_url"),
            "logo_file_name": comp.get("logo_file_name"),
            "description": comp.get("description"),
        }
    except Exception:
        return defaults


# ==================================================
# MAIN PIPELINE
# ==================================================

def run_pipeline():
    cleanup_old_jobs()
    print(f"Starting Optimized Pipeline -> Target: {TARGET_TABLE}")
    print("   - Filter: Job ID Exists in Target")
    print("   - Filter: Description words <= 50")  
    print("   - Filter: Published Date > 90 days ago")  
    print("   - Added: Location Standardization")
    print("   - Added: India-only location filter (Groq LLM)")  
    print("   - Updated: Direct mapping to companies_sustain")

    offset = 0
    total_processed = 0
    india_skipped = 0  

    cutoff_date = pd.Timestamp.now(tz="UTC") - pd.DateOffset(days=90)

    while True:
        print(f"\nFetching batch: Rows {offset} to {offset + BATCH_SIZE}...")

        query = "*, companies_sustain(company_name, company_url, logo_file_name, description)"

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
        mask_long_desc = df["word_count"] > 50  

        df["published_date_dt"] = pd.to_datetime(
            df["published_date"], errors="coerce", utc=True
        )
        mask_fresh_date = (df["published_date_dt"].notna()) & (df["published_date_dt"] >= cutoff_date)

        # Apply all initial filters
        temp_mask = mask_new_id & mask_long_desc & mask_fresh_date
        df_temp = df[temp_mask].copy()

        skipped_exists = (~mask_new_id).sum()
        skipped_short = (mask_new_id & ~mask_long_desc).sum()
        skipped_old = (mask_new_id & mask_long_desc & ~mask_fresh_date).sum()

        if skipped_exists > 0:
            print(f"Skipped {skipped_exists} jobs (Already Exist)")
        if skipped_short > 0:
            print(f"Skipped {skipped_short} jobs (Desc <= 50 words)")  
        if skipped_old > 0:
            print(f"Skipped {skipped_old} jobs (Older than 90 days)")  

        if df_temp.empty:
            print("Batch fully filtered out. Next...")
            offset += BATCH_SIZE
            continue

        print(f"Processing {len(df_temp)} valid jobs for location check...")
        processed_rows = []
        batch_india_skipped = 0

        for _, row in df_temp.iterrows():
            job_id = row["id"]

            raw_location = row.get("location")
            standardized_location = standardize_location(raw_location)
            
            # Check if location is in India (using Groq)
            if not is_india_location(standardized_location):
                batch_india_skipped += 1
                india_skipped += 1
                print(f"Skipping Job ID {job_id} - Non-India location: {standardized_location}")
                continue

            raw_desc = row.get("original_description", "")
            generated_description = clean_description_groq(raw_desc)

            if not generated_description:
               print(f"Skipping Job ID {job_id} due to empty/invalid generated description.")
               continue

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

                "location": standardized_location,
                "job_type": normalize_job_type(row.get("job_type")),

                "logo_file": clean_val(comp_data["logo_file_name"]),
                "internal_slug": internal_slug,
                "generated_description": generated_description,
                "company_description": clean_val(comp_data["description"]),

                "department": mapped_department,
                "industry": clean_val(row.get("industry")),
            }

            processed_rows.append(final_obj)

        if batch_india_skipped > 0:
            print(f"Skipped {batch_india_skipped} jobs in this batch (Non-India location)")

        if processed_rows:
            print(f"Upserting {len(processed_rows)} rows to {TARGET_TABLE}...")
            upsert_query = supabase.table(TARGET_TABLE).upsert(processed_rows)
            upsert_res = supabase_execute_with_retry(upsert_query)
            if upsert_res:
                total_processed += len(processed_rows)
                print(f"Success. Total processed: {total_processed}")

        offset += BATCH_SIZE
        time.sleep(Rate_Limit_Sleep)

    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE")
    print(f"Total jobs processed: {total_processed}")
    print(f"Total non-India jobs skipped: {india_skipped}")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_pipeline()
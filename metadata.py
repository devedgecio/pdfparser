import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from dotenv import load_dotenv
import re
import tempfile
import time
import logging
import openai

load_dotenv()
# Get credentials from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-32k")  # Set your model here
timeout = int(os.getenv("OPENAI_TIMEOUT", 15))
max_retries = int(os.getenv("OPENAI_MAX_RETRIES", 5))
MAX_TOKENS = 32768
SAFE_LIMIT = 3500  # Leave room for model output, prevent 413 errors
START_INDEX_TOKEN = "{START_INDEX}"
openai.api_key = OPENAI_API_KEY

def call_with_retries(call_fn, max_retries=5, base_wait=10, *args, **kwargs):
    """
    call_fn: The function to call (should raise exceptions on failure).
    *args, **kwargs: Passed to call_fn.
    Retries on timeout/rate/connection errors, waits as specified in error if possible.
    """
    tries = 0
    while True:
        try:
            return call_fn(*args, **kwargs)
        except Exception as e:
            # --- Parse for "retry after" time ---
            wait_seconds = None
            msg = str(e)
            # Azure/OpenAI error: "...retry after 51 seconds."
            match = re.search(r"retry after (\d+) seconds", msg, re.IGNORECASE)
            if match:
                wait_seconds = int(match.group(1))
            # Azure RateLimitError has retry_after in headers (if HTTP response is accessible)
            if hasattr(e, 'response') and e.response is not None:
                ra = e.response.headers.get('retry-after')
                if ra:
                    try:
                        wait_seconds = int(ra)
                    except:
                        pass
            if wait_seconds is None:
                wait_seconds = base_wait

            tries += 1
            if tries > max_retries:
                logging.error(f"Max retries exceeded. Last error: {e}")
                raise
            logging.warning(f"Error: {e}\nRetrying in {wait_seconds} seconds... (try {tries}/{max_retries})")
            time.sleep(wait_seconds)

def pdf_to_md_pages(pdf_bytes: bytes):
    # Use a NamedTemporaryFile so PyMuPDF can open it as a file path
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_bytes)
        tmp_pdf.flush()
        doc = fitz.open(tmp_pdf.name)
        md_pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)
            md_pages.append(text or "")
    return md_pages

def extract_vars_from_md_with_retry(md_content, max_retries=5):
    def call_openai():
        system_prompt = """


            You are an accurate extractor. Given content from a well log PDF page first 2 pages, extract the following variables exactly as specified. If any variable is missing or indeterminable, return null. Return strictly valid JSON, following the keys outlined below:
            
            
            1. **Surface_Lat:**
            - *Latitude* of the surface location,  If it is in Degrees Minutes Seconds (DMS), convert it into decimal degrees. Don't Convert decimal degree format into Degrees Minutes Seconds (DMS) format. Pick from the smallest table related to Well Details or text if there are multiple values
            2. **Surface_Long:**
            - *Longitude* of the surface location, If it is in Degrees Minutes Seconds (DMS), convert it into decimal degrees.  Don't Convert decimal degree format into Degrees Minutes Seconds (DMS) format. Pick from the smallest table related to Well Details or text if there are multiple values
            3. **Well_API:**
            - *API number* of the well, which includes a two-digit state code, a three-digit county code, and a five-digit well number, formatted as "42-..." (e.g., 42-003-49073-0000, 42-003-49073 means start with 42, don't skip any digit).
            4. **Projection:**
            - The *projection* or map reference, typically formatted as something like CRS "US-SPC27-EXACT / Texas Central 4203"  or similar terms.
            -  US State Plane 1927 
            5. **Depth_Reference_Elevation:**
            - *Sum of ground elevation + RKB Elevation* you see. or if Well @ is already given. it is just a value, no text. So find both value and sum them or just find the value of Well @ and return it.

            --- 
            Remember: We are collecting this data for the Well Log PDF page, so the values should be relevant to the well log page.
            ---



            ### Extraction Guidelines:

            - When you set the above-mentioned variables' values, you have to **add field text/paragraph/table/text** (i.e., from which paragraph/table/text you took that value).
            - Ensure you keep **units of values** as standard, as mentioned.
            - **Respond strictly with valid JSON format**.
            - For each value, include the **exact text** from which it was extracted.
            - **Avoid **big tables (more than 3,4 rows and columns)** in the document when extracting values; only extract from smaller, relevant text not big tables.**
            - if no text/value is found relevant of variable just set as No Value
            - "RECHECK all values and there text, if anything is not according to instructions then correct it, if something is from big table then set as No Value"
            ---



            **Instruction for JSON output formatting:**

            - Always produce **strictly valid JSON** that can be parsed by standard JSON parsers without errors.
            - Ensure all **strings are enclosed in double quotes** (`"`).
            - **Escape all special characters inside strings**, including double quotes (`"`), backslashes (`\\`), and control characters, using proper JSON escaping rules.
            - Do **not** include trailing commas after the last element in arrays or objects.
            - Verify that all keys and values are properly separated by commas.
            - Avoid using `null` or other special values unless explicitly allowed in the JSON schema.
            - When outputting nested or complex JSON, double-check the syntax carefully before returning.
            - If including text with characters like degrees (`°`), minutes (`'`), or seconds (`"`), make sure these are safely encoded or escaped to avoid parsing errors.
            - When unsure, output the JSON as a single well-escaped string.
            Always prioritize JSON validity over brevity or formatting style.


            Maintain the following order of varaibles in the JSON output:
            - **Well_API**
            - **Projection**
            - **Surface_Lat**
            - **Surface_Long**
            - **Depth_Reference_Elevation**
            ---

            
            ### Example JSON Response:
            {
            "Well_API":["42-461-42137-0000","Field Spraberry (Trend Area) R 40 EXC Upton County, TX API 42-461-42137-0000"],
            "Projection":["NUS State Plane 1927","US State Plane 1927"],
            "Surface_Lat":["31.683603","Slot Location Latitude: 31°49'0.9711\"N"],
            "Surface_Long":["-103.84236","Slot Location Longitude:  -103°50'32.4967\"W"],
            "Depth_Reference_Elevation":["2782.00ft","Rig: H&P 637 (KB) to Facility Vertical Datum 2756.00ft + Rig: H&P 637 (KB) to Ground Level at Slot (Owens-Cravens W31D No. 4H) 26.00ft"]
            }
        """
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": md_content}
            ],
            temperature=0,
        )
        return response
    
    try: 
        response = call_with_retries(call_openai, max_retries=max_retries)
        raw_content = response.choices[0].message.content.strip()
        if raw_content.startswith("```"):
                raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
                raw_content = re.sub(r"\n?```$", "", raw_content)
        return json.loads(raw_content)
    except Exception as e:
            print(f":x: Error with: {e}")
            return None
    
    
def process_pdf(pdf_path, output_dir="page_md"):
    md_files = pdf_to_md_pages(pdf_path, output_dir)
    results = []
    i = 0
    mdd = ""
    for md in md_files:
        mdd += md.read_text(encoding="utf-8") 
        print(len(mdd))
        i = i +1
        if i == 2:
            break
    print(f"→ Processing")

    extractedd = extract_vars_from_md_with_retry(mdd)
    print("extraction done", extractedd)
    results= extractedd
    return results


def process_uploaded_pdf(pdf_bytes: bytes, output_dir=None):
    """
    Accepts PDF file as bytes, returns extracted metadata as a dict.
    """
    md_pages = pdf_to_md_pages(pdf_bytes)
    mdd = ""
    for i, md in enumerate(md_pages):
        mdd += md
        if i == 2:  # Only first 3 pages
            break
    extracted_data = extract_vars_from_md_with_retry(mdd)
    return extracted_data

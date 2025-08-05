from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile, shutil, os, zipfile, hashlib, json
from typing import List

from utils import process_pdf
from metadata import process_uploaded_pdf
import os
from dotenv import load_dotenv
app = FastAPI()

# Load the configuration for input and output paths from the config file
load_dotenv()

# Retrieve the output path from the environment variable
output_path = os.getenv("output_path", "")

if not output_path:
    raise ValueError("Input and Output paths are missing in the config file.")

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Helper functions for saving uploaded files and extracting zip contents
def save_uploaded_file(uploaded: UploadFile, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded.filename)
    with open(file_path, "wb") as f:
        f.write(uploaded.file.read())
    return file_path

def extract_zip(uploaded: UploadFile, extract_dir: str) -> List[str]:
    temp_zip_path = os.path.join(extract_dir, "temp.zip")
    with open(temp_zip_path, "wb") as f:
        f.write(uploaded.file.read())
    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    os.remove(temp_zip_path)
    pdfs = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdfs.append(os.path.join(root, file))
    return pdfs

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    temp_dir = tempfile.mkdtemp()
    pdf_paths = []
    results = []

    try:
        # Save uploaded files
        for uploaded in files:
            filename = uploaded.filename
            if filename.lower().endswith('.zip'):
                pdf_paths += extract_zip(uploaded, temp_dir)
            elif filename.lower().endswith('.pdf'):
                pdf_path = save_uploaded_file(uploaded, temp_dir)
                pdf_paths.append(pdf_path)

        if not pdf_paths:
            shutil.rmtree(temp_dir)
            raise HTTPException(status_code=400, detail="No PDF files found.")

        # Process each PDF file and collect result (success or error)
        summary = []
        for pdf_path in pdf_paths:
            file_result = {
                "file": os.path.basename(pdf_path),
                "status": "success",
                "error": None
            }
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                file_hash = hashlib.sha256(pdf_bytes).hexdigest()

                # Metadata extraction
                extracted_data = process_uploaded_pdf(pdf_bytes, output_path)
                if extracted_data:
                    # Save metadata as JSON in the output folder
                    meta_path = os.path.join(output_path, f"{base_name}.json")
                    with open(meta_path, "w") as jf:
                        json.dump(extracted_data, jf, indent=4)

                # Tables extraction and saving
                tables = process_pdf(pdf_bytes, file_hash)
                for idx, table_df in enumerate(tables if isinstance(tables, list) else [tables]):
                    table_num = idx + 1
                    csv_filename = f"{base_name}.csv"
                    csv_path = os.path.join(output_path, csv_filename)
                    table_df.to_csv(csv_path, index=False)  # Save CSV directly to output folder

                file_result["status"] = "success"
            except Exception as e:
                file_result["status"] = "failed"
                file_result["error"] = str(e)
            summary.append(file_result)

        shutil.rmtree(temp_dir)
        # Returning summary response showing where files are stored
        return JSONResponse(
            status_code=200,
            content={
                "summary": [
                    {
                        "file": r["file"],
                        "status": r["status"],
                        "error": r["error"],
                    } for r in summary
                ],
                "message": f"Processing complete. Data is available in the output folder: {output_path}",
            }
        )
    except Exception as e:
        shutil.rmtree(temp_dir)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Top-level server error: {str(e)}"}
        )

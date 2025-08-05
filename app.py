import streamlit as st
import hashlib
import pandas as pd
from utils import process_pdf
from metadata import process_uploaded_pdf
import json
import os
from dotenv import load_dotenv
# Streamlit page setup
st.set_page_config(":page_facing_up: PDF Table Extractor", layout="wide")
st.title(":page_facing_up: PDF Table Extractor")
# Optional CSS – hide translucent overlay
st.markdown(
    """
    <style>
    div[data-testid="stAppViewContainer"] > div:has(div.stSpinner) {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# :one:  Load JSON Configuration for Input and Output Paths --------------------------------
load_dotenv()

input_path = os.getenv("input_path", "")
output_path = os.getenv("output_path", "")
if not input_path or not output_path:
    st.stop()  # Stop if no paths are provided in config
# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
# :two:  Process PDFs from the Input Path --------------------------------------------------
pdf_files = [f for f in os.listdir(input_path) if f.lower().endswith('.pdf')]
if not pdf_files:
    st.stop()  # Stop if no PDF files are found in the input directory
# Show the loading spinner until the files are processed
with st.spinner("Processing files... Please wait."):
    # Initialize progress bar
    progress_bar = st.progress(0)
    total_files = len(pdf_files)
    processed_files = 0
    # Iterate through each PDF file and process it
    for filename in pdf_files:
        try:
            file_path = os.path.join(input_path, filename)
            # --- Get base file name ---
            base_name = os.path.splitext(filename)[0]
            # Process the uploaded PDF
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
                file_hash = hashlib.sha256(pdf_bytes).hexdigest()
            # Extract metadata
            extracted_data = process_uploaded_pdf(pdf_bytes, output_path)
            # Show extracted data and save as JSON
            if extracted_data:
                st.write(f":page_facing_up: **File:** {filename} - Extraction complete!")
                st.json(extracted_data, expanded=False)  # Display the extracted data as a JSON structure
                first_values = {
                    "Key": [],
                    "First Value": []
                }
                for key, value_list in extracted_data.items():
                    first_value = value_list[0] if value_list else "No value"
                    first_values["Key"].append(key)
                    first_values["First Value"].append(first_value)
                # Convert to DataFrame
                df = pd.DataFrame(first_values)
                # Display the DataFrame in Streamlit
                st.title(f"Extracted MetaData for {filename}: ")
                st.write(df)
                # Save the JSON data in output directory (no separate folder)
                json_path = os.path.join(output_path, f"{base_name}.json")
                with open(json_path, "w") as json_file:
                    json.dump(extracted_data, json_file, indent=4)
                st.success(f"Metadata saved to '{json_path}'.")
            # Process PDF tables
            with st.spinner(f":hammer_and_wrench: Processing PDF: {filename}..."):
                tables = process_pdf(pdf_bytes, file_hash)
                # Display each table
                for idx, table_df in enumerate(tables if isinstance(tables, list) else [tables]):
                    table_num = idx + 1
                    st.write(f":clipboard: **Table #{table_num} from {filename}**")
                    st.dataframe(table_df)  # Display the table (non-editable, for display only)
                    # Save each table as a CSV in the output directory (no separate folder)
                    csv_filename = f"{base_name}.csv"  # Include table number for uniqueness
                    csv_path = os.path.join(output_path, csv_filename)
                    table_df.to_csv(csv_path, index=False)  # Save CSV directly to output directory
                    st.success(f"Table {table_num} from {filename} saved as CSV: {csv_path}")
            # Update progress bar after processing each file
            processed_files += 1
            progress_bar.progress(processed_files / total_files)
        except Exception as e:
            st.error(f":x: **Problem occurred with file: {filename}**")
            st.error(f"Error: {str(e)}")  # Display the error message for the file
            continue  # Skip the current file and continue with the next one




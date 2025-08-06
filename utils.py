import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentAnalysisFeature

from dotenv import load_dotenv
import os
import re
import numpy as np
import warnings
from metadata import call_with_retries

# 🔐 Load environment variables
load_dotenv()

# Get credentials from environment variables

AZURE_KEY = os.getenv("AZURE_FORM_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-32k")  # Set your model here
timeout = int(os.getenv("OPENAI_TIMEOUT", 15))
max_retries = int(os.getenv("OPENAI_MAX_RETRIES", 5))
MAX_TOKENS = 32768
SAFE_LIMIT = 3500  # Leave room for model output, prevent 413 errors
START_INDEX_TOKEN = "{START_INDEX}"






def extract_tables_from_pdf_with_retry(file_bytes: bytes, max_retries=5):
    def azure_call():
        client = DocumentIntelligenceClient(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_KEY)
        )
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=file_bytes),
            features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION]
        )
        return poller.result()

    result = call_with_retries(azure_call, max_retries=max_retries)

    tables = []
    for table in result.tables:
        table_data = [['' for _ in range(table.column_count)] for _ in range(table.row_count)]
        for cell in table.cells:
            table_data[cell.row_index][cell.column_index] = cell.content
        df_raw = pd.DataFrame(table_data)
        tables.append(df_raw)
    return tables



unwanted_values = [':unselected:', ':selected:','\n:unselected: ', '\n:selected:    ']

# Function to check and replace unwanted values with an empty string
def replace_unwanted_values(df):
    # Iterate over each row and column
    for row_index, row in df.iterrows():
        for col_name, value in row.items():
            # Check if the cell value is not NA and matches any unwanted value
            if pd.notna(value) and value in unwanted_values:
                df.at[row_index, col_name] = ""  # Replace with empty string
    return df


def normalize_table_by_column_first_cell(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each column, find the first non-empty cell.
      - If it's a *non-numeric* string, promote it as that column's header.
      - If it's numeric (and cells above were empty), DO NOT promote it;
        instead assign a dummy header 'Column_{i}' and keep that numeric
        value as data.
      - If the entire column is empty, also assign a dummy header.

    Rows at or above a promoted header row are cleared (set to None) for that column.
    Columns with dummy headers keep all rows as data.
    """

    if df.empty:
        return df

    # --- UPDATED helper(s) ---
    def _token_is_num(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False

    def _is_numeric_like(val):
        if isinstance(val, (int, float)) and not pd.isna(val):
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return False
            # NEW: treat multi-token sequences made ONLY of numeric tokens
            # (e.g. "0.00 0.00 0.00 0.00") as numeric-like so they are NOT promoted to header
            tokens = re.split(r"\s+", s)
            if len(tokens) > 1 and all(_token_is_num(t) for t in tokens):
                return True
            # fallback single-token numeric check
            return _token_is_num(s)
        return False

    headers = []
    header_row_indices = []  # -1 means no header promoted (all rows are data)

    for col_idx in range(df.shape[1]):
        header_found = False
        for row_idx in range(df.shape[0]):
            cell = df.iat[row_idx, col_idx]
            if isinstance(cell, str):
                cell = cell.strip()

            if not (pd.isna(cell) or cell in ("", None) or cell is pd.NA):
                # Decide whether to promote or create dummy
                if _is_numeric_like(cell):
                    # numeric => treat as data, create dummy header
                    headers.append(f"Column_{col_idx}")
                    header_row_indices.append(-1)  # so every row remains data
                else:
                    headers.append(str(cell))
                    header_row_indices.append(row_idx)
                header_found = True
                break

        if not header_found:
            headers.append(f"Column_{col_idx}")
            header_row_indices.append(-1)  # all empty -> dummy header, keep all (effectively all None anyway)

    new_data = []
    for row_idx in range(df.shape[0]):
        row = []
        row_has_any = False
        for col_idx in range(df.shape[1]):
            header_row_idx = header_row_indices[col_idx]
            val = df.iat[row_idx, col_idx]
            # If a real header was promoted (index >=0), blank out that header row and anything above it
            if header_row_idx >= 0 and row_idx <= header_row_idx:
                row.append(None)
            else:
                row.append(val)
                if not (pd.isna(val) or val in ("", None)):
                    row_has_any = True
        if row_has_any:
            new_data.append(row)

    return pd.DataFrame(new_data, columns=headers)
def keep_df_if_header_keywords(df: pd.DataFrame,
                               phrases=None,
                               rows_to_check=(0, 1),
                               case_insensitive=True,
                               min_matches=1,
                               check_columns=True) -> bool:
    """
    Return True if ANY (or at least `min_matches`) of the target header phrases
    appear in *either* of the specified rows (default: first two rows).
    Otherwise return False (caller can discard the whole DataFrame).

    Parameters
    ----------
    df : pd.DataFrame
    phrases : list[str] or None
        Header phrases to look for. If None, uses the default survey header set.
    rows_to_check : tuple[int]
        Row indices to scan (default first two: (0,1)).
    case_insensitive : bool
        If True, matching is case-insensitive.
    min_matches : int
        Minimum number of distinct phrases that must be found across the checked rows
        to keep the DataFrame. Default 1 (i.e., any phrase).

    Notes
    -----
    - A "phrase" is matched by simple substring search inside the concatenated row string.
    - Empty / NaN cells are skipped.
    """

    def is_probably_data(df: pd.DataFrame) -> bool:

        row1 = df.iloc[1] if df.shape[0] > 1 else df.iloc[0]
        numeric_count = row1.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().sum()
        numeric_ratio = numeric_count / df.shape[1]
        
        print("DATA TEST GOING ON", numeric_ratio)
        return numeric_ratio > 0.4
    
    if df is None or df.empty:
        return False

    if phrases is None:
        phrases = [
            "MD Usft",
            "INC deg",
            "AZI deg",
            "TVD Usft",
            "NS Usft",
            "EW Usft",
            "VS Usft",
            "DLS deg/100Usft",
            "BR deg/100Usft",
            "TR deg/100Usft",
            "TVD",
            "Measured Depth (usft)",
            "Inclination (°)",
            "Azimuth (°)",
            "Vertical Depth (usft)",
            "+N/-S (usft)",
            "+E/-W (usft)",
            "Vertical Section (usft)",
            "Dogleg Rate (°/100usft)",
            "Build Rate (°/100usft)",
            "Turn Rate (°/100usft)"
        ]

    # Prepare phrases
    target = [p.lower() if case_insensitive else p for p in phrases]

    found = set()

    if check_columns:
        col_cells = [
            str(c).strip()
            for c in df.columns
            if str(c).strip() not in ("", "None", "nan", "NaN")
        ]
        if col_cells:
            col_text = " ".join(col_cells)
            col_cmp = col_text.lower() if case_insensitive else col_text
            for phrase in target:
                if phrase in col_cmp:
                    found.add(phrase)
                    if len(found) >= min_matches:
                        return True

    for r in rows_to_check:
        if r >= df.shape[0]:
            continue
        # Collect row cells (skip NaN)
        cells = [
            str(v).strip()
            for v in df.iloc[r].tolist()
            if not (pd.isna(v) or str(v).strip() == "")
        ]
        if not cells:
            continue
        row_text = " ".join(cells)
        row_cmp = row_text.lower() if case_insensitive else row_text

        for phrase in target:
            if phrase in row_cmp:
                found.add(phrase)
                if len(found) >= min_matches:
                    return True
                
    # If a Df has no matching name but the mostly it's numeric values
    # it should have more than 8 rows
    # it should have 40% values as numbers 
    if df.shape[1] > 8:
        print("this is a big table with no matching headers, potential candidate")
        if is_probably_data(df):
            return True
        else:
            print("skipped table becasue most values were not numbers") 
    
    return False

def filter_dfs_by_header_keywords(dfs,
                                  phrases=None,
                                  rows_to_check=(0, 1),
                                  case_insensitive=True,
                                  min_matches=1,
                                  check_columns=True):   
    """
    Given an iterable of DataFrames, return only those that pass the header keyword test.
    """
    return [
        df for df in dfs
        if keep_df_if_header_keywords(
            df,
            phrases=phrases,
            rows_to_check=rows_to_check,
            case_insensitive=case_insensitive,
            min_matches=min_matches,
            check_columns=check_columns
        )
    ]

def merge_two_line_headers(dfs):

    def is_data_like(val):
        """Return True only if the cell looks like actual numeric data (single
        number or sequence of numbers), NOT a unit label like 'deg/100Usft'."""
        if pd.isna(val):
            return False
        if isinstance(val, (int, float)):
            return True
        s = str(val).strip()
        if s == "":
            return False
        # Pure single number
        if re.fullmatch(r'[-+]?\d+(\.\d+)?', s):
            return True
        # Sequence of numbers separated by spaces (e.g. '0.00 0.00')
        if re.fullmatch(r'([-+]?\d+(\.\d+)?\s+)+[-+]?\d+(\.\d+)?', s):
            return True
        # Otherwise (contains letters mixed with digits like 'deg/100Usft') treat as header text
        return False

    if isinstance(dfs, pd.DataFrame):  # (unchanged logic to support single DF)
        dfs = [dfs]

    merged = []
    for _df in dfs:
        print(_df.columns)
        if _df is None or _df.empty:
            merged.append(_df)
            continue
        first_row = _df.iloc[0]
        # CHANGED: use is_data_like (ignoring first column) instead of any digit anywhere
        if any(is_data_like(v) for v in first_row.iloc[1:]):
            merged.append(_df)
            continue
        print(first_row)
        print("--"*50)

        new_cols = []
        for col_name, second_part in zip(_df.columns, first_row):
            if pd.isna(second_part) or str(second_part).strip() == "":
                combined = str(col_name).strip()
            else:
                combined = f"{col_name} {second_part}".strip()
            combined = " ".join(combined.split())
            new_cols.append(combined)

        new_df = _df.iloc[1:].copy()
        new_df.columns = new_cols
        merged.append(new_df.reset_index(drop=True))
    return merged

def merge_dataframes_with_same_structure(df_list):
    merged_dfs = []  # List to store the merged DataFrames
    unmerged_dfs = []  # List to store the unmerged DataFrames
    visited_dfs = []  # List to track which DataFrames have already been merged
    
    for i, df in enumerate(df_list):
        if i in visited_dfs:  # Skip if this DataFrame has already been merged
            continue

        # Compare with the other DataFrames
        matching_dfs = [df]
        for j, other_df in enumerate(df_list[i+1:], i+1):
            if j in visited_dfs:  # Skip already merged DataFrames
                continue
            
            # Check column count
            if df.shape[1] == other_df.shape[1]:  # Check if column counts match
                # Check if headers (columns) match exactly
                if all(df.columns == other_df.columns):
                    matching_dfs.append(other_df)
                    visited_dfs.append(j)  # Mark this DataFrame as visited
        
        # If there are matching DataFrames, merge them
        if len(matching_dfs) > 1:
            merged_df = pd.concat(matching_dfs, ignore_index=True)  # Concatenate matching DataFrames
            merged_dfs.append(merged_df)  # Add the merged DataFrame to the list
        else:
            unmerged_dfs.append(df)  # Add the original DataFrame to the unmerged list
    
    # Combine both merged and unmerged DataFrames
    final_dfs = merged_dfs + unmerged_dfs  # Merge both lists (merged + unmerged)
    
    # Return the combined list of DataFrames
    return final_dfs

def split_multi_numeric_columns(dfs):
    """
    For each DataFrame:
      - Look at the *first data row* (row 0).
      - If ANY cell in that row contains 2+ numeric values (space-separated),
        then we scan all columns and split those columns whose header looks like
        the combined pattern e.g. "INC AZI deg deg" or "EW VS Usft Usft".
      - Pattern we split:
            <NAME1> <NAME2> <UNIT> <UNIT>   (4 tokens, last two identical)
        and the cell has exactly 2 numeric values.
      - We replace that single column by two columns:
            "<NAME1> <UNIT>", "<NAME2> <UNIT>"
        populated by the two numeric values for every row (parsed each time).
    """

    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    num_re = re.compile(r'[-+]?\d+(?:\.\d+)?')
    
    def parse_numbers(val):
        if isinstance(val, (pd.Series, list, tuple)):
            # take first non-null scalar
            for x in val:
                if not (pd.isna(x) if not isinstance(x, (list, tuple, pd.Series)) else True) :
                    val = x
                    break
            else:
                return []
        try:
            if pd.isna(val):
                return []
        except Exception:
            return []
        s = str(val)
        s = s.replace(',', '')
        return [float(x) for x in num_re.findall(s)]


    processed = []
    for df in dfs:
        if df is None or df.empty:
            processed.append(df)
            continue

        first_row = df.iloc[0]
        # Detect if we even need to do splitting (any cell with 2+ nums)
        if not any(len(parse_numbers(v)) >= 2 for v in first_row):
            processed.append(df)
            continue

        new_col_names = []
        new_col_data = []
        for col_idx, col in enumerate(df.columns):
            header = str(col).strip()
            tokens = header.split()
            col_series = df.iloc[:, col_idx]        # exact Series
            try:
                first_cell_vals = parse_numbers(col_series.iloc[0])
            except Exception as e:
                warnings.warn(f"Skipping column '{header}' (parse error: {e})")
                continue   # ⬅️ skip completely

            try:
                split = False
                if len(first_cell_vals) >= 2:
                    if len(tokens) == 4 and tokens[2] == tokens[3]:
                        # <NAME1> <NAME2> <UNIT> <UNIT>
                        unit = tokens[2]
                        name1 = f"{tokens[0]} {unit}"
                        name2 = f"{tokens[1]} {unit}"
                        split = True
                    elif len(tokens) == 3:
                        # <NAME1> <NAME2> <UNIT>
                        unit = tokens[2]
                        name1 = f"{tokens[0]} {unit}"
                        name2 = f"{tokens[1]} {unit}"
                        split = True
                    elif len(tokens) == 2:
                        # <NAME1> <NAME2>
                        name1, name2 = tokens
                        split = True    # we split this pattern too

                if split:
                    col_vals1, col_vals2 = [], []
                    for v in col_series:
                        nums = parse_numbers(v)
                        if len(nums) >= 2:
                            col_vals1.append(nums[0])
                            col_vals2.append(nums[1])
                        else:
                            col_vals1.append(np.nan)
                            col_vals2.append(np.nan)

                    def uniq(n):
                        base = n
                        k = 1
                        while n in new_col_names:
                            k += 1
                            n = f"{base}_{k}"
                        return n

                    new_col_names.extend([uniq(name1), uniq(name2)])
                    new_col_data.extend([col_vals1, col_vals2])
                else:
                    # keep original (no error, just not a split pattern)
                    new_col_names.append(header)
                    new_col_data.append(col_series.tolist())

            except Exception as e:
                warnings.warn(f"Skipping column '{header}' (split error: {e})")
                # ⬅️ skip entirely (do not append original) per your request
                continue

        new_df = pd.DataFrame(dict(zip(new_col_names, new_col_data)))
        processed.append(new_df)

    return processed

def propagate_multi_values_to_next_column(dfs, min_required=2):
    """
    If a cell in column i holds multiple numeric values (space / comma separated) AND
    the corresponding cell in column i+1 is NA, move the 2nd numeric value into column i+1
    (keep the 1st in column i). Generic: works for any adjacent columns.

    Examples:
        Col[i] = "0.32 112.33", Col[i+1] = <NA>  --> Col[i]="0.32", Col[i+1]=112.33
        Col[i] = "0.11 0.00 5.5", Col[i+1]=<NA>  --> Col[i]="0.11", Col[i+1]=0.00 (extra numbers ignored)

    Parameters
    ----------
    dfs : DataFrame or list[DataFrame]
    min_required : int
        Minimum numbers in source cell to trigger the propagation (default 2).
    """

    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    num_re = re.compile(r'[-+]?\d+(?:\.\d+)?')

    def extract_nums(val):
        try:
            if isinstance(val, pd.Series):
                s = " ".join(str(x) for x in val.tolist())
            else:
                if pd.isna(val):
                    return []
                if isinstance(val, (int, float)):
                    return [float(val)]
                s = str(val)
            # Remove all commas from string before extracting numbers
            s = s.replace(',', '')

            return [float(x) for x in num_re.findall(s)]
        except Exception:
            return []

    def is_na(v):
        try:
            if isinstance(v, pd.Series):
                return v.isna().all()
            return pd.isna(v)
        except Exception:
            return False

    out = []
    for df in dfs:
        if df is None or df.empty:
            out.append(df)
            continue
        print(df.columns)
        df2 = df.copy()
        print("cols:", df2[:5])
        print("*"*50)
        cols = list(df2.columns)
        for ci in range(len(cols) - 1):
            c_curr, c_next = cols[ci], cols[ci + 1]
            for r in range(len(df2)):
                if is_na(df2.at[r, c_next]):
                    nums = extract_nums(df2.at[r, c_curr])
                    # display(nums[:5])
                    if len(nums) >= min_required:
                        # Assign first to current, second to next
                        df2.at[r, c_curr] = nums[0]
                        df2.at[r, c_next] = nums[1]
        out.append(df2)
    return out

def group_and_merge_tables_dynamic(dfs,
                                   header_alpha_vs_numeric=True,
                                   min_key_increase_ratio=0.7,
                                   require_key_increase=True):
    """
    Dynamically merges contiguous DataFrames that are continuations of each other
    even if each has DIFFERENT header names or one (or both) lost headers.

    Heuristics:
      1. Header Detection (no fixed names):
         - A DF is considered to *have* a header if (a) more columns contain
           alphabetic chars than purely numeric tokens, OR (b) pandas did not
           auto-assign a simple RangeIndex (we accept user-provided names).
      2. If exactly one DF in a pair lacks a header, we copy the other's header.
         If both lack headers but column counts match, we still try to merge by
         *position* only.
      3. Row Pattern Continuity: last row of current block vs first row of next
         (sequence of numeric/text markers) must match.
      4. Dynamic Key Column Detection:
         - For each numeric column, compute the ratio of strictly increasing
           steps (col[i] < col[i+1]) over the last window (entire block).
         - Pick the column with the highest ratio >= min_key_increase_ratio as
           the key (tie-breaker = largest range). No hard-coded names.
         - On merge, candidate first value must be > last value (if enabled).
      5. If no suitable key column is found, we can still merge (unless
         require_key_increase=True, then we demand an increasing key).
    Returns list of merged DataFrames (each retains chosen key in attrs['key_col']).
    """

    def is_numeric_token(x):
        try:
            float(str(x).strip())
            return True
        except:
            return False

    def has_header(df):
        cols = list(df.columns)
        # If they're the standard RangeIndex names (0..n-1) or duplicates of first row?
        # Basic heuristic: check alpha vs numeric-like names.
        alpha = sum(any(ch.isalpha() for ch in str(c)) for c in cols)
        numeric_like = sum((not any(ch.isalpha() for ch in str(c))) and any(ch.isdigit() for ch in str(c)) for c in cols)
        if header_alpha_vs_numeric:
            return alpha >= numeric_like
        return True  # fallback

    def row_pattern(row):
        return tuple('N' if (is_numeric_token(v) or pd.isna(v)) else 'T' for v in row)

    def detect_key_column(df):
        numeric_cols = []
        for c in df.columns:
            series = df[c]
            # quick numeric check: at least 80% convertible non-NaNs
            vals = []
            conv_ok = 0
            total = 0
            for v in series:
                if pd.isna(v):
                    continue
                total += 1
                try:
                    vals.append(float(v))
                    conv_ok += 1
                except:
                    pass
            if total == 0:
                continue
            if conv_ok / total < 0.8:
                continue
            numeric_cols.append(c)

        best_col = None
        best_ratio = -1
        best_range = -1
        for c in numeric_cols:
            col = pd.to_numeric(df[c], errors='coerce')
            col = col.dropna()
            if len(col) < 3:
                continue
            increases = (col.diff().dropna() > 0).sum()
            steps = (col.diff().dropna() != 0).sum()
            if steps == 0:
                continue
            ratio = increases / steps
            rng = (col.max() - col.min())
            if ratio > best_ratio or (ratio == best_ratio and rng > best_range):
                best_ratio = ratio
                best_range = rng
                best_col = c
        if best_col and best_ratio >= min_key_increase_ratio:
            return best_col, best_ratio
        return None, best_ratio

    def prepare_headers(base, nxt):
        base_has = has_header(base)
        nxt_has = has_header(nxt)
        b = base.copy()
        n = nxt.copy()
        if base_has and not nxt_has and len(b.columns) == len(n.columns):
            n.columns = b.columns
        elif nxt_has and not base_has and len(b.columns) == len(n.columns):
            b.columns = n.columns
        # if neither has header OR both have (even if different names), keep as-is (position-driven)
        return b, n

    def can_append(base_df, next_df):
        if base_df.shape[1] != next_df.shape[1]:
            return None

        b_fixed, n_fixed = prepare_headers(base_df, next_df)

        # Row pattern continuity
        if row_pattern(b_fixed.iloc[-1]) != row_pattern(n_fixed.iloc[0]):
            return None

        # Dynamic key detection on the combined concept (use base so far)
        key_col, ratio = detect_key_column(b_fixed)
        if key_col:
            try:
                last_val = float(b_fixed[key_col].iloc[-1])
                first_val = float(n_fixed[key_col].iloc[0])
                if require_key_increase and not (first_val > last_val):
                    return None
            except:
                if require_key_increase:
                    return None
        else:
            if require_key_increase:
                return None  # no key column to validate continuity

        merged = pd.concat([b_fixed, n_fixed], ignore_index=True)
        merged.attrs['key_col'] = key_col
        merged.attrs['key_increase_ratio'] = ratio
        return merged

    if not dfs:
        return []

    result = []
    current = dfs[0].copy()
    # annotate initial key
    k0, r0 = detect_key_column(current)
    current.attrs['key_col'] = k0
    current.attrs['key_increase_ratio'] = r0

    for nxt in dfs[1:]:
        merged = can_append(current, nxt)
        if merged is not None:
            current = merged
        else:
            result.append(current)
            current = nxt.copy()
            k, r = detect_key_column(current)
            current.attrs['key_col'] = k
            current.attrs['key_increase_ratio'] = r
    result.append(current)
    return result

def clean_bracket_content(dfs):
    # Regex to find content inside [] or ()
    pattern = re.compile(r'([^\[\(]+)([\[\(])([^\]\)]+)([\]\)])')  # Capture part outside brackets, inside brackets, and brackets

    def is_single_char_inside_brackets(content):
        return len(content) == 1  # Return True if content inside brackets is exactly one character

    cleaned_dfs = []
    for df in dfs:
        new_columns = []
        for col in df.columns:
            # Search for pattern inside brackets
            match = pattern.search(col)
            if match:
                # Extract the different parts
                before_bracket = match.group(1)  # The part before the brackets (MD, Azimuth, etc.)
                opening_bracket = match.group(2)  # The opening bracket
                content = match.group(3)         # The content inside the bracket
                closing_bracket = match.group(4)  # The closing bracket
                
                if is_single_char_inside_brackets(content):
                    # If content inside brackets is one character, remove the content but keep the brackets intact
                    new_col = before_bracket + opening_bracket + closing_bracket  # Keep the brackets and the part outside intact
                else:
                    # If content is more than one character, leave the brackets and content as they are
                    new_col = col
            else:
                new_col = col  # If no brackets are found, keep the column as is

            new_columns.append(new_col)
        df.columns = new_columns
        cleaned_dfs.append(df)
    return cleaned_dfs

def concat_empty_colnames_to_next_list(dfs):
    updated_dfs = []
    for df in dfs:
        cols = df.columns.tolist()
        cols_to_drop = []
        new_cols = cols.copy()

        for i, col in enumerate(cols):
            col_data = df[col]
            # If col_data is DataFrame (multi-columns with same name), take first column only
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            print(f"Column '{col}' type: {type(col_data)}")
            if col.isalpha() and col_data.isna().all():
                if i < len(cols) - 1:
                    new_cols[i + 1] = col + new_cols[i + 1]
                    cols_to_drop.append(col)

        # Drop the empty columns
        df = df.drop(columns=cols_to_drop)

        # Remove dropped columns from new_cols
        new_cols = [c for c in new_cols if c not in cols_to_drop]

        # Assign updated column names
        df.columns = new_cols

        updated_dfs.append(df)
    return updated_dfs


def make_cols_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            cols[idx] = f"{dup}_{i+1}"
    df.columns = cols
    return df



def merge_adjacent_columns(dfs):

    for idx, df in enumerate(dfs):
        dup_cols = df.columns[df.columns.duplicated()]
        if len(dup_cols) > 0:
            print(f"DataFrame {idx} has duplicate columns: {list(dup_cols)}")

    dfs= [make_cols_unique(df) for df in dfs]

    merged_dfs = []
    for df in dfs:
        df = df.copy()
        cols = df.columns.tolist()
        i = 0
        while i < len(cols) - 1:
            col1, col2 = cols[i], cols[i+1]
            col1_vals = df[col1]
            col2_vals = df[col2]

            conflict_mask = (
                col1_vals.notna() & 
                col2_vals.notna() & 
                (col1_vals != col2_vals)
            )
            if not conflict_mask.any() and (col1_vals.isna().any() or col2_vals.isna().any()):
                merged_col = col1_vals.combine_first(col2_vals)
                merged_name = col1 + ' ' + col2

                # Drop old columns before inserting merged
                df = df.drop(columns=[col1, col2])

                # Insert merged column at position i
                df.insert(i, merged_name, merged_col)

                # Update columns list after modification
                cols = df.columns.tolist()

                # Move index forward by 1 since we merged two columns into one
                i += 1
            else:
                i += 1
        merged_dfs.append(df)
    return merged_dfs

def unify_column_names_by_substring(dfs):
    # Collect all columns with their df indices
    all_cols = []
    for idx, df in enumerate(dfs):
        for col in df.columns:
            all_cols.append((idx, col))
    
    # For each column, check substring relationship with columns in other dfs
    for i, (idx_i, col_i) in enumerate(all_cols):
        for j, (idx_j, col_j) in enumerate(all_cols):
            if idx_i != idx_j:
                if col_i in col_j and col_i != col_j:
                    # col_i is substring of col_j, keep longer col name (col_j)
                    if len(col_j) > len(col_i):
                        # Rename col_i to col_j in df idx_i
                        dfs[idx_i].rename(columns={col_i: col_j}, inplace=True)
                        # Update all_cols entry to prevent repeated renaming
                        all_cols[i] = (idx_i, col_j)
                elif col_j in col_i and col_i != col_j:
                    # col_j is substring of col_i, keep longer col name (col_i)
                    if len(col_i) > len(col_j):
                        # Rename col_j to col_i in df idx_j
                        dfs[idx_j].rename(columns={col_j: col_i}, inplace=True)
                        all_cols[j] = (idx_j, col_i)
    return dfs

def clean_selected_cells(df):
    """
    For every cell in the DataFrame, if it contains '\n:selected:' or '\n:unselected:',
    remove that part and keep only the content before it.
    """
    def clean_cell(cell):
        if isinstance(cell, str):
            for flag in ['\n:selected:', '\n:unselected:']:
                if flag in cell:
                    return cell.split(flag)[0]
        return cell

    # Apply clean_cell to every element in each column (works in all modern pandas)
    return df.apply(lambda col: col.map(clean_cell))

def merge_directional_tables(dfs):
    """
    Merge multiple directional survey tables where some tables have
    multiple original numeric columns collapsed into a single
    space-separated string column, or have auto-generated generic headers.

    Strategy:
      1. Choose a reference DF among the widest ones, preferring the one
         whose headers look most descriptive (not 'Column_#').
      2. For every other DF:
            - If same width & mostly auto headers -> just rename to reference headers.
            - If narrower -> attempt expansion by splitting whitespace tokens.
      3. Concatenate all, unify MD column values (e.g. 'MD (ft)' etc.),
         then drop duplicate MD depths (keeping first) and sort.
    """

    print(f"[INFO] Number of input DataFrames: {len(dfs)}")

    if not dfs:
        print("[WARN] No DataFrames provided.")
        return pd.DataFrame()

    # Filter out None / empty
    dfs = [d.copy() for d in dfs if d is not None and not d.empty]
    if not dfs:
        print("[WARN] All DataFrames empty after filtering.")
        return pd.DataFrame()

    print(f"[DEBUG] DataFrames after filtering empty ones: {len(dfs)}")

    # ---------- 1. Choose reference DF (widest + best header) ----------
    def header_score(cols):
        score = 0
        for c in cols:
            if re.search(r'[A-Za-z]', c) and not re.fullmatch(r'Column_\d+', c):
                score += 1
        return score

    max_cols = max(d.shape[1] for d in dfs)
    widest = [d for d in dfs if d.shape[1] == max_cols]
    ref_df = max(widest, key=lambda d: header_score(d.columns))
    ref_cols = list(ref_df.columns)
    print(f"[INFO] Reference DF selected with {len(ref_cols)} columns (score={header_score(ref_cols)}).")

    alpha_token_re = re.compile(r'[A-Za-z]')

    def row_tokens(row):
        toks = []
        for cell in row:
            if pd.isna(cell):
                continue
            parts = str(cell).strip().split()
            parts = [p for p in parts if p != ""]
            toks.extend(parts)
        return toks

    def looks_like_header_token_list(tokens):
        return any(alpha_token_re.search(tok) for tok in tokens)

    def expand_df(df):
        # Reference passes through
        if df is ref_df:
            print("[DEBUG] expand_df: reference DF passthrough.")
            return df

        # Same width as reference
        if df.shape[1] == len(ref_cols):
            auto_like = sum(bool(re.fullmatch(r'Column_\d+', c)) for c in df.columns)
            if auto_like >= df.shape[1] // 2:
                print("[DEBUG] Renaming same-width auto/generic columns to reference header.")
                df2 = df.copy()
                df2.columns = ref_cols
                return df2
            print("[DEBUG] DF already matches reference width with non-generic headers.")
            return df

        # Attempt expansion (narrower DF)
        expanded_rows = []
        for r_idx, (_, row) in enumerate(df.iterrows()):
            toks = row_tokens(row)
            if looks_like_header_token_list(toks):
                # skip embedded header-ish line
                continue
            if len(toks) == len(ref_cols):
                expanded_rows.append(toks)
            else:
                print(f"[WARNING] Row {r_idx} token length {len(toks)} != reference {len(ref_cols)}. Aborting expansion for this DF.")
                return None  # Cannot safely expand

        if not expanded_rows:
            print("[WARNING] No expandable rows found; skipping DF.")
            return None

        new_df = pd.DataFrame(expanded_rows, columns=ref_cols)
        print(f"[DEBUG] Successfully expanded DF to shape {new_df.shape}")

        # Try numeric conversion
        for c in new_df.columns:
            new_df[c] = pd.to_numeric(new_df[c], errors='ignore')
        return new_df
    clean_cell_values_in_dfs(dfs)
    expanded = []
    for idx, d in enumerate(dfs):
        print(f"[INFO] Processing DF {idx + 1}/{len(dfs)} (shape={d.shape}) ...")
        ed = expand_df(d)
        if ed is not None:
            # Reset index to ensure uniqueness
            ed = ed.reset_index(drop=True)
            # Check for duplicate indices after reset
            if ed.index.duplicated().any():
                print(f"[WARNING] Duplicated indices found in DF {idx + 1}.")
            expanded.append(ed)
        else:
            print(f"[INFO] Skipped DF {idx + 1} (could not align).")

    if not expanded:
        print("[ERROR] No DataFrames could be merged.")
        return pd.DataFrame(columns=ref_cols)

    # Ensure unique column names before concatenation
    for idx, df in enumerate(expanded):
        df.columns = [f"{col}_{idx}" if df.columns.tolist().count(col) > 1 else col for col in df.columns]

    for i,df in enumerate(expanded):
        print(f"==================================Table {i}:          {df.shape}")
        # display(df)
    for idx, df in enumerate(expanded):
        print(f"Index of DataFrame {idx}: {df.index}")


    # Concatenate DataFrames
    merged = pd.concat(expanded, ignore_index=True)

    print(f"[INFO] Concatenated merged DF shape (pre-clean): {merged.shape}")

    # ---------- 2. Unify MD column ----------
    md_col = ref_cols[0]
    # Find any other MD-like columns
    alt_md_candidates = [c for c in merged.columns
                         if c != md_col and re.match(r'^MD\b', str(c), flags=re.IGNORECASE)]

    if md_col not in merged.columns:
        print("[ERROR] Reference MD column missing in merged result.")
        return merged.reset_index(drop=True)

    # Coerce md_col to numeric (if original header was generic it will still work)
    merged[md_col] = pd.to_numeric(merged[md_col], errors='coerce')

    for alt in alt_md_candidates:
        before_nulls = merged[md_col].isna().sum()
        merged[alt] = pd.to_numeric(merged[alt], errors='coerce')
        merged.loc[merged[md_col].isna(), md_col] = merged.loc[merged[md_col].isna(), alt]
        after_nulls = merged[md_col].isna().sum()
        if after_nulls < before_nulls:
            print(f"[DEBUG] Filled {before_nulls - after_nulls} MD values from '{alt}'.")
    # md_col = clean_cell_value(md_col)
    merged = merged.sort_values(md_col, kind='mergesort')
    print(f"[INFO] Final merged DF shape: {merged.shape}")

    return merged.reset_index(drop=True)

def drop_mostly_nan_columns(df_list, threshold=0.2):
    cleaned_dfs = []
    
    # Check if df_list is a list of DataFrames or a single DataFrame
    if isinstance(df_list, list):
        for df in df_list:
            print(type(df))  # Debugging line
            data_only = df.iloc[1:]  # Exclude potential header row
            nan_ratio = data_only.isna().mean()
            cleaned_df = df.loc[:, nan_ratio < threshold]
            cleaned_dfs.append(cleaned_df)
        return cleaned_dfs
    else:
        # If a single DataFrame is passed, handle it here    for df in f_df:
        # display(df)
        data_only = df_list.iloc[1:]  # Exclude potential header row
        nan_ratio = data_only.isna().mean()
        cleaned_df = df_list.loc[:, nan_ratio < threshold]
        return cleaned_df

def make_columns_unique(df):
    # If df is a list of DataFrames
    if isinstance(df, list):
        cleaned_dfs = []
        for single_df in df:
            if single_df is None or not hasattr(single_df, "columns"):
                print("Warning: make_columns_unique received invalid DataFrame, returning empty DataFrame.")
                cleaned_dfs.append(pd.DataFrame())  # Append an empty DataFrame
            else:
                cols = pd.Series(single_df.columns)
                for dup in cols[cols.duplicated()].unique():  # For each duplicate column
                    cols[cols[cols == dup].index.values.tolist()] = [
                        dup + f"_B" if i != 0 else dup for i in range(sum(cols == dup))
                    ]
                single_df.columns = cols
                cleaned_dfs.append(single_df)
        return cleaned_dfs

    # If a single DataFrame is passed
    else:
        if df is None or not hasattr(df, "columns"):
            print("Warning: make_columns_unique received invalid DataFrame, returning empty DataFrame.")
            return pd.DataFrame()
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():  # For each duplicate column
            cols[cols[cols == dup].index.values.tolist()] = [
                dup + f"_B" if i != 0 else dup for i in range(sum(cols == dup))
            ]
        df.columns = cols
        return df


def is_date_like(s):
    return bool(re.match(r'^[A-Z][a-z]{2} \d{4}$', str(s).strip()))

def is_time_like(s):
    return bool(re.match(r'^\d{2}:\d{2}$', str(s).strip()))

def is_column_like(s):
    return str(s).startswith("Column_")

def move_and_remove_headers(df):
    df = df.copy()
    new_cols = list(df.columns)
    for idx, col in enumerate(df.columns):
        first_cell = df.iloc[0, idx] if df.shape[0] > 0 else None
        if (is_date_like(col) or is_time_like(col)) and (pd.isna(first_cell) or first_cell in [None, '']):
            # Move header into first cell, blank out header
            df.iloc[0, idx] = col
            new_cols[idx] = ""
        elif is_column_like(col) and (pd.isna(first_cell) or first_cell in [None, '']):
            # Just blank out the header
            new_cols[idx] = ""
    df.columns = new_cols
    return df

def merge_special_cols_to_prev(df):
    df = df.copy()
    all_cols = list(df.columns)
    # Find all special column indexes
    special_col_idxs = [
        idx for idx, col in enumerate(all_cols)
        if is_date_like(col) or is_time_like(col) or is_column_like(col) or col == ""
    ]
    if not special_col_idxs:
        return df

    target_col_idx = special_col_idxs[0] - 1
    if target_col_idx < 0:
        # No valid column to merge into
        return df

    target_col = all_cols[target_col_idx]
    # For each row, collect non-empty data from target col and all special cols
    def combine_row(row):
        values = []
        # Always add value from target column
        v = row.iloc[target_col_idx]
        if pd.notna(v) and str(v).strip() not in ['', 'nan', 'None']:
            values.append(str(v))
        # Add values from all special columns
        for idx in special_col_idxs:
            v = row.iloc[idx]
            if pd.notna(v) and str(v).strip() not in ['', 'nan', 'None']:
                values.append(str(v))
        return ', '.join(values)

    df[target_col] = df.apply(combine_row, axis=1)
    # Drop the special columns by index
    df = df.drop(df.columns[special_col_idxs], axis=1)
    return df

def create_new_columns(df):
    # Define the substrings for each new column mapping
    tvd_substrings = ['TVD', 'TVD Ft','True Vertical Depth (ft)','True Vertical Depth']
    northing_substrings = ['NS', 'N/-S', 'Northing', 'N/S','+N/-S FT','N[+]/S[-]','N-S ft', 'North (ft)', 'North [ft]', 'N[]']
    easting_substrings = ['EW', 'E/-W', 'Easting', 'E/W', '+E/-W FT', 'E[+]/W[-]','E-W ft','East (ft)', 'East [ft]','E[]']
    
    def find_column(cols, substrings):
        for sub in substrings:
            for col in cols:
                if sub in col:
                    return col
        return None
    
    # Check if df is a list of DataFrames or a single DataFrame
    if isinstance(df, list):
        cleaned_dfs = []
        for single_df in df:
            if single_df is None or not hasattr(single_df, "columns"):
                print("Warning: create_new_columns received invalid DataFrame, returning empty DataFrame.")
                cleaned_dfs.append(pd.DataFrame())  # Append an empty DataFrame
            else:
                # Find matching columns in the single DataFrame
                tvd_col = find_column(single_df.columns, tvd_substrings)
                northing_col = find_column(single_df.columns, northing_substrings)
                easting_col = find_column(single_df.columns, easting_substrings)
                
                # Create new columns using the found columns or fill with NaN
                single_df['TVD_'] = single_df[tvd_col] if tvd_col else np.nan
                single_df['Northing_'] = single_df[northing_col] if northing_col else np.nan
                single_df['Easting_'] = single_df[easting_col] if easting_col else np.nan
                
                cleaned_dfs.append(single_df)
        return cleaned_dfs
    else:
        # Handle single DataFrame
        if df is None or not hasattr(df, "columns"):
            print("Warning: create_new_columns received invalid DataFrame, returning empty DataFrame.")
            return pd.DataFrame()

        # Find matching columns in the DataFrame
        tvd_col = find_column(df.columns, tvd_substrings)
        northing_col = find_column(df.columns, northing_substrings)
        easting_col = find_column(df.columns, easting_substrings)
        
        # Create new columns using the found columns or fill with NaN
        df['TVD_'] = df[tvd_col] if tvd_col else np.nan
        df['Northing_'] = df[northing_col] if northing_col else np.nan
        df['Easting_'] = df[easting_col] if easting_col else np.nan
        
        return df


def remove_small_dfs(dfs):
    # Filter the DataFrames to remove those with 4 or fewer columns
    return [df for df in dfs if df.shape[1] > 3]

def clean_cell_values_in_dfs(dfs):
    def clean_cell_value(value):
        if isinstance(value, str):  # Check if the value is a string
            return value.replace(",", "")  # Remove commas
        return value  # If it's not a string, return the value as is

    # Iterate over each DataFrame and update it in place
    for i in range(len(dfs)):
        dfs[i] = dfs[i].applymap(clean_cell_value)  # Clean each cell and update the DataFrame in place

def remove_extra_text(dfs):
    # If a single DataFrame is passed, wrap it into a list for uniform processing
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    
    # Process each DataFrame
    updated_dfs = []
    for df in dfs:
        # Create a copy of the DataFrame to avoid modifying the original
        updated_df = df.copy()
        
        # Iterate through the rows (excluding the first and last)
        rows_to_remove = []
        for idx, row in updated_df.iloc[1:].iterrows():  # Skipping first and last rows
            # Count the number of None (NaN) values in the row
            none_count = row.isna().sum()
            # Check if the row has more than half NaN values
            if none_count >= len(row) / 2:
                rows_to_remove.append(idx)
        
        # Drop the identified rows
        updated_df = updated_df.drop(rows_to_remove)
        updated_dfs.append(updated_df)
    
    # If only one DataFrame was passed, return it directly instead of a list
    if len(updated_dfs) == 1:
        return updated_dfs[0]
    else:
        return updated_dfs


def clean_last_row_numbers(dfs):
    """
    Cleans the last row of each DataFrame (or a single DataFrame).
    Extracts all numeric values from each cell in the last row.
    If one or more numbers are found, stores them as a comma-separated string.
    Accepts a single DataFrame or a list of them.
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]  # Wrap single DataFrame in a list

    for df in dfs:
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        last_row_idx = df.index[-1]

        for col in df.columns:
            val = df.at[last_row_idx, col]
            if isinstance(val, str):
                val = val.replace('\n', ' ')  # Normalize newlines
                matches = re.findall(r'[-+]?\d*\.\d+|\d+', val)
                if matches:
                    # Join all found numbers into a string, preserving decimals
                    cleaned = ', '.join(matches)
                    df.at[last_row_idx, col] = cleaned

    return dfs if len(dfs) > 1 else dfs[0]


def split_numeric_columns(dfs, columns=None, prefix=None):
    """
    Splits comma-separated numeric values in specified columns into separate new columns.
    Only the split columns are renamed (e.g., 'A_1', 'A_2'); others remain unchanged.

    Parameters:
    - dfs: A single DataFrame or a list of DataFrames.
    - columns: List of column names to split (default: all columns).
    - prefix: Optional prefix for new columns (default: original column name).

    Returns:
    - A single DataFrame or a list of DataFrames with split columns.
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    result_dfs = []

    for df in dfs:
        if not isinstance(df, pd.DataFrame) or df.empty:
            result_dfs.append(df)
            continue

        df_copy = df.copy()
        cols_to_process = columns or df_copy.columns

        for col in cols_to_process:
            if col not in df_copy.columns:
                continue

            # Check if any cell has a comma — only then proceed to split
            if not df_copy[col].astype(str).str.contains(',').any():
                continue

            # Perform the split
            split_data = df_copy[col].astype(str).str.split(r'\s*,\s*', expand=True)

            # Create new column names for the split parts
            new_col_names = [
                f"{prefix or col}_{i+1}" for i in range(split_data.shape[1])
            ]
            split_data.columns = new_col_names


            # Insert split columns at the position of the original column
            insert_at = df_copy.columns.get_loc(col)
            df_copy = df_copy.drop(columns=[col])
            for i, new_col in enumerate(split_data.columns):
                df_copy.insert(insert_at + i, new_col, split_data[new_col])

        result_dfs.append(df_copy)

    return result_dfs if len(result_dfs) > 1 else result_dfs[0]




def remove_mostly_zero_rows(dfs, zero_tolerance=2):
    """
    Removes rows where number of non-zero columns is <= zero_tolerance.
    E.g., zero_tolerance=1 means: remove rows with only 0 or 1 non-zero values.
    Accepts a single DataFrame or a list of DataFrames.
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    cleaned_dfs = []
    for df in dfs:
        if not isinstance(df, pd.DataFrame) or df.empty:
            cleaned_dfs.append(df)
            continue

        numeric_df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        non_zero_counts = (numeric_df != 0).sum(axis=1)
        mask = non_zero_counts > zero_tolerance - 1  # keep rows with >= tolerance
        cleaned_df = df[mask].copy()

        cleaned_dfs.append(cleaned_df)

    return cleaned_dfs if len(cleaned_dfs) > 1 else cleaned_dfs[0]

def process_pdf(pdf_bytes: bytes, _hash: str):

    raw_tables= extract_tables_from_pdf_with_retry(pdf_bytes,max_retries)

    clean = []
    for i,df in enumerate(raw_tables):
        if df is not None:
            df = df.replace(r'^\s*$', pd.NA, regex=True).dropna(how='all', axis=0).dropna(how='all', axis=1).reset_index(drop=True)
            df.columns = [f'{i}' for i in range(df.shape[1])]  # Reset the column indices to default names
            print(f"+++++++++++++++++++++++++++++++++++++++++++++ Table {i} ===========================================")
            # display(df.head(10)) 
            print(df.shape)
            clean.append(df)

    for df in clean:
        df = replace_unwanted_values(df)

    normalized_dfs = []
    clean_cell_values_in_dfs(clean)
    for i,df in enumerate(clean):
        df = normalize_table_by_column_first_cell(df)
        print(f"================================================= Table {i} ===========================================")
        print(df.shape)
        normalized_dfs.append(df)
    
    filtered_dfs = remove_small_dfs(normalized_dfs)
    clean_dfs = filter_dfs_by_header_keywords(filtered_dfs)
    df1 = merge_two_line_headers(clean_dfs)
    df1 = merge_two_line_headers(df1)
    cleared_content = clean_bracket_content(df1)
    df2 = merge_dataframes_with_same_structure(cleared_content)
    df3 = split_multi_numeric_columns(df2)
    df4 = propagate_multi_values_to_next_column(df3)
    df5 = group_and_merge_tables_dynamic(df4)
    dfs555 = concat_empty_colnames_to_next_list(df5)
    dfs555 = merge_dataframes_with_same_structure(dfs555)
    dfs555 = [make_cols_unique(df) for df in dfs555]
    df5_temp=merge_adjacent_columns(dfs555)
    df55_temp = merge_two_line_headers(df5_temp)
    df51_temp = unify_column_names_by_substring(df5_temp)
    df55_temp = merge_two_line_headers(df51_temp)
    df5 = unify_column_names_by_substring(df55_temp)

    combined_cols =[]

    for df in df5:
        df = move_and_remove_headers(df)
        special_cols = merge_special_cols_to_prev(df)
        combined_cols.append(special_cols)

    for i,df in enumerate(combined_cols):
        df = df.reset_index(drop=True)

    dfinal= merge_dataframes_with_same_structure(combined_cols)

    if (len(dfinal) > 1):
        merged_table = merge_directional_tables(dfinal)
    elif(len(dfinal)==1):
        merged_table = dfinal
    else:
        merged_table = None
    final_df = drop_mostly_nan_columns(merged_table)
    df = make_columns_unique(final_df)
    
    f_df = create_new_columns(df)
    f_df = remove_extra_text(f_df)
    f_df = clean_last_row_numbers(f_df)
    f_df = remove_mostly_zero_rows(f_df)
    f_df = split_numeric_columns(f_df)
    print(f_df)
    return f_df


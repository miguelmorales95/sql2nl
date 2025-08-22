import re
import sqlparse

def _normalize_ws(s: str) -> str:
    return " ".join(s.strip().split())

def _find_tables(parsed):
    # crude table finder: looks for FROM and JOIN identifiers
    tables = []
    tokens = [t for t in parsed.tokens if not t.is_whitespace]
    text = parsed.value
    for m in re.finditer(r"\bfrom\s+([a-zA-Z0-9_.\"]+)", text, re.IGNORECASE):
        tables.append(m.group(1).strip('"'))
    for m in re.finditer(r"\bjoin\s+([a-zA-Z0-9_.\"]+)", text, re.IGNORECASE):
        tables.append(m.group(1).strip('"'))
    return list(dict.fromkeys(tables))

def _find_select_list(sql: str):
    m = re.search(r"select\s+(.*?)\s+from\b", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    cols = _normalize_ws(m.group(1))
    return cols

def _has_group_by(sql): return bool(re.search(r"\bgroup\s+by\b", sql, re.IGNORECASE))
def _has_order_by(sql): return bool(re.search(r"\border\s+by\b", sql, re.IGNORECASE))
def _has_limit(sql): return bool(re.search(r"\blimit\s+\d+", sql, re.IGNORECASE))
def _has_qualify(sql): return bool(re.search(r"\bqualify\b", sql, re.IGNORECASE))
def _has_window(sql): return bool(re.search(r"\bover\s*\(", sql, re.IGNORECASE))
def _has_where(sql): return bool(re.search(r"\bwhere\b", sql, re.IGNORECASE))
def _has_spectrum(sql): return bool(re.search(r"\bexternal\s+schema\b|\bspectrum\b", sql, re.IGNORECASE))
def _has_system_tables(sql): return bool(re.search(r"\bsvv_|\\bstl_|\\bpg_", sql, re.IGNORECASE))

def explain_sql(sql: str) -> str:
    """
    Lightweight heuristic explainer. Best-effort extraction of:
    - selected columns
    - source tables
    - where/group/order/limit
    - window/qualify
    - Redshift-specific hints
    """
    if not sql or not sql.strip():
        return "Empty SQL."
    sql_clean = _normalize_ws(sql)
    try:
        parsed = sqlparse.parse(sql)[0]
    except Exception:
        parsed = sql

    tables = _find_tables(parsed if hasattr(parsed, "tokens") else sql)
    cols = _find_select_list(sql)

    bits = []
    if tables:
        bits.append(f"Reads from {', '.join(tables)}.")
    else:
        bits.append("Reads from an unspecified table or source.")

    if cols:
        # compress SELECT * into friendlier phrasing
        if re.fullmatch(r"\*", cols):
            bits.append("Selects all columns.")
        else:
            # keep short if very long
            short_cols = cols
            if len(short_cols) > 120:
                short_cols = short_cols[:117] + "..."
            bits.append(f"Selects: {short_cols}.")

    if _has_where(sql_clean):
        bits.append("Filters rows with a WHERE clause.")
    if _has_group_by(sql_clean):
        bits.append("Aggregates using GROUP BY.")
    if _has_window(sql_clean):
        bits.append("Computes window functions (OVER ...).")
    if _has_qualify(sql_clean):
        bits.append("Applies QUALIFY to filter by window results (Redshift).")
    if _has_order_by(sql_clean):
        bits.append("Orders the result with ORDER BY.")
    if _has_limit(sql_clean):
        m = re.search(r"\blimit\s+(\d+)", sql_clean, re.IGNORECASE)
        if m:
            bits.append(f"Limits output to {m.group(1)} rows.")
    if _has_spectrum(sql_clean):
        bits.append("References Redshift Spectrum (external data).")
    if _has_system_tables(sql_clean):
        bits.append("Queries Redshift system tables (SVV_/STL_/PG_).")

    # Try to turn into one or two sentences
    if not bits:
        return "Describes a Redshift SQL query."
    if len(bits) == 1:
        return bits[0]
    # First sentence: gist
    first = bits[0]
    rest = " ".join(bits[1:])
    return f"{first} {rest}"

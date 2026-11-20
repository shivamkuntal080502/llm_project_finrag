"""Rule-based parser for financial-statement tables (markdown from Docling).
Works for US 10-K and IFRS / US-GAAP 20-F tables (Infosys, Wipro, HDFC Bank, ...).
Deterministic: finds the fiscal-year columns in the header, maps row labels to fields,
returns {year: {field: value}}. The LLM is only a fallback for fields this cannot find."""
import re

# field -> ordered list of (regex on normalised row label, needs_to_be_after_'total current liabilities')
PATTERNS = {
    "income": {
        "revenue": [(r"^(total (net )?(sales|revenues?)|(net )?revenues?|net sales|total revenue, net)$", False)],
        "gross_profit": [(r"^(total )?gross (margin|profit)$", False)],
        "operating_income": [(r"^(total )?operating (income|profit)|^income from operations|"
                              r"^results from operating activities", False)],
        "net_income": [(r"^net (income|profit|earnings) attributable to", False),   # prefer shareholders' share
                       (r"^equity holders of the company$", False),
                       (r"^net (income|profit|earnings)\b", False),
                       (r"^profit for the (year|period)$", False)],
        "interest_expense": [(r"^(interest expense|finance (costs?|expenses?))$", False)],
    },
    "balance": {
        "total_assets": [(r"^total assets$", False)],
        "current_assets": [(r"^total current assets$", False)],
        "current_liabilities": [(r"^total current liabilities$", False)],
        "total_liabilities": [(r"^total liabilities$", False)],
        "shareholders_equity": [(r"^total (shareholders|stockholders)' equity$|^total equity$", False)],
        "retained_earnings": [(r"^retained earnings|^accumulated deficit", False)],
        "cash": [(r"^cash and cash equivalents$", False), (r"^cash and due from banks", False)],
        "long_term_debt": [(r"^(long-term debt|long term debt|long-term borrowings|loans and borrowings|"
                            r"non-current borrowings|debt and finance leases, net of current portion)$", False),
                           (r"^term debt$", True)],           # Apple: 2nd 'Term debt' row is the non-current one
    },
    "cashflow": {
        "depreciation_amortization": [(r"^depreciation", False)],
        "operating_cash_flow": [(r"^(cash generated (by|from)|net cash)[a-z /\-()'’]* "
                                 r"(operating activities|operations)$", False)],
        "capex": [(r"^(payments? for (the )?(acquisition|purchase) of property|additions to property|"
                   r"purchases? of property|capital expenditures?|expenditure on property)", False)],
    },
}

_CURRENCY = re.compile(r"(?i)rs\.?|us\$|inr|usd|₹")


def _cell_num(c):
    c = _CURRENCY.sub("", c)
    c = c.replace("$", "").replace("\u00a0", "").replace(" ", "").replace("*", "")
    if not c or c in {"—", "-", "–"}:
        return None
    neg = c.startswith("(") or c.startswith("-")
    c = c.strip("()-")
    if not re.fullmatch(r"\d[\d,]*(\.\d+)?", c):
        return None
    v = float(c.replace(",", ""))
    return -v if neg else v


def _rows(md):
    rows = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.fullmatch(r"[|\-:\s]+", line):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def _norm(label):
    label = label.lower().replace("’", "'").replace("\u2019", "'").strip(" :")
    return re.sub(r"\s+", " ", label)


def parse_table(md, statement, default_year=None):
    """Return ({year: {field: value}}, [fiscal years found]).
    default_year: filing year, used only if the table has no year header (assumes newest column first)."""
    spec = {f: [(re.compile(p), after) for p, after in lst] for f, lst in PATTERNS[statement].items()}

    header_rows, data_rows = [], []
    for cells in _rows(md):
        text = " ".join(cells)
        found = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
        is_header = bool(found) and not re.search(r"\d{1,3},\d{3}", text)
        if is_header and not data_rows:
            header_rows.append(found)
        elif any(_cell_num(c) is not None for c in cells[1:]):          # real data row
            data_rows.append(cells)                                      # (title rows w/o numbers skipped)

    rows = []
    for cells in data_rows:
        vals = [v for v in (_cell_num(c) for c in cells[1:]) if v is not None]
        rows.append((_norm(cells[0]), vals))

    years_all = max(header_rows, key=len) if header_rows else []         # e.g. [2024, 2025, 2026, 2026]
    if not years_all:                                                    # header missing in the table
        if not default_year or not rows:
            return {}, []
        lens = [len(v) for _, v in rows if len(v) >= 2]
        if not lens:
            return {}, []
        n = max(set(lens), key=lens.count)
        years_all = [default_year - i for i in range(n)]
    uniq = list(dict.fromkeys(years_all))
    n_all, n_uniq = len(years_all), len(uniq)

    tcl_idx = next((i for i, (l, _) in enumerate(rows) if l == "total current liabilities"), None)

    def assign(vals):
        if len(vals) >= n_all:                       # takes last n values: skips a leading 'Note' column
            ys, use = years_all, vals[-n_all:]
        elif len(vals) >= n_uniq:
            ys, use = uniq, vals[-n_uniq:]
        else:
            return None
        d = {}
        for y, v in zip(ys, use):
            d.setdefault(y, v)                       # first occurrence wins (local currency before US$ column)
        return d

    out = {}
    for field, plist in spec.items():
        for pat, after_tcl in plist:
            hit = None
            for i, (label, vals) in enumerate(rows):
                if after_tcl and (tcl_idx is None or i <= tcl_idx):
                    continue
                if pat.search(label):
                    hit = assign(vals)
                    if hit:
                        break
            if hit:
                for y, v in hit.items():
                    out.setdefault(y, {})[field] = v
                break
    return out, uniq
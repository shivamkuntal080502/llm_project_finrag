"""Question -> parse -> statement-wise table extraction -> checks -> metrics -> grounded report."""
import json, re
import config, llm
from retriever import Retriever
from metrics import compute, parse_json, forecast, benchmark, checks, facts
from analysis import check_numbers
from table_parser import parse_table

_r = None

STATEMENTS = {
    "income": {
        "fields": ["revenue", "gross_profit", "operating_income", "net_income", "interest_expense"],
        "required": ["revenue", "net_income"],        # banks have no operating income -> optional
        "keys": ["total net sales", "net sales", "revenue", "gross margin", "gross profit",
                 "operating income", "operating profit", "income from operations", "net income",
                 "net profit", "profit for the year", "finance cost", "earnings per",
                 "total revenue, net", "results from operating activities",
                 "net income attributable to"],
        "n_filings": 1},
    "balance": {
        "fields": ["total_assets", "total_liabilities", "current_assets", "current_liabilities",
                   "long_term_debt", "shareholders_equity", "retained_earnings", "cash"],
        "required": ["total_assets", "total_liabilities", "shareholders_equity"],   # banks: no current items
        "keys": ["total assets", "total current assets", "total liabilities", "total current liabilities",
                 "shareholders' equity", "stockholders' equity", "total equity",
                 "retained earnings", "accumulated deficit", "cash and due from banks"],
        "n_filings": 2},
    "cashflow": {
        "fields": ["depreciation_amortization", "operating_cash_flow", "capex"],
        "required": ["depreciation_amortization", "operating_cash_flow", "capex"],
        "keys": ["operating activities", "investing activities", "financing activities",
                 "net cash from operations", "net cash used in financing", "net cash used in investing",
                 "depreciation and amortization", "depreciation, amortization",
                 "payments for acquisition of property", "additions to property and equipment",
                 "purchases of property", "capital expenditures", "payment for purchase of property",
                 "expenditure on property, plant and equipment"],
        "n_filings": 1},
}


def extract_prompt(fields):
    return ("Extract values from the financial statement table(s) of an annual report (10-K or 20-F). "
            'Reply with ONLY valid JSON: {"<fiscal year>": {'
            + ", ".join(f'"{f}": number|null' for f in fields) + "}}. "
            "Values in millions, one entry per fiscal year column in the table. "
            "'revenue' = total net sales / revenues (NOT gross margin). "
            "'gross_profit' = gross margin / gross profit in money. "
            "'operating_income' = operating income or operating profit (NOT gross profit). "
            "'net_income' = net income, net profit or profit for the year. "
            "'interest_expense' = interest expense or finance cost. "
            "'shareholders_equity' = total shareholders' equity or total equity. "
            "'long_term_debt' = non-current borrowings / term debt. 'capex' as a positive number. "
            "Use null if not shown. Never guess.")


REPORT_SYS = ("You are a financial analyst. Use ONLY the FACTS and METRICS provided. "
              "Never invent causes, products, events or risks. Never speculate why numbers changed. "
              "Do not use words like 'suggesting', 'indicating', 'efficiency' or 'recovery'. "
              "Describe a trend only if the FACTS contain a trend line for it, using 'rose'/'fell' exactly as written. "
              "If a metric or year is missing, say it was not available. Max 250 words. "
              "Sections: Summary, Key trends, Comparison (only if several companies), "
              "Takeaways (restate what the numbers show, no advice about products or strategy). "
              "Amounts are in each company's own currency (see UNITS): never compare absolute amounts across "
              "different currencies, compare ratios only. If the user asks whether to invest or buy, say you "
              "cannot give investment advice and only summarise the data.")


def get_retriever():
    global _r
    _r = _r or Retriever()
    return _r


def _fmt(chunks, limit):
    return "\n\n".join(f"[{c['source']} | FY{c['year']} | p.{c['page']}]\n{c['text'][:limit]}" for c in chunks)


def extract(r, company, years):
    """Per statement: rule-based table parser first, LLM only fills fields the parser could not find."""
    merged, used = {}, []
    for name, spec in STATEMENTS.items():
        tabs = r.find_statement(company, years, spec["keys"], spec["n_filings"])
        if not tabs:
            print(f"[debug] {company}: no {name} table found")
            continue
        used += tabs

        # 1) rules (latest filing first, so newest numbers win)
        for t in tabs:
            parsed, ys = parse_table(t["text"], name, t.get("year"))
            print(f"[parse] {company} {name} {t['source']} p.{t['page']}: years={ys} "
                  f"fields={sorted({f for v in parsed.values() for f in v})}")
            for y, vals in parsed.items():
                for f, v in vals.items():
                    merged.setdefault(str(y), {}).setdefault(f, v)

        # 2) LLM fallback only for fields still missing
        have = {f for v in merged.values() for f in v}
        missing = [f for f in spec["required"] if f not in have]   # optional fields are never guessed
        if missing:
            print(f"[llm]   {company} {name}: asking LLM for {missing}")
            txt = llm.chat(extract_prompt(missing),
                           "Table(s):\n" + _fmt(tabs, config.MAX_TABLE_CHARS), 500)
            out = parse_json(txt)
            if not out:
                print(f"[debug] {company}: {name} LLM gave no valid JSON. Raw output:\n{txt[:300]}")
            for y, vals in out.items():
                m = re.search(r"\d{4}", str(y))
                if m and isinstance(vals, dict):
                    for k, v in vals.items():
                        if v is not None and k in missing:
                            merged.setdefault(m.group(), {}).setdefault(k, v)
    return merged, used


def run(question, memory=None, progress=None):
    say = progress or (lambda s: None)
    say("Loading the knowledge base...")
    r = get_retriever()
    companies, years, last_n = r.parse_query(question, memory)
    if not companies:
        return {"error": f"Mention a company. Available: {', '.join(config.display(c) for c in r.companies)}"}

    dfs, fc, text_chunks, tables, raws, warns = {}, {}, [], [], {}, {}
    for c in companies:
        ys = r.resolve_years(c, years, last_n)
        say(f"Reading financial statements: {config.display(c)} ({min(ys)}-{max(ys)})...")
        raw, tabs = extract(r, c, ys)
        raws[c] = raw
        warns[c] = checks(raw)
        df = compute(raw)
        dfs[c] = df
        if not df.empty:
            fc[c] = forecast(df)
        say(f"Calculating ratios for {config.display(c)}...")
        text_chunks += r.search(question, c, ys, top_k=3)
        tables += tabs

    ok = {c: d for c, d in dfs.items() if not d.empty}
    bench = benchmark(ok)
    src = sorted({(t["source"], t["page"]) for t in tables})
    payload = {c: json.loads(d.to_json(orient="index")) for c, d in ok.items()}
    fx = facts(ok) or "No multi-year trend is available (only one year of data was extracted)."
    units = "; ".join(f"{c}: {config.currency(c)} millions" for c in ok)
    prompt = (f"Question: {question}\n\nUNITS: {units}\n\nFACTS:\n{fx}\n\n"
              f"METRICS (amounts in millions / ratios):\n{json.dumps(payload)}\n\n"
              f"Sources: " + ", ".join(f"[{s} p.{p}]" for s, p in src))
    say("Writing the report...")
    report = llm.chat(REPORT_SYS, prompt, 700)
    flags = check_numbers(report, ok, tables + [{"text": fx}])

    return {"companies": companies, "years": years, "metrics": dfs, "forecast": fc,
            "benchmark": bench, "report": report, "flags": flags, "raw": raws, "warnings": warns,
            "chunks": text_chunks, "tables": tables,
            "sources": sorted({(t["source"], t["page"]) for t in tables}),
            "units": {c: config.currency(c) for c in companies},
            "memory": {"companies": companies, "years": years, "last_n": last_n}}


if __name__ == "__main__":
    import sys
    res = run(" ".join(sys.argv[1:]) or "Give financial report of Apple in last 3 years")
    if "error" in res:
        print(res["error"])
    else:
        for c, d in res["metrics"].items():
            print(f"\n== {c} ==\n{d.T}")
            for w in res["warnings"][c]:
                print("  WARNING:", w)
        print("\n" + res["report"])
        print("\nUnverified numbers:", res["flags"])
        print("Sources:", res["sources"])
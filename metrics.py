"""Python engine: ratios, forecast, peer benchmark. LLM never does maths."""
import json, re
import numpy as np
import pandas as pd

FIELDS = ["revenue", "gross_profit", "operating_income", "net_income", "depreciation_amortization",
          "interest_expense", "operating_cash_flow", "capex",
          "total_assets", "total_liabilities", "current_assets", "current_liabilities",
          "long_term_debt", "shareholders_equity", "retained_earnings", "cash"]


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _num(v):
    if isinstance(v, str):
        neg = "(" in v or v.strip().startswith("-")
        v = re.sub(r"[^\d.]", "", v)
        if not v:
            return np.nan
        v = float(v) * (-1 if neg else 1)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def clean_raw(raw: dict) -> pd.DataFrame:
    clean = {}
    for k, v in raw.items():
        m = re.search(r"\d{4}", str(k))
        if m and isinstance(v, dict):
            clean[int(m.group())] = {f: _num(v.get(f)) for f in FIELDS}
    if not clean:
        return pd.DataFrame()
    return pd.DataFrame(clean).T.reindex(columns=FIELDS).sort_index()


def checks(raw: dict) -> list:
    """Sanity checks on extracted numbers. Returns human-readable warnings."""
    df = clean_raw(raw)
    msgs = []
    for y, r in df.iterrows():
        if pd.notna(r.operating_income) and pd.notna(r.gross_profit) and r.operating_income > r.gross_profit:
            msgs.append(f"{y}: operating income > gross profit (values probably mixed up)")
        if pd.notna(r.net_income) and pd.notna(r.revenue) and r.net_income > r.revenue:
            msgs.append(f"{y}: net income > revenue")
        if all(pd.notna(x) for x in (r.total_assets, r.total_liabilities, r.shareholders_equity)):
            if abs(r.total_assets - r.total_liabilities - r.shareholders_equity) / abs(r.total_assets) > 0.02:
                msgs.append(f"{y}: assets != liabilities + equity (balance sheet values wrong)")
    return msgs


def compute(raw: dict) -> pd.DataFrame:
    df = clean_raw(raw)
    if df.empty:
        return df
    o = pd.DataFrame(index=df.index)
    o["Revenue"] = df.revenue
    o["Net income"] = df.net_income
    o["EBITDA"] = df.operating_income + df.depreciation_amortization
    o["Free cash flow"] = df.operating_cash_flow - df.capex.abs()
    o["Gross margin %"] = df.gross_profit / df.revenue * 100
    o["Operating margin %"] = df.operating_income / df.revenue * 100
    o["Net margin %"] = df.net_income / df.revenue * 100
    o["Revenue growth %"] = df.revenue.pct_change() * 100
    o["Current ratio"] = df.current_assets / df.current_liabilities
    o["Cash ratio"] = df.cash / df.current_liabilities
    o["Debt / equity"] = df.long_term_debt / df.shareholders_equity
    o["Interest coverage"] = df.operating_income / df.interest_expense.abs()
    o["ROE %"] = df.net_income / df.shareholders_equity * 100
    o["ROA %"] = df.net_income / df.total_assets * 100
    o["Asset turnover"] = df.revenue / df.total_assets
    wc = df.current_assets - df.current_liabilities       # Altman Z'' (non-manufacturing)
    o["Altman Z''"] = (6.56 * wc / df.total_assets + 3.26 * df.retained_earnings / df.total_assets
                       + 6.72 * df.operating_income / df.total_assets
                       + 1.05 * df.shareholders_equity / df.total_liabilities)
    return o.replace([np.inf, -np.inf], np.nan).dropna(how="all", axis=1).round(2)


def facts(dfs: dict) -> str:
    """Pre-computed trend sentences so the LLM never has to judge up/down itself."""
    lines = []
    for c, d in dfs.items():
        for col in ("Revenue", "Net income", "Operating margin %", "Net margin %"):
            if col not in d.columns:
                continue
            s = d[col].dropna()
            if len(s) < 2:
                continue
            is_pct = "%" in col
            parts = [f"{int(s.index[0])}: {s.iloc[0]:,.2f}"]
            for i in range(1, len(s)):
                word = "rose" if s.iloc[i] > s.iloc[i - 1] else "fell"
                if is_pct:                                   # margins: change in percentage points
                    chg = f"{abs(s.iloc[i] - s.iloc[i - 1]):.1f} points"
                else:
                    ch = (s.iloc[i] - s.iloc[i - 1]) / abs(s.iloc[i - 1]) * 100 if s.iloc[i - 1] else float("nan")
                    chg = f"{abs(ch):.1f}%"
                parts.append(f"{int(s.index[i])}: {s.iloc[i]:,.2f} ({word} {chg})")
            lines.append(f"{c} {col} -> " + "; ".join(parts))
    return "\n".join(lines)


def forecast(df, cols=("Revenue", "Net income", "EBITDA"), horizon=1):
    """Simple linear trend (needs >=2 years). Illustrative only."""
    rows = {}
    for h in range(1, horizon + 1):
        y_next = int(df.index.max()) + h
        rows[y_next] = {}
        for c in cols:
            if c not in df.columns:                      # column missing if extraction failed
                continue
            s = df[c].dropna()
            if len(s) >= 2:
                slope, icpt = np.polyfit(s.index.astype(float), s.values, 1)
                rows[y_next][c] = round(slope * y_next + icpt, 2)
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def benchmark(dfs: dict) -> pd.DataFrame:
    """Latest-year ratios of each company vs the average of the companies compared."""
    if len(dfs) < 2:
        return pd.DataFrame()
    dfs = {c: d for c, d in dfs.items() if not d.empty}
    if len(dfs) < 2:
        return pd.DataFrame()
    latest = pd.DataFrame({c: d.iloc[-1] for c, d in dfs.items() if not d.empty})
    ratios = [r for r in latest.index if "%" in r or "ratio" in r or "/" in r
              or r in ("Interest coverage", "Asset turnover", "Altman Z''")]
    t = latest.loc[ratios]
    t["Peer average"] = t.mean(axis=1).round(2)
    return t
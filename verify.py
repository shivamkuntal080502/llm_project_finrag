"""Compare extracted numbers with official data from yfinance."""
import pandas as pd
import config


class CurrencyMismatch(Exception):
    pass


def check_currency(company, ticker):
    """yfinance may report in another currency than the filing (e.g. Infosys: filing US$, Yahoo INR)."""
    try:
        cur = (ticker.info or {}).get("financialCurrency")
    except Exception:
        return
    want = config.currency(company)
    if cur and cur != want:
        raise CurrencyMismatch(f"Yahoo Finance reports {config.display(company)} in {cur}, but the filing is in "
                               f"{want}. Verify manually against the 20-F.")


def verify_company(company, metrics_df):
    import yfinance as yf
    tk = config.TICKERS.get(company)
    if not tk:
        return pd.DataFrame()
    t = yf.Ticker(tk)
    check_currency(company, t)
    fin = t.financials
    rows = []
    pairs = [("Revenue", ["Total Revenue", "Operating Revenue"]),
             ("Net income", ["Net Income", "Net Income Common Stockholders"])]
    for metric, names in pairs:
        name = next((n for n in names if n in fin.index), None)
        if not name or metric not in metrics_df.columns:
            continue
        for col in fin.columns:
            y = col.year
            if y in metrics_df.index and pd.notna(metrics_df.loc[y, metric]):
                off = fin.loc[name, col] / 1e6               # units -> millions
                ext = metrics_df.loc[y, metric]
                diff = abs(ext - off) / abs(off) * 100 if off else None
                rows.append({"Year": y, "Metric": metric, "Extracted (M)": ext,
                             "Official (M)": round(off, 1), "Diff %": round(diff, 2),
                             "Match": bool(diff is not None and diff <= 2)})
    return pd.DataFrame(rows)
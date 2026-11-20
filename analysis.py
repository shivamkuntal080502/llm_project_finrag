"""Risk summaries + hallucination guard."""
import re
import numpy as np
import llm

RISK_SYS = ("You summarise risk factors from a 10-K. Give at most 5 short bullet points, "
            "only from the excerpts. No numbers unless in the text.")


def risk_summary(r, company, years):
    out = {}
    ys = sorted(set([min(years), max(years)])) if years else []
    for y in ys:
        ch = r.search(f"{company} risk factors material adverse effect", company, [y], top_k=5)
        ctx = "\n\n".join(c["text"][:1200] for c in ch)
        out[y] = llm.chat(RISK_SYS, ctx, 400)
    return out


_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _nums(text):
    vals = []
    for m in _NUM.findall(text):
        try:
            vals.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return vals


def check_numbers(report, dfs, chunks):
    """Return numbers in the report that appear nowhere in metrics or retrieved text."""
    allowed = []
    for d in dfs.values():
        allowed += [float(x) for x in d.values.flatten() if x == x]
    for c in chunks:
        allowed += _nums(c["text"])
    arr = np.array(allowed) if allowed else np.array([0.0])
    report = re.sub(r"\[[^\]]*\]", " ", report)          # drop [file p.12] citations
    report = re.sub(r"\bp\.?\s*\d+", " ", report)         # drop page refs
    report = re.sub(r"(?m)^\s*\d+[.)]", " ", report)       # drop list numbering
    flags = []
    for n in _nums(report):
        if (n == int(n) and 1900 <= n <= 2100) or (n == int(n) and n < 10) or n == 0:
            continue                                  # years, list numbers
        tol = max(0.01 * abs(n), 0.05)
        if not np.any(np.abs(arr - n) <= tol):
            flags.append(n)
    return sorted(set(flags))

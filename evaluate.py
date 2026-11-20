"""
Evaluation for the paper.

python evaluate.py --build
    # make eval/testset.json from yfinance ground truth

python evaluate.py
    # retrieval hit-rate: vector vs hybrid vs hybrid+rerank

python evaluate.py --llm
    # + number-extraction accuracy (slow on CPU)
"""

import argparse
import json

import pandas as pd

import config
from retriever import Retriever
from verify import check_currency, CurrencyMismatch


TEST = config.EVAL_DIR / "testset.json"


def build():
    import yfinance as yf

    r = Retriever()
    qs = []

    for c in r.companies:
        tk = config.TICKERS.get(c)

        if not tk:
            continue

        # Verify currency before using yfinance financial data
        t = yf.Ticker(tk)

        try:
            check_currency(c, t)
        except CurrencyMismatch as ex:
            print("skip", c, "-", ex)
            continue

        fin = t.financials

        have = {
            x["year"]
            for x in r.chunks
            if x["company"] == c
        }

        for label, names in [
            (
                "Revenue",
                [
                    "Total Revenue",
                    "Operating Revenue"
                ]
            ),
            (
                "Net income",
                [
                    "Net Income",
                    "Net Income Common Stockholders"
                ]
            )
        ]:
            name = next(
                (n for n in names if n in fin.index),
                None
            )

            if not name:
                continue

            for col in fin.columns:
                if col.year in have:
                    qs.append(
                        {
                            "company": c,
                            "year": col.year,
                            "field": label,
                            "value_m": round(
                                float(fin.loc[name, col]) / 1e6
                            )
                        }
                    )

    config.EVAL_DIR.mkdir(exist_ok=True)

    TEST.write_text(
        json.dumps(qs, indent=1)
    )

    print(
        f"{len(qs)} questions saved to {TEST}"
    )


def retrieval(qs):
    r = Retriever()
    rows = []

    for mode in [
        "vector",
        "hybrid",
        "rerank"
    ]:
        hits = 0

        for q in qs:
            query = (
                f"{q['company']} "
                f"{q['year']} total "
                f"{q['field'].lower()}"
            )

            ch = r.search(
                query,
                q["company"],
                [q["year"]],
                top_k=config.TOP_K,
                mode=mode
            )

            needle = f"{q['value_m']:,}"

            hits += any(
                needle in c["text"]
                for c in ch
            )

        rows.append(
            {
                "Method": mode,
                "Hit rate %": round(
                    100 * hits / len(qs),
                    1
                )
            }
        )

    return pd.DataFrame(rows)


def extraction(qs):
    from pipeline import extract, get_retriever
    from metrics import compute

    r = get_retriever()

    cache = {}
    ok = 0

    for q in qs:
        key = (
            q["company"],
            q["year"]
        )

        if key not in cache:
            raw, _ = extract(
                r,
                q["company"],
                [q["year"]]
            )

            cache[key] = compute(raw)

        df = cache[key]

        if (
            q["year"] in df.index
            and df.loc[
                q["year"],
                q["field"]
            ] == df.loc[
                q["year"],
                q["field"]
            ]
        ):
            got = df.loc[
                q["year"],
                q["field"]
            ]

            ok += (
                abs(got - q["value_m"])
                / abs(q["value_m"])
                <= 0.01
            )

    return round(
        100 * ok / len(qs),
        1
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--build",
        action="store_true"
    )

    ap.add_argument(
        "--llm",
        action="store_true"
    )

    a = ap.parse_args()

    if a.build:
        build()

    else:
        qs = json.loads(
            TEST.read_text()
        )

        res = retrieval(qs)

        print(
            res.to_string(index=False)
        )

        res.to_csv(
            config.EVAL_DIR / "retrieval_results.csv",
            index=False
        )

        if a.llm:
            print(
                "Extraction accuracy (within 1%):",
                extraction(qs),
                "%"
            )
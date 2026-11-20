"""Hybrid retrieval + table retrieval + multi-company query parsing."""

import json
import re
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import config


def tok(s):
    return re.findall(r"\w+", s.lower())


class Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(config.INDEX_DIR / "faiss.index"))

        with open(config.INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(l) for l in f]

        for c in self.chunks:
            c.setdefault("type", "text")

        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.reranker = CrossEncoder(config.RERANK_MODEL)

        self.bm25 = BM25Okapi(
            [tok(c["text"]) for c in self.chunks]
        )

        self.companies = sorted(
            {c["company"] for c in self.chunks}
        )

    # ---------- query understanding (multi-company + memory) ----------
    def parse_query(self, q, memory=None):
        ql = q.lower()

        found = sorted(
            (ql.find(c.lower()), c)
            for c in self.companies
            if c.lower() in ql
        )

        companies = [c for _, c in found]

        years = [
            int(y)
            for y in re.findall(r"\b(20\d{2})\b", q)
        ]

        m = re.search(
            r"last\s+(\d+)\s*(?:y|yr|yrs|year|years)",
            ql
        )

        last_n = int(m.group(1)) if m else None

        if memory:
            # Follow-up questions
            companies = companies or memory.get("companies", [])

            if not years and not last_n:
                years = memory.get("years", [])
                last_n = memory.get("last_n")

        return companies, years, last_n

    def resolve_years(self, company, years, last_n):
        avail = sorted({
            c["year"]
            for c in self.chunks
            if c["company"] == company and c["year"]
        })

        if years:
            return [y for y in years if y in avail] or avail

        return avail[-last_n:] if last_n else avail

    def _ok(self, c, company, years):
        return (
            (not company or c["company"] == company)
            and (not years or c["year"] in years)
        )

    # ---------- text search: mode = vector | hybrid | rerank ----------
    def search(
        self,
        query,
        company=None,
        years=None,
        top_k=config.TOP_K,
        mode="rerank"
    ):
        kw = [
            k
            for k in config.KEYWORDS
            if k.lower() in query.lower()
        ]

        qv = self.embedder.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        _, ids = self.index.search(
            qv,
            min(
                self.index.ntotal,
                config.FETCH_K * 20
            )
        )

        vec = [
            i
            for i in ids[0]
            if i >= 0
            and self._ok(self.chunks[i], company, years)
        ][:config.FETCH_K]

        if mode == "vector":
            return [
                self.chunks[i]
                for i in vec[:top_k]
            ]

        scores = self.bm25.get_scores(
            tok(
                query + " " + " ".join(kw)
            )
        )

        kwd = [
            i
            for i in np.argsort(scores)[::-1]
            if self._ok(self.chunks[i], company, years)
        ][:config.FETCH_K]

        fused = {}

        for lst in (vec, kwd):
            for r, i in enumerate(lst):
                fused[i] = fused.get(i, 0) + 1 / (60 + r)

        cand = sorted(
            fused,
            key=fused.get,
            reverse=True
        )[:config.FETCH_K]

        if mode == "hybrid" or not cand:
            return [
                self.chunks[i]
                for i in cand[:top_k]
            ]

        ce = self.reranker.predict([
            (query, self.chunks[i]["text"])
            for i in cand
        ])

        return [
            self.chunks[cand[j]]
            for j in np.argsort(ce)[::-1][:top_k]
        ]

    # ---------- table search (financial statements) ----------
    def search_tables(
        self,
        company,
        years,
        latest_n=4,
        other_n=2,
        cap=8
    ):
        years = (
            sorted(years)
            if years
            else sorted({
                c["year"]
                for c in self.chunks
                if c["company"] == company and c["year"]
            })
        )

        out = []

        for y in reversed(years):
            tabs = [
                c
                for c in self.chunks
                if c["type"] == "table"
                and c["company"] == company
                and c["year"] == y
            ]

            score = lambda c: sum(
                k in c["text"].lower()
                for k in config.TABLE_KEYWORDS
            )

            tabs = sorted(
                tabs,
                key=score,
                reverse=True
            )

            out += tabs[
                :latest_n if y == years[-1] else other_n
            ]

        return out[:cap]

    # ---------- pick the best table for one financial statement ----------
    def find_statement(
        self,
        company,
        years,
        keys,
        n_filings=1,
        min_hits=2
    ):
        avail = sorted({
            c["year"]
            for c in self.chunks
            if c["company"] == company and c["year"]
        })

        ys = sorted(
            y for y in (years or avail)
            if y in avail
        ) or avail

        out = []

        for y in reversed(ys[-n_filings:]):
            tabs = [
                c
                for c in self.chunks
                if c["type"] == "table"
                and c["company"] == company
                and c["year"] == y
            ]

            # Normalize curly apostrophe (’) to straight apostrophe (')
            # so labels like "Shareholders’ equity" match
            # "Shareholders' equity".
            norm = lambda s: s.lower().replace("\u2019", "'")

            scored = [
                (
                    sum(
                        norm(k) in norm(c["text"])
                        for k in keys
                    ),
                    len(c["text"]),
                    c
                )
                for c in tabs
            ]

            scored = [
                t
                for t in scored
                if t[0] >= min_hits
            ]

            if scored:
                out.append(
                    max(
                        scored,
                        key=lambda t: (t[0], t[1])
                    )[2]
                )

        return out
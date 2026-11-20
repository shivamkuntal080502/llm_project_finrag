"""Run ONCE before the new ingest.py: builds the per-PDF cache from your existing index,
so PDFs you already parsed (Apple, Microsoft) are NOT parsed again."""
import json
import faiss, numpy as np
import config

CACHE = config.INDEX_DIR / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

index = faiss.read_index(str(config.INDEX_DIR / "faiss.index"))
emb = index.reconstruct_n(0, index.ntotal)
with open(config.INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
    records = [json.loads(l) for l in f]
assert len(records) == len(emb), "index and chunks.jsonl do not match"

by_src = {}
for i, r in enumerate(records):
    by_src.setdefault(r["source"], []).append(i)

for src, idx in by_src.items():
    pdf = config.PDF_DIR / src
    if not pdf.exists():
        print("skip (pdf not in data/pdfs):", src)
        continue
    stem = pdf.stem
    with open(CACHE / f"{stem}.jsonl", "w", encoding="utf-8") as f:
        for i in idx:
            f.write(json.dumps(records[i]) + "\n")
    np.save(CACHE / f"{stem}.npy", np.asarray(emb[idx], dtype="float32"))
    st = pdf.stat()
    (CACHE / f"{stem}.sig").write_text(f"{st.st_size}-{int(st.st_mtime)}")
    print("cached", src, len(idx), "chunks")
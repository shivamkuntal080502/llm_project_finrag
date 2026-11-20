"""PDFs -> Docling parse -> text chunks + TABLE chunks -> embeddings -> FAISS.
Per-PDF cache: already-parsed PDFs are skipped, so adding a company only parses the new files."""
import json
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
import config

CACHE = config.INDEX_DIR / "cache"


def parse_name(path):
    parts = path.stem.split("_")
    name = parts[0].strip()
    return {"company": name if (name.isupper() and len(name) > 1) else name.title(),   # apple -> Apple, HDFC stays
            "year": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None,
            "doc": parts[2] if len(parts) > 2 else "Annual report"}


def signature(pdf):
    st = pdf.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def parse_pdf(pdf, conv, chunker):
    meta = {**parse_name(pdf), "source": pdf.name}
    doc = conv.convert(str(pdf)).document
    records = []
    for ch in chunker.chunk(dl_doc=doc):
        page = next((it.prov[0].page_no for it in ch.meta.doc_items if it.prov), None)
        records.append({**meta, "page": page, "type": "text", "text": ch.text})
    for t in doc.tables:                                   # keep every table as markdown
        try:
            md = t.export_to_markdown(doc=doc)
        except TypeError:
            md = t.export_to_markdown()
        page = t.prov[0].page_no if t.prov else None
        if len(md) > 80:
            records.append({**meta, "page": page, "type": "table", "text": md})
    return records


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    conv = chunker = model = None
    all_records, all_emb = [], []

    for pdf in sorted(config.PDF_DIR.glob("*.pdf")):
        cj, ce, cs = (CACHE / f"{pdf.stem}{ext}" for ext in (".jsonl", ".npy", ".sig"))
        if cj.exists() and ce.exists() and cs.exists() and cs.read_text() == signature(pdf):
            print("Cached ", pdf.name)
            with open(cj, encoding="utf-8") as f:
                recs = [json.loads(l) for l in f]
            emb = np.load(ce)
        else:
            if conv is None:                                # load Docling only when needed
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling.chunking import HybridChunker
                opts = PdfPipelineOptions()
                opts.do_ocr = False                         # browser-printed PDFs already have text
                conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
                chunker = HybridChunker()
            print("Parsing", pdf.name)
            recs = parse_pdf(pdf, conv, chunker)
            if not recs:
                print("  (no content found, skipped)")
                continue
            model = model or SentenceTransformer(config.EMBED_MODEL)
            emb = model.encode([r["text"] for r in recs], batch_size=32,
                               normalize_embeddings=True, show_progress_bar=True)
            with open(cj, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            np.save(ce, np.asarray(emb, dtype="float32"))
            cs.write_text(signature(pdf))
        all_records += recs
        all_emb.append(np.asarray(emb, dtype="float32"))

    if not all_records:
        print("No PDFs found in", config.PDF_DIR)
        return
    emb = np.vstack(all_emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(config.INDEX_DIR / "faiss.index"))
    with open(config.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    n_tab = sum(r["type"] == "table" for r in all_records)
    print(f"Done. {len(all_records)} chunks ({n_tab} tables) from {len(all_emb)} PDFs. Index saved.")


if __name__ == "__main__":
    main()
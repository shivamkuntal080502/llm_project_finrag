"""Report -> PDF bytes."""
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


def make_pdf(res):
    buf = io.BytesIO()
    st = getSampleStyleSheet()
    els = [Paragraph("Financial Analysis Report", st["Title"])]
    for line in res["report"].split("\n"):
        line = line.replace("&", "&amp;").replace("<", "&lt;").replace("**", "").replace("#", "").strip()
        els.append(Paragraph(line, st["BodyText"]) if line else Spacer(1, 6))
    for c, df in res["metrics"].items():
        if df.empty:
            continue
        els += [Spacer(1, 12), Paragraph(f"{c} - key metrics", st["Heading2"])]
        t = df.T.reset_index()
        data = [["Metric"] + [str(x) for x in df.index]] + \
               [[str(v) if v == v else "-" for v in row] for row in t.values.tolist()]
        tb = Table(data, repeatRows=1)
        tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                ("FONTSIZE", (0, 0), (-1, -1), 7)]))
        els.append(tb)
    els += [Spacer(1, 12), Paragraph("Sources: " + ", ".join(f"{s} p.{p}" for s, p in res["sources"]),
                                     st["BodyText"])]
    SimpleDocTemplate(buf, pagesize=A4).build(els)
    return buf.getvalue()

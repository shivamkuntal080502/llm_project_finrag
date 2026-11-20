"""FinRAG - chat-style financial analyst UI (Streamlit)."""
import html as _html
import json
import time
import uuid

import altair as alt
import pandas as pd
import streamlit as st

import config
import llm
from analysis import risk_summary
from export import make_pdf
from pipeline import get_retriever, run
from verify import CurrencyMismatch, verify_company

st.set_page_config(page_title="FinRAG - Financial analyst", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- theme
DARK = st.get_option("theme.base") == "dark"
PAL = ({"bg": "#212121", "panel": "#171717", "card": "#2A2A2A", "border": "#3B3B3B", "text": "#ECECEC",
        "muted": "#9B9B9B", "accent": "#E07B5A", "good": "#4CAF7A", "bad": "#E5604F", "bubble": "#2F2F2F"}
       if DARK else
       {"bg": "#FAF9F5", "panel": "#F0EEE6", "card": "#FFFFFF", "border": "#E4E1D8", "text": "#1F1E1D",
        "muted": "#6B6A66", "accent": "#C96442", "good": "#2E7D4F", "bad": "#C0392B", "bubble": "#EFEDE4"})

ROOT_VARS = ":root{" + "".join(f"--{k}:{v};" for k, v in PAL.items()) + "}"
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea {font-family:'Inter',-apple-system,'Segoe UI',sans-serif !important;}
.stApp {background:var(--bg); color:var(--text);}
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
header[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:900px; padding-top:1.4rem; padding-bottom:8rem;}
section[data-testid="stSidebar"] {background:var(--panel); border-right:1px solid var(--border);}
section[data-testid="stSidebar"] .block-container {padding-top:1.2rem;}
[data-testid="stBottom"] > div {background:var(--bg);}

/* brand + sidebar */
.brand {display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.15rem; margin-bottom:.8rem;}
.brand .lg {width:32px; height:32px; border-radius:9px; background:var(--accent); color:#fff;
            display:flex; align-items:center; justify-content:center; font-size:1rem;}
.side-h {font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
         font-weight:600; margin:1.1rem 0 .35rem;}
.chip {display:inline-block; padding:2px 10px; border-radius:999px; background:var(--card);
       border:1px solid var(--border); font-size:.74rem; color:var(--muted); margin:2px 4px 2px 0;}

/* buttons */
.stButton > button {border-radius:12px; border:1px solid var(--border); background:var(--card);
    color:var(--text); font-weight:500; text-align:left; transition:all .15s; justify-content:flex-start;}
.stButton > button:hover {border-color:var(--accent); color:var(--accent); background:var(--card);}
.stButton > button[kind="primary"] {background:var(--accent); border-color:var(--accent); color:#fff; justify-content:center;}
.stButton > button[kind="primary"]:hover {filter:brightness(1.08); color:#fff;}
.stDownloadButton > button {border-radius:12px; border:1px solid var(--border); background:var(--card); color:var(--text);}

/* hero (empty state) */
.hero {text-align:center; padding:9vh 0 1.6rem;}
.hero .logo {width:58px; height:58px; border-radius:17px; background:var(--accent); color:#fff;
             display:flex; align-items:center; justify-content:center; font-size:1.7rem; margin:0 auto 16px;}
.hero h1 {font-size:2.05rem; font-weight:650; letter-spacing:-.02em; margin:0 0 .4rem;}
.hero p {color:var(--muted); font-size:1.02rem; margin:0;}

/* chat */
[data-testid="stChatMessage"] {background:transparent; border:none; padding:.4rem 0;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {flex-direction:row-reverse;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]
    {background:var(--bubble); border-radius:18px; padding:.65rem 1.05rem; flex:none; max-width:82%;}
[data-testid="stChatInput"] {border-radius:26px; border:1px solid var(--border); background:var(--card);
    box-shadow:0 4px 18px rgba(0,0,0,.07);}
[data-testid="stChatInput"] textarea {font-size:1rem;}

/* tabs / expanders / status */
.stTabs [data-baseweb="tab-list"] {gap:2px; border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"] {padding:6px 14px; font-weight:500; border-radius:8px 8px 0 0;}
[data-testid="stExpander"], [data-testid="stStatusWidget"], div[data-testid="stStatus"] {
    border:1px solid var(--border) !important; border-radius:14px !important; background:var(--card);}

/* KPI tiles + badges */
.cohead {display:flex; align-items:center; gap:8px; font-weight:650; margin:.9rem 0 .1rem; font-size:1.02rem;}
.kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:.45rem 0 .9rem;}
.kpi {background:var(--card); border:1px solid var(--border); border-radius:14px; padding:12px 14px;}
.kl {font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; font-weight:600;}
.kv {font-size:1.5rem; font-weight:650; margin-top:2px; letter-spacing:-.01em;}
.kd {font-size:.78rem; font-weight:600; margin-top:3px;}
.kd.up {color:var(--good);} .kd.down {color:var(--bad);}
.badge {display:inline-block; font-size:.78rem; font-weight:500; padding:4px 11px; border-radius:999px; margin:.2rem 0 .3rem;}
.badge.ok {background:color-mix(in srgb, var(--good) 14%, transparent); color:var(--good);}
.badge.warn {background:color-mix(in srgb, var(--bad) 14%, transparent); color:var(--bad);}
.disc {color:var(--muted); font-size:.74rem; margin-top:.7rem;}
"""
st.markdown(f"<style>{ROOT_VARS}{CSS}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def esc(t):
    """Streamlit treats $...$ as LaTeX: escape dollar signs so money renders normally."""
    return t.replace("$", "\\$")


def fmt_money(v, cur):
    """Values are in millions of the reporting currency."""
    if v is None or v != v:
        return "–"
    if cur == "INR":
        cr = v / 10.0                                   # 1 crore = 10 million
        return f"₹{cr / 1e5:,.2f} L Cr" if abs(cr) >= 1e5 else f"₹{cr:,.0f} Cr"
    a = abs(v)
    if a >= 1e6:
        return f"${v / 1e6:,.2f}T"
    if a >= 1e3:
        return f"${v / 1e3:,.1f}B"
    return f"${v:,.0f}M"


def token_stream(text):
    for tok in esc(text).split(" "):
        yield tok + " "
        time.sleep(0.012)


@st.cache_data(show_spinner=False)
def kb_summary(mtime):
    path = config.INDEX_DIR / "chunks.jsonl"
    seen = {}
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            seen.setdefault(r["company"], set()).add(r["year"])
    return {c: sorted(y for y in ys if y) for c, ys in sorted(seen.items())}


def kb():
    p = config.INDEX_DIR / "chunks.jsonl"
    return kb_summary(p.stat().st_mtime if p.exists() else 0)


@st.cache_resource(show_spinner="Loading AI models... (the first start takes about a minute)")
def warm():
    get_retriever()
    llm.load()
    return True


def suggestions(companies):
    n = [config.display(c) for c in companies]
    out = []
    if n:
        out.append(f"Give financial report of {n[0]} in last 3 years")
    if len(n) >= 2:
        out.append(f"Compare {n[0]} and {n[1]} over last 3 years")
    if len(n) >= 3:
        out.append(f"How profitable is {n[2]}? Show margins and ROE")
    if len(n) >= 4:
        out.append(f"Compare {n[-2]} and {n[-1]} balance sheet strength")
    return out


# ----------------------------------------------------------------------------- state
ss = st.session_state
if "chats" not in ss:
    ss.chats, ss.active = {}, None
ss.setdefault("typing", True)


def new_chat():
    if ss.active in ss.chats and not ss.chats[ss.active]["messages"]:
        return                                           # reuse the empty chat
    cid = uuid.uuid4().hex[:8]
    ss.chats[cid] = {"title": "New chat", "messages": [], "memory": {}}
    ss.active = cid


if ss.active not in ss.chats:
    new_chat()

try:
    warm()
except FileNotFoundError:
    st.error("No index found. Put your PDFs in `data/pdfs/` and run `python ingest.py` first.")
    st.stop()

KB = kb()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="brand"><span class="lg">📊</span>FinRAG</div>', unsafe_allow_html=True)
    if st.button("＋  New chat", key="newchat", type="primary"):
        new_chat()
        st.rerun()

    used = [(cid, c) for cid, c in ss.chats.items() if c["messages"]]
    if used:
        st.markdown('<div class="side-h">Recent</div>', unsafe_allow_html=True)
        for cid, c in reversed(used):
            label = c["title"] if len(c["title"]) <= 30 else c["title"][:29] + "…"
            if st.button(label, key=f"chat-{cid}", type="primary" if cid == ss.active else "secondary"):
                ss.active = cid
                st.rerun()

    st.markdown('<div class="side-h">Knowledge base</div>', unsafe_allow_html=True)
    if KB:
        st.markdown("".join(
            f'<span class="chip">{_html.escape(config.display(c))} · FY{min(ys)}–{max(ys)}</span>'
            for c, ys in KB.items()), unsafe_allow_html=True)
    else:
        st.caption("No documents indexed yet.")

    st.markdown('<div class="side-h">Settings</div>', unsafe_allow_html=True)
    ss.typing = st.toggle("Typing effect", value=ss.typing)
    st.caption(f"Model: {config.LLM_MODEL.split('/')[-1]}  \nRuns locally on your machine.")

chat = ss.chats[ss.active]


# ----------------------------------------------------------------------------- renderers
def kpi_html(res):
    out = ""
    for c, df in res["metrics"].items():
        if df.empty:
            continue
        cur = config.currency(c)
        last = df.index.max()
        prev = df.index[df.index < last].max() if len(df.index) > 1 else None
        op = "Operating margin %" if "Operating margin %" in df.columns else "Net margin %"
        tiles = []
        for col, kind in (("Revenue", "money"), ("Net income", "money"), (op, "pct"), ("ROE %", "pct")):
            if col not in df.columns or pd.isna(df.loc[last, col]):
                continue
            v = df.loc[last, col]
            val = fmt_money(v, cur) if kind == "money" else f"{v:.1f}%"
            delta = ""
            if prev is not None and pd.notna(df.loc[prev, col]):
                pv = df.loc[prev, col]
                if kind == "money":
                    d = (v - pv) / abs(pv) * 100 if pv else None
                    txt = f"{abs(d):.1f}%" if d is not None else ""
                else:
                    d = v - pv
                    txt = f"{abs(d):.1f} pts"
                if d is not None and d == d:
                    cls, arrow = ("up", "▲") if d >= 0 else ("down", "▼")
                    delta = f'<div class="kd {cls}">{arrow} {txt} vs FY{int(prev)}</div>'
            tiles.append(f'<div class="kpi"><div class="kl">{_html.escape(col.replace(" %", ""))}</div>'
                         f'<div class="kv">{val}</div>{delta}</div>')
        out += (f'<div class="cohead">{_html.escape(config.display(c))}'
                f'<span class="chip">FY{int(last)} · {cur}</span></div>'
                f'<div class="kpis">{"".join(tiles)}</div>')
    return out


def show_df(df):
    d = df.T.copy()
    d.columns = [f"FY{int(c)}" for c in d.columns]
    st.dataframe(d.style.format("{:,.2f}", na_rep="–"))


def line_chart(dfs, metric):
    rows = [{"Year": f"FY{int(y)}", "Company": config.display(c), "Value": float(v)}
            for c, d in dfs.items() if metric in d.columns for y, v in d[metric].dropna().items()]
    if not rows:
        st.info("No data for this metric.")
        return
    ch = (alt.Chart(pd.DataFrame(rows)).mark_line(point=True, strokeWidth=2.6)
          .encode(x=alt.X("Year:O", title=None),
                  y=alt.Y("Value:Q", title=metric),
                  color=alt.Color("Company:N", legend=alt.Legend(orient="bottom", title=None)),
                  tooltip=["Company", "Year", alt.Tooltip("Value:Q", format=",.2f")])
          .properties(height=300).configure_view(strokeWidth=0))
    st.altair_chart(ch)


def bar_chart(bench, metric):
    if metric not in bench.index:
        return
    row = bench.loc[metric].dropna()
    data = pd.DataFrame({"Company": [config.display(c) for c in row.index], "Value": row.values})
    ch = (alt.Chart(data).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
          .encode(x=alt.X("Company:N", title=None, sort=None), y=alt.Y("Value:Q", title=metric),
                  color=alt.Color("Company:N", legend=None),
                  tooltip=["Company", alt.Tooltip("Value:Q", format=",.2f")])
          .properties(height=280).configure_view(strokeWidth=0))
    st.altair_chart(ch)


def render_assistant(m, key):
    if m.get("error"):
        st.warning(m["content"])
        return
    res = m["res"]
    st.markdown(esc(m["content"]))

    if res["flags"]:
        nums = ", ".join(f"{f:,.2f}".rstrip("0").rstrip(".") for f in res["flags"][:6])
        st.markdown(f'<span class="badge warn">⚠ Could not match to the data: {nums}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge ok">✓ Every number in this answer matches the extracted data</span>',
                    unsafe_allow_html=True)

    st.markdown(kpi_html(res), unsafe_allow_html=True)
    dfs = {c: d for c, d in res["metrics"].items() if not d.empty}

    t = st.tabs(["Metrics", "Charts", "Compare", "Forecast", "Risks", "Verify", "Sources"])
    with t[0]:
        for c, df in res["metrics"].items():
            st.markdown(f"**{config.display(c)}**  ·  amounts in {config.currency(c)} millions")
            if df.empty:
                st.info("Could not extract the statements for this company.")
                continue
            show_df(df)
            for w in res.get("warnings", {}).get(c, []):
                st.warning(f"{config.display(c)} {w}")
    with t[1]:
        if dfs:
            cols = list(next(iter(dfs.values())).columns)
            metric = st.selectbox("Metric", cols, key=f"metric-{key}")
            line_chart(dfs, metric)
            if len({config.currency(c) for c in dfs}) > 1 and metric in ("Revenue", "Net income", "EBITDA", "Free cash flow"):
                st.caption("These companies report in different currencies: compare ratios, not absolute amounts.")
    with t[2]:
        b = res["benchmark"]
        if b is None or b.empty:
            st.info("Ask about two or more companies to see a side-by-side comparison.")
        else:
            bd = b.rename(columns=lambda c: config.display(c))
            st.dataframe(bd.style.format("{:,.2f}", na_rep="–"))
            bm = st.selectbox("Compare metric", list(b.index), key=f"bm-{key}")
            bar_chart(b.drop(columns=["Peer average"], errors="ignore"), bm)
    with t[3]:
        st.caption("Simple linear trend of the last years. Illustrative only, not investment advice.")
        for c, f in res["forecast"].items():
            if f is not None and not f.empty:
                st.markdown(f"**{config.display(c)}** · next year ({config.currency(c)} millions)")
                st.dataframe(f.style.format("{:,.0f}", na_rep="–"))
    with t[4]:
        r = get_retriever()
        for c in res["companies"]:
            if st.button(f"Summarise risk factors · {config.display(c)}", key=f"risk-{key}-{c}"):
                with st.spinner("Reading the risk factors..."):
                    ys = r.resolve_years(c, res["years"], res["memory"].get("last_n"))
                    m.setdefault("extra", {})[f"risk_{c}"] = risk_summary(r, c, ys)
            for y, txt in m.get("extra", {}).get(f"risk_{c}", {}).items():
                st.markdown(f"**{config.display(c)} · FY{y}**\n\n{esc(txt)}")
    with t[5]:
        st.caption("Compares extracted revenue and net income with Yahoo Finance (needs internet).")
        for c in res["companies"]:
            if st.button(f"Verify with Yahoo Finance · {config.display(c)}", key=f"ver-{key}-{c}"):
                try:
                    m.setdefault("extra", {})[f"ver_{c}"] = verify_company(c, res["metrics"][c])
                except CurrencyMismatch as e:
                    st.info(str(e))
                except Exception as e:
                    st.error(f"Could not reach Yahoo Finance: {e}")
            v = m.get("extra", {}).get(f"ver_{c}")
            if v is not None and not v.empty:
                st.markdown(f"**{config.display(c)}**")
                st.dataframe(v)
    with t[6]:
        st.write(" · ".join(f"{s} p.{p}" for s, p in res["sources"]))
        for ch in res["tables"]:
            with st.expander(f"{ch['source']} · page {ch['page']}"):
                st.text(ch["text"][:2500])

    st.download_button("⬇  Download PDF report", make_pdf(res), file_name="financial_report.pdf",
                       mime="application/pdf", key=f"pdf-{key}")
    st.markdown('<div class="disc">Generated from the filings in your knowledge base. '
                'Verify important numbers before relying on them. Not investment advice.</div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------- input
q = st.chat_input("Ask about revenue, margins, ROE, cash flow…  e.g. Compare Infosys and Wipro over last 3 years")
q = q or ss.pop("pending", None)
if q:
    chat["messages"].append({"role": "user", "content": q})
    if chat["title"] == "New chat":
        chat["title"] = q

# ----------------------------------------------------------------------------- conversation
if not chat["messages"]:
    st.markdown('<div class="hero"><div class="logo">📊</div>'
                '<h1>How can I help with your financial research?</h1>'
                '<p>Ask about any company in your knowledge base. Every number comes straight from the filings.</p>'
                '</div>', unsafe_allow_html=True)
    sug = suggestions(list(KB))
    cols = st.columns(2)
    for i, s in enumerate(sug):
        if cols[i % 2].button(s, key=f"sug-{i}"):
            ss.pending = s
            st.rerun()

for i, m in enumerate(chat["messages"]):
    with st.chat_message(m["role"], avatar="📊" if m["role"] == "assistant" else None):
        if m["role"] == "user":
            st.markdown(esc(m["content"]))
        else:
            render_assistant(m, f"{ss.active}-{i}")

if chat["messages"] and chat["messages"][-1]["role"] == "user":
    question = chat["messages"][-1]["content"]
    with st.chat_message("assistant", avatar="📊"):
        status = st.status("Analysing your question…", expanded=True)
        reply = None
        try:
            res = run(question, chat["memory"], progress=lambda s: status.write(s))
            if "error" in res:
                status.update(label="I need a company name", state="error", expanded=False)
                reply = {"role": "assistant", "error": True, "content": res["error"]}
            else:
                status.update(label="Analysis complete", state="complete", expanded=False)
                if ss.typing:
                    st.write_stream(token_stream(res["report"]))
                chat["memory"] = res["memory"]
                reply = {"role": "assistant", "content": res["report"], "res": res}
        except Exception as e:                           # keep the UI alive, show what happened
            status.update(label="Something went wrong", state="error", expanded=False)
            reply = {"role": "assistant", "error": True, "content": f"Something went wrong: {e}"}
    chat["messages"].append(reply)
    st.rerun()
import streamlit as st
import requests
import time
import re
import os
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

import base64

def _img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_RAG_B64 = _img_b64("RAG_LLM_pipeline_scheme.png")
_LLM_B64 = _img_b64("LLM_pipeline_scheme.png")

st.set_page_config(page_title="PubMed RAG Explorer", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background-color: #F0F4F8;
  color: #1A1D23;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
h1 { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #1A1D23; margin-bottom: 0.2rem; }

.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
  font-size: 0.88rem !important; line-height: 1.65 !important;
}
.stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
  font-size: 0.92rem !important; font-weight: 600 !important;
}

/* Column titles */
.col-title-rag { font-weight: 700; font-size: 1.05rem; color: #1A6B9A; border-bottom: 2px solid #1A6B9A; padding-bottom: 0.4rem; margin-bottom: 0.75rem; }
.col-title-llm { font-weight: 700; font-size: 1.05rem; color: #0F2D45; border-bottom: 2px solid #0F2D45; padding-bottom: 0.4rem; margin-bottom: 0.75rem; }

/* Pipeline Overview */
.how-label { font-size: 0.7rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem; }

/* Fork pipeline — stretches to fill column */
.pipeline-fork {
  display: flex;
  align-items: flex-start;
  gap: 0.3rem;
  width: 100%;
  box-sizing: border-box;
  padding-bottom: 0.5rem;
}
.pipeline-shared {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  flex-shrink: 0;
}
.pipeline-fork-point {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 0 0.2rem;
  flex-shrink: 0;
}
.fork-line-top { width: 1px; height: 14px; background: #CBD5E1; }
.fork-dot { width: 5px; height: 5px; border-radius: 50%; background: #CBD5E1; }
.fork-line-bot { width: 1px; height: 14px; background: #CBD5E1; }
.pipeline-branches {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  flex: 1;
}
.branch-row {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  flex-wrap: wrap;
}
.step-item { display: flex; flex-direction: column; align-items: center; gap: 0.2rem; flex-shrink: 0; }

/* Badges — palette from mockup */
.step-badge-rag    { background: #1A6B9A; color: white; border-radius: 6px; padding: 0.06rem 0.35rem; font-size: 0.58rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }
.step-badge-llm    { background: #0F2D45; color: white; border-radius: 6px; padding: 0.06rem 0.35rem; font-size: 0.58rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }
.step-badge-q      { background: #C0531E; color: white; border-radius: 6px; padding: 0.06rem 0.35rem; font-size: 0.58rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }
.step-badge-stress { background: #6B7280; color: white; border-radius: 6px; padding: 0.06rem 0.35rem; font-size: 0.58rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }
.step-badge-na     { background: #D1D5DB; color: #6B7280; border-radius: 6px; padding: 0.06rem 0.35rem; font-size: 0.58rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; }

/* Step text */
.step-text-rag    { font-size: 0.62rem; font-weight: 500; color: #475569; text-align: center; white-space: nowrap; text-transform: uppercase; letter-spacing: 0.04em; }
.step-text-llm    { font-size: 0.62rem; font-weight: 500; color: #475569; text-align: center; white-space: nowrap; text-transform: uppercase; letter-spacing: 0.04em; }
.step-text-q      { font-size: 0.62rem; font-weight: 500; color: #475569; text-align: center; white-space: nowrap; text-transform: uppercase; letter-spacing: 0.04em; }
.step-text-na     { font-size: 0.62rem; color: #9CA3AF; text-align: center; white-space: nowrap; font-style: italic; text-transform: uppercase; letter-spacing: 0.04em; }
.step-text-stress { font-size: 0.62rem; font-weight: 500; color: #475569; text-align: center; white-space: nowrap; text-transform: uppercase; letter-spacing: 0.04em; }

.step-arrow    { color: #CBD5E1; font-size: 0.62rem; padding: 0 0.05rem; margin-bottom: 0.35rem; flex-shrink: 0; }
.step-arrow-na { color: #E9EFF5; font-size: 0.62rem; padding: 0 0.05rem; margin-bottom: 0.35rem; flex-shrink: 0; }

/* Your turn divider */
.your-turn-divider { display: flex; align-items: center; gap: 0.75rem; margin: 1.1rem 0 0.75rem; }
.your-turn-line { flex: 1; height: 1px; background: #E2E8F0; }
.your-turn-label { font-size: 0.7rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; white-space: nowrap; }

/* Section labels */
.section-label-row { display: flex; align-items: center; gap: 0.4rem; margin-top: 0.75rem; margin-bottom: 0.3rem; }
.section-label-text-rag    { font-size: 0.68rem; font-weight: 600; color: #1A6B9A; text-transform: uppercase; letter-spacing: 0.06em; }
.section-label-text-llm    { font-size: 0.68rem; font-weight: 600; color: #0F2D45; text-transform: uppercase; letter-spacing: 0.06em; }
.section-label-text-q      { font-size: 0.68rem; font-weight: 600; color: #C0531E; text-transform: uppercase; letter-spacing: 0.06em; }
.section-label-text-stress { font-size: 0.68rem; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.06em; }

/* About — details/summary styled like step badges */
details.about-details {
  background: white;
  border: 1px solid #E2E8F0;
  border-radius: 8px;
  margin-bottom: 0.75rem;
  overflow: hidden;
}
details.about-details summary {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.85rem;
  cursor: pointer;
  list-style: none;
  user-select: none;
}
details.about-details summary::-webkit-details-marker { display: none; }
.about-badge {
  background: #0F2D45; color: white; border-radius: 3px;
  padding: 0.06rem 0.45rem; font-size: 0.58rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;
}
.about-label { font-size: 0.75rem; font-weight: 600; color: #475569; }
.about-chevron { margin-left: auto; font-size: 0.6rem; color: #94A3B8; }
details.about-details[open] .about-chevron { transform: rotate(180deg); display: inline-block; }
.about-body {
  padding: 0.75rem 1rem 1rem; font-size: 0.83rem;
  color: #334155; line-height: 1.7;
  border-top: 1px solid #F1F5F9;
}
.about-h { font-size: 0.72rem; font-weight: 700; color: #1A1D23;
  text-transform: uppercase; letter-spacing: 0.05em;
  margin: 0.7rem 0 0.25rem; }
.about-h:first-child { margin-top: 0; }
.about-tech { font-size: 0.72rem; color: #94A3B8; margin-top: 0.7rem;
  padding-top: 0.7rem; border-top: 1px solid #F1F5F9; }
/* Keep result expanders neutral */
div[data-testid="stExpander"] { border-radius: 8px !important; }

/* Result details — same pattern as about */
details.result-details {
  background: white;
  border: 1px solid #E2E8F0;
  border-radius: 8px;
  margin-bottom: 0.75rem;
  overflow: visible;
}
details.result-details summary {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.55rem 0.85rem;
  cursor: pointer;
  list-style: none;
  user-select: none;
}
details.result-details summary::-webkit-details-marker { display: none; }
.result-summary-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: #334155;
}
.result-chevron {
  margin-left: auto;
  font-size: 0.6rem;
  color: #94A3B8;
  transition: transform 0.15s;
}
details.result-details[open] .result-chevron { transform: rotate(180deg); }
.result-note {
  font-size: 0.75rem;
  color: #64748B;
  font-style: italic;
  padding: 0 0.85rem 0.5rem;
  border-bottom: 1px solid #F1F5F9;
  margin-bottom: 0.5rem;
}

/* PubMed intro bar */
.pubmed-bar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  background: white;
  border: 1px solid #E2E8F0;
  border-radius: 8px;
  padding: 0.6rem 1rem;
  margin-bottom: 1.25rem;
  font-size: 0.82rem;
  color: #475569;
}
.pubmed-badge {
  background: #2463AE;
  color: white;
  font-weight: 700;
  font-size: 0.78rem;
  border-radius: 4px;
  padding: 0.2rem 0.6rem;
  flex-shrink: 0;
  letter-spacing: 0.02em;
}
.pubmed-link { color: #1A6B9A; font-weight: 600; text-decoration: none; }

/* References */
.references-block { background: #EEF6FF; border-left: 3px solid #1A6B9A; border-radius: 0 4px 4px 0; padding: 0.6rem 0.85rem; margin-top: 0.85rem; font-size: 0.88rem !important; color: #1A3A5C; line-height: 1.8; }
.references-block p { margin: 0; }
.no-sources-block { background: #F8FAFC; border-left: 3px solid #94A3B8; border-radius: 0 4px 4px 0; padding: 0.6rem 0.85rem; margin-top: 0.85rem; font-size: 0.76rem !important; color: #475569; }

/* Expander note */
.expander-note { font-size: 0.75rem; color: #64748B; font-style: italic; margin-bottom: 0.75rem; }

/* Search button — neutral slate */
div.stButton > button {
  background: #475569; color: white; border: none; border-radius: 6px;
  padding: 0.55rem 2rem; font-weight: 600; font-size: 0.88rem;
  width: auto !important; min-width: 120px;
}
div.stButton > button:hover { background: #334155; }

.divider { border: none; border-top: 1px solid #CBD5E1; margin: 1.25rem 0; }

/* Result details headers */
details.result-details {
  background: white;
  border: 1px solid #E2E8F0;
  border-radius: 8px;
  margin-bottom: 0.5rem;
}
details.result-details summary {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.55rem 0.85rem;
  cursor: pointer;
  list-style: none;
  user-select: none;
}
details.result-details summary::-webkit-details-marker { display: none; }
.result-summary-label { font-size: 0.75rem; font-weight: 600; color: #334155; }
.result-chevron { margin-left: auto; font-size: 0.6rem; color: #94A3B8; }
details.result-details[open] .result-chevron { transform: rotate(180deg); display: inline-block; }
.result-note {
  font-size: 0.75rem; color: #64748B; font-style: italic;
  padding: 0 0.85rem 0.6rem; border-bottom: 1px solid #F1F5F9; margin-bottom: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# ── Env + clients ───────────────────────────────────────────────────────────────
load_dotenv()
NCBI_API_KEY      = os.getenv("NCBI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def fetch_pubmed_abstracts_batched(query, max_results=20, batch_size=20):
    resp = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db":"pubmed","term":query,"retmax":max_results,"retmode":"json","api_key":NCBI_API_KEY},
        timeout=30
    ).json()
    ids = resp.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return ""
    all_text = ""
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]
        all_text += requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db":"pubmed","id":",".join(batch),"rettype":"abstract","retmode":"text","api_key":NCBI_API_KEY}
        ).text
        time.sleep(0.5)
    return all_text

def clean_abstract(text):
    for kw in ["Declarations","Conflict of interest","Ethical Approval","Competing interests","©","Funding","Disclosures"]:
        m = re.compile(re.escape(kw), re.IGNORECASE).search(text)
        if m: text = text[:m.start()].strip()
    return text

def parse_abstracts_with_metadata(raw):
    papers = []
    for entry in re.split(r'\n\d+\.', raw):
        if "Author information:" not in entry: continue
        pm = re.search(r'PMID:\s*(\d+)', entry)
        if not pm: continue
        pmid  = pm.group(1)
        doi_m = re.search(r'\nDOI:\s*(10\.\S+)', entry)
        doi   = doi_m.group(1).strip('.') if doi_m else None
        yr_m  = re.search(r'\b(19|20)\d{2}\b', entry[:100])
        yr    = yr_m.group(0) if yr_m else None
        lines = [l.strip() for l in entry.split("Author information:")[0].strip().split('\n') if l.strip()]
        title   = " ".join(lines[1:2]) if len(lines)>1 else None
        authors = lines[2] if len(lines)>2 else None
        parts     = entry.split("Author information:")[-1].split("\n\n")
        abs_parts = [p.strip() for p in parts[1:] if len(p.strip())>100]
        abstract  = clean_abstract(" ".join(abs_parts)) if abs_parts else None
        if abstract and len(abstract)>50:
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            papers.append({"abstract":abstract,"title":title,"authors":authors,"year":yr,
                           "pmid":pmid,"pubmed_url":pubmed_url,
                           "doi_url":f"https://doi.org/{doi}" if doi else pubmed_url})
    return papers

def fetch_structured_metadata(pmids):
    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db":"pubmed","id":",".join(pmids),"retmode":"json","api_key":NCBI_API_KEY},
            timeout=30
        )
        data = response.json()
    except Exception:
        return {}
    out = {}
    result = data.get("result", {})
    if not isinstance(result, dict):
        return {}
    for pmid, rec in result.items():
        if pmid == "uids": continue
        if not isinstance(rec, dict): continue
        out[pmid] = {
            "title":   rec.get("title", "Unknown title"),
            "authors": ", ".join(a["name"] for a in rec.get("authors", []) if isinstance(a, dict)) or "Unknown",
            "year":    rec.get("pubdate", "")[:4],
        }
    return out

@st.cache_data(show_spinner=False)
def build_corpus(topic):
    raw    = fetch_pubmed_abstracts_batched(topic, max_results=20)
    papers = parse_abstracts_with_metadata(raw)
    if not papers: return [], []
    meta = fetch_structured_metadata([p["pmid"] for p in papers])
    for p in papers:
        if p["pmid"] in meta: p.update(meta[p["pmid"]])
    embs = load_embedding_model().encode([p["abstract"] for p in papers])
    return papers, embs

def build_collection(papers, embeddings):
    client = chromadb.Client()
    try: client.delete_collection("pubmed_rag")
    except: pass
    col = client.create_collection("pubmed_rag")
    col.add(
        documents=[p["abstract"] for p in papers],
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{"title":p.get("title",""),"authors":p.get("authors",""),
                    "year":p.get("year",""),"pubmed_url":p["pubmed_url"]} for p in papers],
        ids=[f"abstract_{i}" for i in range(len(papers))]
    )
    return col

def answer_with_rag(query, collection, n_results=5):
    emb = load_embedding_model().encode([query])[0].tolist()
    res = collection.query(query_embeddings=[emb], n_results=n_results, include=["documents","metadatas"])
    chunks, metas = res["documents"][0], res["metadatas"][0]
    context = ""
    for i,(chunk,m) in enumerate(zip(chunks,metas)):
        context += f"[Source {i+1}]\nTitle: {m['title']}\nAuthors: {m['authors']}\nYear: {m['year']}\nURL: {m['pubmed_url']}\nAbstract: {chunk}\n\n"
    for attempt in range(3):
        try:
            resp = anthropic_client.messages.create(
                model="claude-sonnet-4-6", max_tokens=1500, temperature=0,
                system="""You are a biomedical research assistant. Answer using ONLY the provided sources.
- Cite inline e.g. (Source 1).
- End with a References section: [Source N] Authors (Year). Title. URL
- If sources don't fully answer the question, say so explicitly.
- Never introduce information not in the sources.""",
                messages=[{"role":"user","content":f"Sources:\n{context}\nQuestion: {query}"}]
            )
            return resp.content[0].text, metas
        except anthropic.OverloadedError:
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
            else:
                return "The AI service is currently overloaded. Please wait a moment and try again.", metas

def answer_without_rag(query):
    for attempt in range(3):
        try:
            return anthropic_client.messages.create(
                model="claude-sonnet-4-6", max_tokens=1000, temperature=0,
                messages=[{"role":"user","content":query}]
            ).content[0].text
        except anthropic.OverloadedError:
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
            else:
                return "The AI service is currently overloaded. Please wait a moment and try again."

def split_refs(text):
    for marker in ["## References","### References","**References**"]:
        if marker in text:
            parts = text.split(marker,1)
            return parts[0].strip(), marker+"\n"+parts[1].strip()
    return text, None

def normalise_headings(text):
    text = re.sub(r'^### ','##### ', text, flags=re.MULTILINE)
    text = re.sub(r'^## ', '##### ', text, flags=re.MULTILINE)
    text = re.sub(r'^# ',  '##### ', text, flags=re.MULTILINE)
    return text

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'Inter',sans-serif; font-size:1.75rem; font-weight:700; color:#1A1D23; letter-spacing:-0.02em; line-height:1.2; margin-bottom:0.25rem; padding-top:0.5rem;">PubMed RAG Explorer</div>
<div style="font-family:'Inter',sans-serif; font-size:0.88rem; color:#64748B; margin-bottom:1rem;">Compare retrieval-augmented generation (RAG) against a plain LLM on live scientific literature.</div>
""", unsafe_allow_html=True)

# ── About — instant HTML details/summary ─────────────────────────────────────
st.markdown("""
<details class="about-details">
  <summary>
    <span class="about-badge">About</span>
    <span class="about-label">Why RAG in a world of LLMs?</span>
    <span class="about-chevron">▼</span>
  </summary>
  <div class="about-body">
    <div class="about-h">The problem with LLMs alone</div>
    <p>Large language models are remarkably capable — but their knowledge is frozen at training time.
    Ask them about a paper published last month, a niche subfield, or proprietary internal data,
    and they either don't know or — worse — confidently hallucinate a plausible-sounding answer
    that isn't grounded in any real source.</p>
    <div class="about-h">What RAG adds</div>
    <p>Retrieval-Augmented Generation (RAG) fixes this by giving the LLM a live, queryable knowledge
    base at inference time. Instead of relying on memorised training data, the system fetches
    relevant documents, converts them into searchable vectors, retrieves the most relevant passages
    for your question, and only then asks the LLM to generate an answer — grounded in those
    specific sources, which it cites.</p>
    <div class="about-h">What this tool demonstrates</div>
    <p>This proof-of-concept builds a RAG pipeline over live PubMed abstracts on any topic you choose.
    It answers the same question twice — once with RAG (grounded, cited) and once with the LLM alone
    (broad but unverifiable) — so you can directly compare outputs. A built-in stress test (Step 5b)
    deliberately asks an out-of-scope question: RAG should recognise it can't answer and say so;
    the LLM will answer confidently regardless.</p>
    <div class="about-tech">Built with: Python · PubMed API · sentence-transformers · ChromaDB · Anthropic Claude API · Streamlit</div>
  </div>
</details>
""", unsafe_allow_html=True)



st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── RAG pipeline HTML ─────────────────────────────────────────────────────────────
RAG_FORK = f"""
<img src="data:image/png;base64,{_RAG_B64}" style="width:100%; max-width:100%; max-height:140px; height:auto; display:block; object-fit:contain;" alt="LLM + RAG pipeline overview"/>
"""

LLM_FORK = f"""
<img src="data:image/png;base64,{_LLM_B64}" style="width:100%; max-width:100%; max-height:140px; height:auto; display:block; object-fit:contain;" alt="LLM pipeline overview"/>
"""

# ── Two columns ──────────────────────────────────────────────────────────────────
col_rag, col_llm = st.columns(2)

with col_rag:
    st.markdown('<div class="col-title-rag">LLM + RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="how-label">Pipeline Overview</div>', unsafe_allow_html=True)
    st.markdown(RAG_FORK, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""<div class="section-label-row">
      <span class="step-badge-rag">Step 1</span>
      <span class="section-label-text-rag">Your topic</span>
      <span style="font-size:0.68rem; color:#94A3B8; margin-left:0.4rem;">choose a field of scientific research, the tool will then fetch abstracts from <a class="pubmed-link" href="https://pubmed.ncbi.nlm.nih.gov" target="_blank">PubMed</a>
    </span>
    </div>""", unsafe_allow_html=True)



    topic = st.text_input("topic", value="Stem cell therapy Parkinson's disease",
                          label_visibility="collapsed",
                          help="20 recent abstracts on this topic will be fetched and indexed.")



with col_llm:
    st.markdown('<div class="col-title-llm">LLM</div>', unsafe_allow_html=True)
    st.markdown('<div class="how-label">Pipeline Overview</div>', unsafe_allow_html=True)
    st.markdown(LLM_FORK, unsafe_allow_html=True)

# ── Shared inputs ────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider' style='margin:1rem 0 0.75rem'/>", unsafe_allow_html=True)

st.markdown("""<div class="section-label-row">
  <span class="step-badge-q">Step 5a</span>
  <span class="section-label-text-q">In-scope query</span>
  <span style="font-size:0.68rem; color:#94A3B8; margin-left:0.4rem;">ask a question to the LLM, it must be related to your topic</span>

</div>""", unsafe_allow_html=True)
question = st.text_input("question",
                         value="What cell types are used in stem cell therapy for Parkinson's disease?",
                         label_visibility="collapsed")

st.markdown("""<div class="section-label-row" style="margin-top:0.5rem;">
  <span class="step-badge-stress">Step 5b</span>
  <span class="section-label-text-stress">Out-of-scope query</span>
  <span style="font-size:0.68rem; color:#94A3B8; margin-left:0.4rem;">ask a question to the LLM, it must NOT related to your topic</span>
</div>""", unsafe_allow_html=True)
oos_question = st.text_input("oos_question",
                             value="Why are cats afraid of cucumbers?",
                             label_visibility="collapsed",
                             help="RAG should decline; the LLM will answer confidently — that contrast is the point.")

# ── Centred search button ─────────────────────────────────────────────────────────
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
_, btn_mid, _ = st.columns([3, 1, 3])
with btn_mid:
    search = st.button("Search")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Run pipelines — parallel Claude calls ─────────────────────────────────────────
if search and topic and question and oos_question:
    with st.spinner(f"Fetching 20 abstracts from PubMed on '{topic}'..."):
        papers, embeddings = build_corpus(topic)
    if not papers:
        st.error("No abstracts retrieved. Try a different topic.")
        st.stop()

    collection = build_collection(papers, embeddings)
    st.session_state["collection"]   = collection
    st.session_state["corpus_built"] = True
    st.session_state["topic"]        = topic

    with st.spinner("Running all four queries in parallel..."):
        def run_rag_main():    return answer_with_rag(question, collection)
        def run_llm_main():    return answer_without_rag(question), None
        def run_rag_stress():  return answer_with_rag(oos_question, collection)
        def run_llm_stress():  return answer_without_rag(oos_question), None

        with ThreadPoolExecutor(max_workers=4) as executor:
            f_rag_main   = executor.submit(run_rag_main)
            f_llm_main   = executor.submit(run_llm_main)
            f_rag_stress = executor.submit(run_rag_stress)
            f_llm_stress = executor.submit(run_llm_stress)

        rag_main_text,   _ = f_rag_main.result()
        llm_main_result, _ = f_llm_main.result()
        rag_stress_text, _ = f_rag_stress.result()
        llm_stress_result, _ = f_llm_stress.result()

        llm_main_text   = llm_main_result
        llm_stress_text = llm_stress_result

    st.session_state["rag_body"],     st.session_state["rag_refs"]     = split_refs(rag_main_text)
    st.session_state["plain_raw"]     = llm_main_text
    st.session_state["oos_rag_body"], st.session_state["oos_rag_refs"] = split_refs(rag_stress_text)
    st.session_state["oos_plain"]     = llm_stress_text
    st.success(f"Done — {len(papers)} abstracts indexed, both questions answered.")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.get("corpus_built"):
    st.markdown("""<div class="section-label-row" style="margin-top:0.5rem;">
    <span class="step-badge-rag">Step 7a</span><span class="step-badge-llm">Step 7a</span>
    <span class="section-label-text-stress">Answers</span>
    </div>""", unsafe_allow_html=True)

    with st.expander("Results", expanded=True):
        st.markdown('<p class="expander-note">LLM + RAG grounds its answer in retrieved abstracts and cites sources. The LLM answers from training data — broader but unverifiable.</p>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("""<div class="section-label-row">
              <span class="step-badge-rag">Step 7a</span>
              <span class="section-label-text-rag">LLM + RAG — Answer (cited)</span>
            </div>""", unsafe_allow_html=True)
            rag_body = normalise_headings(st.session_state.get("rag_body",""))
            rag_refs = st.session_state.get("rag_refs")
            st.markdown(rag_body)
            if rag_refs:
                import re as _re
                clean_refs = rag_refs.replace('## References', 'References').replace('### References', 'References').replace('**References**', 'References')
                clean_refs = _re.sub(r'\s*(\[Source \d+\])', r'<br>\1', clean_refs).lstrip('<br>')
                st.markdown(f'<div class="references-block">{clean_refs}</div>', unsafe_allow_html=True)
        with r2:
            st.markdown("""<div class="section-label-row">
              <span class="step-badge-llm">Step 7a</span>
              <span class="section-label-text-llm">LLM — Answer (uncited)</span>
            </div>""", unsafe_allow_html=True)
            plain_raw = normalise_headings(st.session_state.get("plain_raw",""))
            st.markdown(plain_raw)
            st.markdown('<div class="no-sources-block">No sources cited — answer from training data only. Claims cannot be verified.</div>', unsafe_allow_html=True)

    st.markdown("""<div class="section-label-row" style="margin-top:0.5rem;">
    <span class="step-badge-rag">Step 7b</span><span class="step-badge-llm">Step 7b</span>
    <span class="section-label-text-stress">Answers</span>
    </div>""", unsafe_allow_html=True)

    with st.expander("Results"):
        st.markdown('<p class="expander-note">RAG should recognise the question falls outside its corpus and say so. The LLM answers confidently regardless — that contrast illustrates RAG&#39;s key advantage: <em>knowing what it doesn&#39;t know.</em></p>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("""<div class="section-label-row">
              <span class="step-badge-rag">Step 7b</span>
              <span class="section-label-text-rag">LLM + RAG — Answer (cited)</span>
            </div>""", unsafe_allow_html=True)
            oos_rag_body = normalise_headings(st.session_state.get("oos_rag_body",""))
            oos_rag_refs = st.session_state.get("oos_rag_refs")
            st.markdown(oos_rag_body)
            if oos_rag_refs:
                import re as _re
                clean_oos_refs = oos_rag_refs.replace('## References', 'References').replace('### References', 'References').replace('**References**', 'References')
                clean_oos_refs = _re.sub(r'\s*(\[Source \d+\])', r'<br>\1', clean_oos_refs).lstrip('<br>')
                st.markdown(f'<div class="references-block">{clean_oos_refs}</div>', unsafe_allow_html=True)
        with s2:
            st.markdown("""<div class="section-label-row">
              <span class="step-badge-llm">Step 7b</span>
              <span class="section-label-text-llm">LLM — Answer (uncited)</span>
            </div>""", unsafe_allow_html=True)
            oos_plain = normalise_headings(st.session_state.get("oos_plain",""))
            st.markdown(oos_plain)
            st.markdown('<div class="no-sources-block">No sources cited — answer from training data only.</div>', unsafe_allow_html=True)

if search and (not topic or not question or not oos_question):
    st.warning("Please fill in the topic and both question fields before searching.")

# PubMed RAG Explorer

> ⚠️ **Proof-of-concept.** This tool fetches live PubMed abstracts and uses them as a retrieval corpus. Results depend on what PubMed returns for your query at the time of search. The architecture, pipeline design, and LLM integration are the demonstrable outputs.

A Streamlit application that compares **retrieval-augmented generation (RAG)** against a plain **LLM** on live scientific literature — side by side, on any topic you choose.

🔗 **Live demo:** *(add Streamlit Cloud URL here)*

---

## Why this tool exists

Large language models are remarkably capable — but their knowledge is frozen at training time. Ask them about a paper published last month, a niche subfield, or proprietary internal data, and they either don't know or — worse — confidently hallucinate a plausible-sounding answer with no grounding in any real source.

**RAG fixes this** by giving the LLM a live, queryable knowledge base at inference time. Instead of relying on memorised training data, the system fetches relevant documents, converts them into searchable vectors, retrieves the most relevant passages for your question, and only then asks the LLM to generate an answer — grounded in those specific sources, which it cites.

This tool makes that difference visible and empirically testable.

---

## What it demonstrates

The app runs two parallel pipelines on the same question:

| | LLM + RAG | LLM alone |
|---|---|---|
| **Knowledge source** | Live PubMed abstracts, fetched at search time | Training data only |
| **Recency** | Reflects latest indexed literature | Frozen at training cutoff |
| **Citations** | Every claim traced to a specific paper + PubMed URL | No sources |
| **Out-of-scope behaviour** | Recognises corpus limits and declines | Answers confidently regardless |

A built-in **stress test (Step 5b)** asks a deliberately out-of-scope question — from a different field of medicine. RAG should say it can't answer; the LLM will answer confidently with no accountability. That contrast is the point.

### The honest caveat

For a well-covered public topic, a plain LLM (or one with web search) may give a similarly good answer. RAG's strongest use case is **private or specialised corpora** — internal trial data, unpublished research, proprietary document sets — where no LLM training data or web search can reach. This prototype demonstrates the architecture on a public corpus for accessibility; the design generalises directly to those higher-value private settings.

---

## Pipeline overview

**LLM + RAG**
```
Your topic → Fetch (PubMed) → Embed → Index (ChromaDB) → Retrieve top-5
                                                               ↓
                                        Your question → Generate → Answer (cited)
                                        Out-of-scope  → Generate → Answer (cited / declined)
```

**LLM alone**
```
Your question → Generate (training data only) → Answer (uncited)
Out-of-scope  → Generate (training data only) → Answer (uncited, confident)
```

---

## Features

- **Live PubMed corpus** — fetches 20 recent abstracts on any topic via the NCBI E-utilities API, with proper metadata (authors, year, PubMed URL)
- **Semantic retrieval** — abstracts embedded with `all-MiniLM-L6-v2`, stored in ChromaDB, queried by cosine similarity
- **Proper citations** — every RAG answer includes a References section with author, year, title, and clickable PubMed URL
- **Dual query** — Step 5a (in-scope) and Step 5b (stress test) run simultaneously via parallel API calls
- **Boilerplate cleaning** — abstracts stripped of declarations, competing interests, and copyright notices before embedding
- **Structured metadata** — titles and authors fetched from PubMed's eSummary endpoint rather than parsed from raw text, for reliability
- **Defensive error handling** — graceful fallback if PubMed returns unexpected responses for unusual queries

---

## Development approach

This prototype was built using AI-assisted code generation (Claude, Anthropic). In the interest of transparency:

**What was AI-generated:** the implementation code (Streamlit app, embedding pipeline, ChromaDB integration, API calls, CSS styling).

**What was not:** the problem framing, the choice of corpus source and retrieval architecture, the dual-query stress test design, the empirical RAG vs. LLM comparison methodology, the biological and scientific domain decisions, and the critical review of limitations.

The code is scaffolding. The framework and the findings are the contribution.

---

## Empirical finding

Running the same question twice — once with RAG, once with the LLM alone — surfaces a consistent pattern:

- **Plain LLM** gives the canonical, textbook answer: well-structured, broad, drawing on training consensus. Confident but unverifiable.
- **RAG** surfaces niche or recent findings present in the fetched corpus but underrepresented in training data — and cites exactly which paper supports each claim.

The trade-off: **RAG trades breadth for specificity and provenance.** It's not always "better" — it's a different shape of answer, narrower but traceable. For a regulated or scientific context where every claim must be verifiable, that difference matters enormously.

---

## Running locally

```bash
pip install streamlit sentence-transformers chromadb anthropic requests python-dotenv
streamlit run app.py
```

Create a `.env` file in the project root:

```
NCBI_API_KEY=your_ncbi_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

Get a free NCBI API key at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/).

> **Note:** the pipeline diagram images (`RAG_LLM_pipeline_scheme.png`, `LLM_pipeline_scheme.png`) must be in the same directory as `app.py`.

---

## Known limitations and future directions

| Limitation | Impact | Intended fix |
|---|---|---|
| General-purpose embedding model | `all-MiniLM-L6-v2` may miss biomedical semantic nuance | Replace with `PubMedBERT` or `BioLORD` |
| 20-abstract corpus | Rare or niche findings may not surface | Increase corpus size; add hybrid BM25 + semantic retrieval |
| Abstracts only (no full text) | Some answers require full paper content | Integrate PubMed Central full-text API |
| Public corpus only | Doesn't demonstrate RAG's strongest use case | Extend to private/proprietary document sets |
| Single LLM (Claude Sonnet) | No model comparison | Add model selector |
| No retrieval evaluation | Retrieval quality not formally measured | Add a labelled benchmark query set |

---

## Tech stack

Python · PubMed E-utilities API · `sentence-transformers` · ChromaDB · Anthropic Claude API · Streamlit

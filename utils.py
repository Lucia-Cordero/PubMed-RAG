"""
utils.py — Shared PubMed RAG pipeline functions.

Used by:
  - PubMed_RAG.ipynb          (v1 development notebook)
  - RAG_Encoder_Evaluation.ipynb  (v2 encoder comparison)
  - app.py                    (Streamlit app)

Functions here are the canonical, defensive versions with timeout
and error handling, sourced from app.py.
"""

import re
import time
import requests
import os


def fetch_pubmed_abstracts_batched(query, ncbi_api_key, max_results=20, batch_size=20):
    resp = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db":"pubmed","term":query,"retmax":max_results,"retmode":"json", "sort":"date", "api_key":ncbi_api_key},
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
            params={"db":"pubmed","id":",".join(batch),"rettype":"abstract","retmode":"text","api_key":ncbi_api_key},
            timeout=30
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


def fetch_structured_metadata(pmids, ncbi_api_key):
    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db":"pubmed","id":",".join(pmids),"retmode":"json","api_key":ncbi_api_key},
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

# precompute.py
"""
Run once: encodes all 100K candidates with SBERT and BM25.
Saves precomputed/ folder consumed by rank.py.

Usage:
    python precompute.py
    python precompute.py --candidates candidates.jsonl --out precomputed
"""

import argparse
import gzip
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np

# ─── JD text and Keyword Boost for semantic matching ────────────────────────
KEYWORD_BOOST = """
senior ai engineer founding team production embeddings retrieval ranking nlp vector search hybrid search
elasticsearch faiss pinecone qdrant weaviate milvus opensearch python machine learning llm rhlf lora qlora peft
evaluation framework ndcg mrr map recommendation systems startup founding team scrappy product-engineering attitude
shipper product company deployed at scale nlp ir information retrieval learning to rank
disfavors consulting no services no consulting Pune Noida relocation willing to relocate
"""


def get_full_jd_text() -> str:
    docx_path = Path('job_description.docx')
    if docx_path.exists():
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(docx_path) as zip_ref:
                doc_xml = zip_ref.read('word/document.xml')
                root = ET.fromstring(doc_xml)
                texts = [node.text for node in root.iter() if node.tag.endswith('t') and node.text]
                return '\n'.join(texts)
        except Exception as e:
            print(f"Warning: Failed to parse job_description.docx: {e}. Using fallback JD text.")
    
    # Fallback to full text of JD (copied from docx)
    return """Job Description: Senior AI Engineer – Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid – flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years (see "what we mean by this" below)

Let's be honest about this role
We're going to write this JD differently from most. We're a Series A company that just raised our round and we're building a new AI Engineering org from scratch. This is the kind of role where the JD changes every six months because the company changes every six months. So instead of pretending we have a fixed checklist, we're going to tell you what we actually need and what we've gotten wrong before.
If you've spent your career at Google or Meta and you want a well-scoped role with a defined ladder, this isn't it.
If you've spent your career bouncing between early-stage startups and you want to "just code" without having to think about product or recruiter workflows or eval frameworks, this also isn't it.
We need someone who is simultaneously comfortable with two things that sound contradictory:
Deep technical depth in modern ML systems – embeddings, retrieval, ranking, LLMs, fine-tuning.
Scrappy product-engineering attitude – willing to ship a working ranker in a week even if the underlying ML is "obviously suboptimal," because we need to learn from real users before we know what to actually optimize for.
These are not contradictory in real life. They feel contradictory because of how engineering culture sorted itself into "researcher" vs "shipper" archetypes. We need both modes available in the same person, and we'd rather you tilt slightly toward shipper than toward researcher.

What you'd actually be doing
The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles.
In practical terms, your first 90 days will probably look like:
Weeks 1-3: Audit what we currently have (it's mostly BM25 + rule-based scoring, working but not great). Identify the 3-4 highest-leverage things to fix.
Weeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking, but the architecture is your call.
Weeks 9-12: Set up the evaluation infrastructure – offline benchmarks, online A/B testing, recruiter-feedback loops – so we can keep improving without flying blind.
Beyond that, you'll be driving the long-term architecture of how we do candidate-JD matching at scale, mentoring the next round of hires (we're growing the team from 4 to 12 engineers in the next year), and working closely with our recruiter-experience PM on what to build.

What we mean by "5-9 years"
This is a range, not a requirement. Some people hit "senior engineer" judgment at 4 years; some never hit it after 15. We've used 5-9 because it's roughly where people we've hired into this kind of role have landed, but we'll seriously consider candidates outside the band if other signals are strong.
That said, here are the disqualifiers we actually apply:
If you've spent your career in pure research environments (academic labs, research-only roles) without any production deployment – we will not move forward. We are explicit about this. We've tried it twice and it didn't work for either side.
If your "AI experience" consists primarily of recent (under 12 months) projects using LangChain to call OpenAI – we will probably not move forward, unless you can demonstrate substantial pre-LLM-era ML production experience. We're looking for people who understood retrieval and ranking before it became fashionable.
If you are a senior engineer who hasn't written production code in the last 18 months because you've moved into "architecture" or tech lead roles – we will probably not move forward. This role writes code.

The skills inventory (please read carefully)
Most JDs list 20 skills and you're supposed to have all of them. We're going to do this differently.
Things you absolutely need:
Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We don't care which model – we care that you've handled embedding drift, index refresh, retrieval-quality regression in production.
Production experience with vector databases or hybrid search infrastructure – Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar. Again, the specific tech doesn't matter; the operational experience does.
Strong Python. Yes really, we care about code quality.
Hands-on experience designing evaluation frameworks for ranking systems – NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful.
Things we'd like you to have but won't reject you for:
LLM fine-tuning experience (LoRA, QLoRA, PEFT)
Experience with learning-to-rank models (XGBoost-based or neural)
Prior exposure to HR-tech, recruiting tech, or marketplace products
Background in distributed systems or large-scale inference optimization
Open-source contributions in the AI/ML space

Things we explicitly do NOT want:
Title-chasers. If your career trajectory shows you optimizing for "Senior" – "Staff" – "Principal" titles by switching companies every 1.5 years, we're not a fit. We need someone who plans to be here for 3+ years.
Framework enthusiasts. If your GitHub is full of LangChain tutorials and your blog posts are "How I used [hot framework] to build [demo]" – that's fine but it's not what we need. We need people who think about systems, not frameworks.
People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career. We've had bad fit experiences in both directions. If you're currently at one of these companies but have prior product-company experience, that's fine.
People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure. We respect your work but you'd be re-learning fundamentals here.
People whose work has been entirely on closed-source proprietary systems for 5+ years without external validation (papers, talks, open-source). We need to see how you think, not just trust that you can think.

On location, comp, and logistics:
Location: Pune/Noida-preferred but flexible. We have offices in Noida and Pune(mostly used Tue/Thu). We don't require any specific number of in-office days but we expect quarterly travel for offsites. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply. Outside India: case-by-case, but we don't sponsor work visas.
Notice period: We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice candidates are still in scope but the bar gets higher.

The vibe check:
We genuinely believe culture-fit matters more at this stage than skills-fit. Skills are teachable; the rest mostly isn't.
We work async-first and write a lot. If you find writing painful, you'll find this role painful.
We disagree openly and decide quickly. If you find that style abrasive, you'll find this role abrasive.
We move fast and break things, with the caveat that "things" are usually our internal assumptions, not user-facing systems. If you need a stable, mature codebase to be productive, you'll find this role unstable.

How to read between the lines:
The "ideal candidate" we're imagining is roughly:
6-8 years total experience, of which 4-5 are in applied ML/AI roles at product companies (not pure services).
Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale.
Has strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online), and LLM integration (when to fine-tune vs prompt) – and can defend them with reference to systems they actually built.
Located in or willing to relocate to Noida or Pune.
Active on Redrob platform (or has clear signal of being in the job market) so we can actually talk to them.
We are aware this is a narrow profile. We're not expecting to find many matches in a 100K candidate pool. We're explicitly OK with that – we'd rather see 10 great matches than 1000 maybes.
"""


def load_candidates(path: str) -> list:
    path = str(path)
    if path.endswith('.gz'):
        opener = lambda: gzip.open(path, 'rt', encoding='utf-8')
    else:
        opener = lambda: open(path, 'r', encoding='utf-8')
    candidates = []
    with opener() as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates


def build_career_text(c: dict) -> str:
    """Concatenate candidate profile text for semantic matching."""
    p = c['profile']
    parts = [
        p.get('headline', ''),
        p.get('current_title', ''),
        p.get('summary', '')[:300],
    ]
    career = c.get('career_history', [])
    for i, role in enumerate(career):
        title = role.get('title', '')
        company = role.get('company', '')
        desc = role.get('description', '')
        parts.append(f"{title} at {company}")
        if i < 5 and desc:
            parts.append(desc[:300])
    skills = [s['name'] for s in c.get('skills', [])[:50]]
    parts.append(' '.join(skills))
    return ' '.join(filter(None, parts))


def tokenize(text: str) -> list:
    return re.findall(r'\b[a-z]+\b', text.lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='candidates.jsonl')
    parser.add_argument('--out', default='precomputed')
    args = parser.parse_args()

    # Fallback checking for candidates file path (gzip vs raw)
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        if args.candidates.endswith('.gz') and Path(args.candidates[:-3]).exists():
            candidates_path = Path(args.candidates[:-3])
        elif not args.candidates.endswith('.gz') and Path(args.candidates + '.gz').exists():
            candidates_path = Path(args.candidates + '.gz')
        else:
            raise FileNotFoundError(f"Could not find candidates file: {args.candidates}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()

    # ── Load candidates ───────────────────────────────────────────────────────
    print(f'Loading candidates from {candidates_path}...')
    candidates = load_candidates(candidates_path)
    print(f'  {len(candidates)} candidates loaded in {time.time()-t0:.1f}s')

    # ── Save id → index mapping ───────────────────────────────────────────────
    id_to_idx = {c['candidate_id']: i for i, c in enumerate(candidates)}
    with open(out_dir / 'id_to_idx.pkl', 'wb') as f:
        pickle.dump(id_to_idx, f)
    print('  id_to_idx.pkl saved')

    # ── Extract career texts ──────────────────────────────────────────────────
    print('Extracting career texts...')
    texts = [build_career_text(c) for c in candidates]

    # ── SBERT embeddings ──────────────────────────────────────────────────────
    print('Computing SBERT embeddings (~20 min on CPU)...')
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')   # 22 MB, downloads once
    embeddings = model.encode(
        texts,
        batch_size=512,
        show_progress_bar=True,
        normalize_embeddings=True,    # L2-normalize: cosine = dot product
    )
    
    # SBERT chunking & averaging
    full_jd = get_full_jd_text()
    paragraphs = [p.strip() for p in full_jd.split('\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_words = 0
    for p in paragraphs:
        words = len(p.split())
        if current_words + words > 150:
            chunks.append(' '.join(current_chunk))
            current_chunk = [p]
            current_words = words
        else:
            current_chunk.append(p)
            current_words += words
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Add keyword boost as its own chunk
    chunks.append(KEYWORD_BOOST)
    
    # Encode all chunks and average
    chunk_embs = model.encode(chunks, normalize_embeddings=True)
    jd_emb = np.mean(chunk_embs, axis=0)
    jd_emb = jd_emb / (np.linalg.norm(jd_emb) + 1e-9)

    raw     = embeddings @ jd_emb                      # shape (n,)
    sbert_sc = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    np.save(out_dir / 'embeddings.npy',    embeddings.astype(np.float32))
    np.save(out_dir / 'sbert_scores.npy',  sbert_sc.astype(np.float32))
    print(f'  SBERT done. Range: [{sbert_sc.min():.3f}, {sbert_sc.max():.3f}]')

    # Quick sanity check — print top-5 by SBERT score
    top5_idx = sbert_sc.argsort()[-5:][::-1]
    print('  Top 5 by SBERT:')
    for idx in top5_idx:
        c = candidates[idx]
        print(f'    {sbert_sc[idx]:.3f}  {c["profile"]["current_title"]}'
              f'  |  {c["profile"]["location"]}')

    # ── BM25 scores ───────────────────────────────────────────────────────────
    print('Computing BM25 scores...')
    from rank_bm25 import BM25Okapi

    corpus      = [tokenize(t) for t in texts]
    jd_tokens   = tokenize(full_jd + " " + KEYWORD_BOOST)
    bm25        = BM25Okapi(corpus)
    bm25_raw    = np.array(bm25.get_scores(jd_tokens), dtype=np.float32)
    bm25_sc     = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-9)

    np.save(out_dir / 'bm25_scores.npy', bm25_sc)
    print(f'  BM25 done.  Range: [{bm25_sc.min():.3f}, {bm25_sc.max():.3f}]')

    elapsed = time.time() - t0
    print(f'\nPrecompute complete in {elapsed/60:.1f} min.')
    print(f'Files saved to {out_dir}/')
    print(f'  embeddings.npy   {embeddings.nbytes/1e6:.0f} MB')
    print(f'  sbert_scores.npy  bm25_scores.npy  id_to_idx.pkl')


if __name__ == '__main__':
    main()

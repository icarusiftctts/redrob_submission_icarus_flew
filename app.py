# app.py
"""
HuggingFace Spaces sandbox.
Push this file + features.py + requirements.txt to your HF Space repo.

The app accepts a JSON array of candidate profiles and returns a ranked CSV.
Designed with a premium theme and Outfit typography.
"""

import csv
import io
import json
import os
import tempfile
import uuid
import warnings
import gradio as gr
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from features import is_honeypot, score_candidate, has_minimum_evidence
from reasoning import generate_reasoning


def parse_candidates(input_str: str, file_obj) -> list:
    """
    Parses candidates from either a pasted text string or an uploaded file.
    Supports JSON list, JSONL format, and zipped gzip files.
    """
    candidates = []
    
    # 1. Check if file is uploaded
    if file_obj is not None:
        file_path = file_obj.name
        if file_path.endswith('.gz'):
            import gzip
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            candidates.append(json.loads(line))
                        except Exception:
                            pass
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('['):
                    try:
                        candidates = json.loads(content)
                    except Exception:
                        pass
                else:
                    # Try reading line-by-line (JSONL)
                    f.seek(0)
                    for line in f:
                        if line.strip():
                            try:
                                candidates.append(json.loads(line))
                            except Exception:
                                pass
    # 2. Otherwise use text input
    elif input_str.strip():
        content = input_str.strip()
        if content.startswith('['):
            try:
                candidates = json.loads(content)
            except Exception as e:
                raise ValueError(f"JSON Array Parse Error: {e}")
        else:
            # Maybe it's JSONL pasted
            for line in content.splitlines():
                if line.strip():
                    try:
                        candidates.append(json.loads(line))
                    except Exception:
                        pass
            if not candidates:
                # Try single dict parsing
                try:
                    candidates = [json.loads(content)]
                except Exception as e:
                    raise ValueError(f"Could not parse text input as JSON array, JSONL, or single object: {e}")
                    
    return candidates


def rank_candidates_ui(json_input: str, file_obj):
    """
    Core function called by the UI to process the candidate list.
    """
    try:
        candidates = parse_candidates(json_input, file_obj)
    except Exception as e:
        empty_df = pd.DataFrame(columns=['Rank', 'Candidate ID', 'Name', 'Score', 'Years of Exp', 'Location', 'Reasoning'])
        return (
            empty_df,  # Dataframe
            f"Error parsing input: {str(e)}",  # CSV output text
            None,  # Downloadable file
            "N/A", "N/A", "N/A", "N/A", "N/A"  # Stats
        )
        
    if not candidates:
        empty_df = pd.DataFrame(columns=['Rank', 'Candidate ID', 'Name', 'Score', 'Years of Exp', 'Location', 'Reasoning'])
        return (
            empty_df,
            "No candidates found. Please provide candidate data via the textbox or file upload.",
            None,
            "0", "0", "0", "N/A", "N/A"
        )
        
    total_input = len(candidates)
    honeypots_filtered = 0
    min_evidence_filtered = 0
    scored = []
    
    for c in candidates:
        if not isinstance(c, dict):
            continue
        if is_honeypot(c):
            honeypots_filtered += 1
            continue
        if not has_minimum_evidence(c):
            min_evidence_filtered += 1
            continue
            
        # Support inline semantic scores if provided in candidate profile dictionary
        # otherwise default to 0.0 (as the lightweight sandbox is offline)
        sbert = float(c.get('sbert_score', 0.0))
        bm25 = float(c.get('bm25_score', 0.0))
        
        score, feats = score_candidate(c, sbert_score=sbert, bm25_score=bm25)
        # Note: features.py now automatically populates 'overall_score' key.
        if score > 0:
            scored.append((c, score, feats))
            
    # Sort: score descending, candidate_id ascending (tie-break)
    scored.sort(key=lambda x: (-x[1], x[0].get('candidate_id', '')))
    top = scored[:100]
    
    total_valid = len(scored)
    total_filtered = honeypots_filtered + min_evidence_filtered
    
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
    
    df_rows = []
    for rank, (c, score, feats) in enumerate(top, 1):
        cid = c.get('candidate_id', f'UNKNOWN_{rank}')
        reasoning = generate_reasoning(c, feats)
        writer.writerow([cid, rank, score, reasoning])
        
        # UI Table fields
        profile = c.get('profile', {})
        name = profile.get('anonymized_name', profile.get('name', 'Anonymous'))
        yoe = profile.get('years_of_experience', 'N/A')
        loc = profile.get('location', 'N/A')
        df_rows.append({
            'Rank': rank,
            'Candidate ID': cid,
            'Name': name,
            'Score': score,
            'Years of Exp': yoe,
            'Location': loc,
            'Reasoning': reasoning
        })
        
    csv_str = buf.getvalue()
    
    # Save to a dedicated, allowlisted temp directory with a unique filename
    temp_dir = os.path.join(tempfile.gettempdir(), "gradio_outputs")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"submission_{uuid.uuid4().hex[:8]}.csv")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(csv_str)
        
    top_score = f"{top[0][1]:.4f}" if top else "N/A"
    top_cid = top[0][0].get('candidate_id', 'N/A') if top else "N/A"
    avg_score = f"{sum(s for _, s, _ in scored)/len(scored):.4f}" if scored else "N/A"
    
    df = pd.DataFrame(df_rows) if df_rows else pd.DataFrame(columns=['Rank', 'Candidate ID', 'Name', 'Score', 'Years of Exp', 'Location', 'Reasoning'])
    
    return (
        df,
        csv_str,
        temp_path,
        str(total_input),
        str(total_valid),
        str(total_filtered),
        f"{top_cid} (Score: {top_score})" if top else "N/A",
        avg_score
    )


# Soft modern premium theme using Indigo/Slate palettes with Outfit typography
premium_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

# Custom premium styling
custom_css = """
.header-container {
    text-align: center;
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(168, 85, 247, 0.05) 100%);
    border-radius: 16px;
    border: 1px solid rgba(99, 102, 241, 0.1);
}
.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.header-subtitle {
    font-size: 1.1rem;
    color: #4b5563;
}
.stat-card {
    border-radius: 12px;
    padding: 12px;
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
}
"""

# Sample candidate JSON block for the "Sample Data" tab
sample_json_text = """[
  {
    "candidate_id": "CAND_0000001",
    "profile": {
      "anonymized_name": "Ira Vora",
      "headline": "Backend Engineer | SQL, Spark, Cloud",
      "summary": "Software / data professional with 6.9 years of experience building data pipelines, backend systems, and analytics infrastructure. Deep knowledge of SQL and query optimization.",
      "location": "Pune",
      "country": "India",
      "years_of_experience": 6.9,
      "current_title": "Senior Backend Engineer",
      "current_company": "Mindtree",
      "current_company_size": "10001+",
      "current_industry": "IT Services"
    },
    "career_history": [
      {
        "company": "Mindtree",
        "title": "Senior Backend Engineer",
        "start_date": "2024-03-08",
        "end_date": null,
        "duration_months": 27,
        "is_current": true,
        "industry": "IT Services",
        "company_size": "10001+",
        "description": "Implemented streaming data pipelines on Kafka and Spark Streaming."
      }
    ],
    "education": [
      {
        "institution": "Lovely Professional University",
        "degree": "B.Tech",
        "field_of_study": "Computer Science",
        "start_year": 2017,
        "end_year": 2021,
        "grade": "8.24 CGPA",
        "tier": "tier_3"
      }
    ],
    "skills": [
      {
        "name": "NLP",
        "proficiency": "advanced",
        "endorsements": 8,
        "duration_months": 36
      },
      {
        "name": "Python",
        "proficiency": "expert",
        "endorsements": 15,
        "duration_months": 72
      },
      {
        "name": "Semantic Search",
        "proficiency": "advanced",
        "endorsements": 5,
        "duration_months": 24
      }
    ],
    "redrob_signals": {
      "skill_assessment_scores": {
        "NLP": 85,
        "Python": 90,
        "Semantic Search": 80
      },
      "willing_to_relocate": true,
      "expected_salary_range_inr_lpa": {
        "min": 35.0,
        "max": 45.0
      },
      "github_commits_last_12_months": 150,
      "notice_period_days": 30,
      "interview_completion_rate": 0.95
    }
  }
]"""

# JD details Markdown
jd_markdown = """
### Target Role: Senior AI Engineer (Retrieval & Semantic Search)

The model evaluates candidates based on suitability for designing high-performance embedding, indexing, retrieval, and evaluation layers.

#### Scoring Framework (Weights sum to 100%):
1. **Core Skills (32%)**: Expertise in vector databases (Faiss, Pinecone, Qdrant, Weaviate), retrieval methods (dense, sparse, BM25), NLP/Transformers, and Python.
2. **Career Relevancy (27%)**: Current & historical job titles, relevant tenure, job-hopping penalty, consulting firm penalty, and career progression.
3. **Semantic Alignment (19%)**: Precomputed similarity of resumes to the Job Description (SBERT / BM25 scores).
4. **Experience, Location, & Education (16%)**: Optimal experience fit (4-12 years YOE), geographic proximity to key hubs (Pune, Noida), and CS education background.
5. **New Signals (6%)**: Relevant certifications, recruiter activity, and headline density.

#### Programmatic Filters:
* **Honeypot Detection**: Automatically catches synthetic profiles using 4 strict rules (missing skill assessment scores, current job duration mismatch, skill alignment anomalies, and assessment ID consistency).
* **Minimum Evidence Rule**: Candidate must possess at least one verified core-domain skill (advanced/expert for >6 months, or intermediate for >24 months).
"""

with gr.Blocks(theme=premium_theme, css=custom_css, title="Redrob Candidate Ranker") as demo:
    gr.HTML(
        '''
        <div class="header-container">
            <h1 class="header-title">⚡ Redrob Candidate Ranker</h1>
            <p class="header-subtitle">Founding Team Matcher & Sandbox Ranker for Senior AI Engineer (Retrieval Focus)</p>
        </div>
        '''
    )
    
    with gr.Tabs():
        with gr.Tab("Interactive Ranker"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("### 📂 Input Candidate Data")
                    input_text = gr.Textbox(
                        lines=10,
                        placeholder="Paste JSON array or JSONL objects here...",
                        label="Candidate JSON Text",
                    )
                    input_file = gr.File(
                        label="Or upload file (.json, .jsonl, .gz)",
                        file_types=[".json", ".jsonl", ".gz"]
                    )
                    with gr.Row():
                        clear_btn = gr.Button("Reset", variant="secondary")
                        submit_btn = gr.Button("Rank Candidates", variant="primary")
                
                with gr.Column(scale=5):
                    gr.Markdown("### 📊 Metrics Summary")
                    with gr.Row():
                        stat_total = gr.Textbox(label="Total Candidates", value="0", interactive=False)
                        stat_valid = gr.Textbox(label="Valid Profiles", value="0", interactive=False)
                        stat_filtered = gr.Textbox(label="Filtered (Honeypots)", value="0", interactive=False)
                    with gr.Row():
                        stat_top = gr.Textbox(label="Top Rank Profile", value="N/A", interactive=False)
                        stat_avg = gr.Textbox(label="Average Score", value="N/A", interactive=False)
                        
                    gr.Markdown("### 💾 Export Ranked Output")
                    download_btn = gr.File(label="Download CSV Results", interactive=False)
            
            gr.Markdown("### 🏆 Top Ranked Candidates (Max 100)")
            output_table = gr.Dataframe(
                headers=['Rank', 'Candidate ID', 'Name', 'Score', 'Years of Exp', 'Location', 'Reasoning'],
                datatype=['number', 'str', 'str', 'number', 'number', 'str', 'str'],
                label="Ranked Table",
                interactive=False,
                wrap=True
            )
            
            with gr.Accordion("Raw CSV Output Text", open=False):
                output_csv_text = gr.Textbox(
                    lines=10,
                    label="CSV Content",
                    interactive=False
                )
                
        with gr.Tab("Role details & Weight allocation"):
            gr.Markdown(jd_markdown)
            
        with gr.Tab("Sample Test Data"):
            gr.Markdown("Use this snippet to quickly test the ranking engine. Paste it in the **Candidate JSON Text** field on the first tab.")
            gr.Code(value=sample_json_text, language="json")
            load_sample_btn = gr.Button("Load Sample into Textbox", variant="secondary")
            
    # Connect Components
    def load_sample():
        return sample_json_text
        
    load_sample_btn.click(
        fn=load_sample,
        inputs=None,
        outputs=input_text
    )
    
    def reset_ui():
        empty_df = pd.DataFrame(columns=['Rank', 'Candidate ID', 'Name', 'Score', 'Years of Exp', 'Location', 'Reasoning'])
        return "", None, None, "", "0", "0", "0", "N/A", "N/A", empty_df
        
    clear_btn.click(
        fn=reset_ui,
        inputs=None,
        outputs=[input_text, input_file, download_btn, output_csv_text, stat_total, stat_valid, stat_filtered, stat_top, stat_avg, output_table]
    )
    
    submit_btn.click(
        fn=rank_candidates_ui,
        inputs=[input_text, input_file],
        outputs=[output_table, output_csv_text, download_btn, stat_total, stat_valid, stat_filtered, stat_top, stat_avg]
    )

if __name__ == '__main__':
    output_dir = os.path.join(tempfile.gettempdir(), "gradio_outputs")
    os.makedirs(output_dir, exist_ok=True)
    demo.launch(allowed_paths=[output_dir])
# LLM Evaluator - Scientific Evaluation Framework

A FastAPI-based evaluation system for Large Language Models (LLMs) across 24 EU languages using measurable, reproducible, and scientifically-backed criteria.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM EVALUATOR                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │   5 Questions│   │  24 Languages│   │   5 Models   │   │  N Runs Each │  │
│  │              │ × │              │ × │              │ × │              │  │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘  │
│                           ↓                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    EVALUATION PIPELINE                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │   Prompt    │→ │   vLLM API  │→ │   Response  │→ │   Metrics   │  │   │
│  │  │   Builder   │  │   Client    │  │   Logger    │  │   Computer  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                           ↓                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    SCIENTIFIC METRICS                                │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │   │
│  │  │   Relevance  │ │Factual Acc.  │ │ Completeness │ │    Fluency   │ │   │
│  │  │  (Semantic   │ │   (NLI +     │ │ (Checklist/  │ │ (Perplexity+ │ │   │
│  │  │  Similarity) │ │  Reference)  │ │  LLM-Judge)  │ │   Grammar)   │ │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  │   │
│  │  │   Coherence  │ │Prompt Align. │ │Token Effic.  │                  │   │
│  │  │ (Discourse   │ │ (Hallucin.   │ │ (Info/Tok)   │                  │   │
│  │  │   Flow)      │ │  Detection)  │ │              │                  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                           ↓                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                 STATISTICAL ANALYSIS                                 │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │   │
│  │  │  Confidence  │ │  ICC (Inter- │ │  Paired t-   │ │  Cohen's d   │ │   │
│  │  │   Intervals  │ │   class Corr │ │    test      │ │  Effect Size │ │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │   │
│  │  ┌──────────────┐ ┌──────────────┐                                   │   │
│  │  │ Cross-Lang   │ │   ANOVA      │                                   │   │
│  │  │Consistency   │ │ (Language    │                                   │   │
│  │  │  (CV, CLRS)  │ │   Families)  │                                   │   │
│  │  └──────────────┘ └──────────────┘                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                           ↓                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      OUTPUTS                                         │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │   │
│  │  │   SQLite DB  │ │   JSON/CSV   │ │ Excel Report │ │  Statistical │ │   │
│  │  │   (Raw Data) │ │   (Export)   │ │   (Pretty)   │ │   Summary    │ │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Evaluation Framework

### 5 Evaluation Questions

| ID | Category | Description | Expected Elements |
|----|----------|-------------|-------------------|
| Q1 | Factual Knowledge | Portugal facts (capital, population, EU accession) | Lisbon, ~500k, 1986, Iberian Peninsula |
| Q2 | Logical Reasoning | Sheep math word problem | Step-by-step reasoning, correct answer (15) |
| Q3 | Instruction Following | JSON translation output | Valid JSON, exact keys, no markdown |
| Q4 | Cultural Nuance | EU multilingualism discussion | 3+ language examples, policy knowledge |
| Q5 | Summarization | CAP policy summary | Max 3 sentences, key facts preserved |

### Quality Metrics (Weighted Composite Score)

```
OQS = 0.25×RS + 0.20×FA + 0.15×CS + 0.15×FL + 0.10×CO + 0.10×PA + 0.05×TE

Where:
- RS = Relevance Score (semantic similarity)
- FA = Factual Accuracy (NLI + reference comparison)
- CS = Completeness Score (checklist-based)
- FL = Fluency Score (perplexity + grammar)
- CO = Coherence Score (discourse flow)
- PA = Prompt Alignment (hallucination detection)
- TE = Token Efficiency (info density)
```

### Cross-Language Robustness

```
CLRS = mean(OQS_all_languages) - 2×std(OQS_all_languages)
```

Penalizes high variance across languages.

### 24 EU Languages Supported

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| BG | Bulgarian | HR | Croatian | CS | Czech |
| DA | Danish | NL | Dutch | EN | English |
| ET | Estonian | FI | Finnish | FR | French |
| DE | German | EL | Greek | HU | Hungarian |
| GA | Irish | IT | Italian | LV | Latvian |
| LT | Lithuanian | MT | Maltese | PL | Polish |
| PT | Portuguese | RO | Romanian | SK | Slovak |
| SL | Slovenian | ES | Spanish | SV | Swedish |

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file:

```env
# vLLM Endpoint
LLM_URL=http://localhost:8000/v1/chat/completions
MODEL=mistral-7b-instruct-v0.2
API_KEY=your-api-key

# Evaluation Settings
NUM_RUNS_PER_QUESTION=3
BATCH_SIZE=5

# Embedding Model for Metrics
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# Storage
DATABASE_URL=sqlite:///./results/evaluation_results.db
RESULTS_DIR=./results
```

### 3. Run Server

```bash
cd llm_evaluator
uvicorn app.main:app --reload --port 8000
```

### 4. Run Evaluation

**Single Model:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "mistral-7b",
    "languages": ["EN", "DE", "FR"],
    "num_runs": 3,
    "temperature": 0.0
  }'
```

**All 24 Languages:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "mistral-7b",
    "languages": ["BG","HR","CS","DA","NL","EN","ET","FI","FR","DE","EL","HU","GA","IT","LV","LT","MT","PL","PT","RO","SK","SL","ES","SV"],
    "num_runs": 3
  }'
```

### 5. Get Results

```bash
# Get report
curl http://localhost:8000/report/mistral-7b

# Export to Excel
curl -O http://localhost:8000/export/mistral-7b?format=xlsx

# Compare models
curl "http://localhost:8000/compare?model_names=mistral-7b&model_names=llama-2-7b"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/languages` | GET | List 24 EU languages |
| `/questions` | GET | List evaluation questions |
| `/evaluate` | POST | Run single model evaluation |
| `/evaluate/batch` | POST | Run multi-model evaluation |
| `/results/{model}` | GET | Get raw results |
| `/report/{model}` | GET | Get statistical report |
| `/export/{model}` | GET | Export results (JSON/CSV/XLSX) |
| `/compare` | GET | Statistical model comparison |

## Directory Structure

```
llm_evaluator/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Settings & configuration
│   ├── models.py            # Database models
│   ├── schemas.py           # Pydantic schemas
│   └── services.py          # LLM client & pipeline
├── metrics/
│   ├── __init__.py
│   ├── scientific_metrics.py # Quality metrics
│   └── statistical_analysis.py # Statistical methods
├── translations/
│   └── eu_24_languages.py   # Question translations
├── data/
│   ├── evaluation_framework.md
│   └── evaluation_questions.json
├── results/                 # Output directory
├── requirements.txt
├── .env.example
└── README.md
```

## Statistical Validation

### Reproducibility (ICC)
```python
# Run same question 3x, compute Intraclass Correlation
icc, ci = compute_icc([[0.85, 0.87, 0.86], [0.82, 0.84, 0.83]])
# icc > 0.75 indicates good reproducibility
```

### Significance Testing
```python
# Compare two models
t_stat, p_value = paired_ttest(model_a_scores, model_b_scores)
# p < 0.05 indicates statistically significant difference

# Effect size
d = cohens_d(model_a_scores, model_b_scores)
# d > 0.8 is large effect, 0.5 medium, 0.2 small
```

### Bootstrap Confidence Intervals
```python
# Non-parametric confidence intervals
ci_lower, ci_upper = bootstrap_confidence_interval(scores, n_bootstrap=1000)
```

## License

MIT License - See LICENSE file for details.

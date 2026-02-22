# LLM Evaluator - RunPod Edition

Evaluate 5 Large Language Models across 24 EU languages using scientific metrics.

## 🚀 Quick Start (RunPod)

The fastest way to run evaluations is on RunPod with A40 48GB GPU.

```bash
# 1. SSH into RunPod and clone
ssh root@YOUR_RUNPOD_IP
cd /workspace
git clone https://github.com/YOUR_USERNAME/llm_evaluator.git
cd llm_evaluator

# 2. Setup (installs vLLM, creates venv)
bash runpod_setup/setup.sh

# 3. Edit config with your tokens
nano runpod_setup/config.yaml
# Set: hf_token, openai_api_key

# 4. Download models (~1-3 hours)
export HF_TOKEN="hf_your_token"
.venv/bin/python runpod_setup/download_models.py

# 5. Run evaluation (~2-3 hours)
export OPENAI_API_KEY="sk_your_key"
.venv/bin/python runpod_setup/evaluate.py

# 6. Download results to your laptop
scp -r root@YOUR_RUNPOD_IP:/workspace/evaluation_results ./
```

See [runpod_setup/README.md](runpod_setup/README.md) for detailed instructions.

---

## 📊 What This Evaluates

### 5 Models
- **EuroLLM-9B** - EU-focused model
- **Qwen3-30B-AWQ** - Alibaba's Qwen (quantized)
- **DeepSeek-14B** - Reasoning model
- **Mixtral-8x7B** - Mistral's MoE model
- **Mistral-Small-24B** - Mistral's dense model

### 24 EU Languages
BG, HR, CS, DA, NL, EN, ET, FI, FR, DE, EL, HU, GA, IT, LV, LT, MT, PL, PT, RO, SK, SL, ES, SV

### 5 Evaluation Questions
1. **Factual Knowledge** - Portugal facts
2. **Logical Reasoning** - Math word problem
3. **Instruction Following** - JSON output
4. **Cultural Nuance** - EU multilingualism
5. **Summarization** - CAP policy summary

### Quality Metrics
- Relevance (semantic similarity)
- Factual Accuracy (NLI-based)
- Completeness (checklist-based)
- Fluency (perplexity + grammar)
- Coherence (discourse flow)
- Prompt Alignment (hallucination detection)
- Token Efficiency

---

## 📁 Repository Structure

```
llm_evaluator/
├── runpod_setup/           # ⭐ RunPod evaluation scripts
│   ├── config.yaml         # Edit this with your tokens
│   ├── setup.sh            # One-time setup
│   ├── download_models.py  # Download all 5 models
│   ├── evaluate.py         # Main evaluation script
│   └── README.md           # Detailed instructions
│
├── app/                    # FastAPI service (legacy)
├── metrics/                # Scientific evaluation metrics
├── translations/           # EU 24 language questions
├── data/                   # Evaluation questions
├── requirements.txt
└── README.md               # This file
```

---

## 🔬 Scientific Framework

### Evaluation Pipeline
```
Questions × Languages × Models × Runs → Responses → Metrics → Analysis
```

### Quality Score Formula
```
OQS = 0.25×RS + 0.20×FA + 0.15×CS + 0.15×FL + 0.10×CO + 0.10×PA + 0.05×TE

Where:
- RS = Relevance Score
- FA = Factual Accuracy
- CS = Completeness Score
- FL = Fluency Score
- CO = Coherence Score
- PA = Prompt Alignment
- TE = Token Efficiency
```

### Statistical Validation
- **ICC (Intraclass Correlation)** - Measures reproducibility across runs
- **Paired t-test** - Compares models statistically
- **Cohen's d** - Effect size between models
- **Bootstrap CI** - Non-parametric confidence intervals
- **Cross-Language Robustness Score (CLRS)** - Penalizes high variance across languages

---

## 📈 Results

After evaluation, you get:
- **SQLite database** (`evaluation_results.db`) - All responses
- **JSON summaries** - Per-model statistics
- **Statistical reports** - Significance tests, comparisons

Query the database:
```bash
sqlite3 evaluation_results.db "SELECT model_name, AVG(latency_ms) FROM evaluations GROUP BY model_name;"
```

---

## 💰 Costs (RunPod)

- **A40 GPU**: ~$0.50-1.00/hour
- **Storage**: ~$0.20/GB/month
- **Full evaluation**: ~$3-6 (3-6 hours)

Models (~150GB) persist in `/workspace/` across pod restarts.

## 🔬 Evaluation Method

All metrics are computed **locally** on your RunPod GPU - no external API calls needed:

| Metric | Method |
|--------|--------|
| Relevance | Sentence-transformers embeddings |
| Factual Accuracy | Rule-based + reference matching |
| Completeness | Checklist-based scoring |
| Fluency | Perplexity + language detection |
| Coherence | Statistical discourse analysis |
| Prompt Alignment | Pattern matching |

---

## 📄 License

MIT License

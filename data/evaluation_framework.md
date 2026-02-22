# LLM Evaluation Framework - Scientific & Statistical Design

## Overview
This framework provides **measurable, reproducible, and statistically valid** criteria for evaluating LLM responses across 24 EU languages.

---

## 1. Evaluation Dimensions & Metrics

### A. Response Quality Metrics (Semantic)

| Metric | Description | Measurement Method | Statistical Basis |
|--------|-------------|-------------------|-------------------|
| **Relevance Score (RS)** | How well the response addresses the question | Semantic similarity (cosine similarity using embeddings) | Range [0, 1], mean ± SD per language |
| **Completeness Score (CS)** | Coverage of all aspects in the question | Checklist-based binary scoring / LLM-as-judge | Percentage, confidence intervals |
| **Factual Accuracy (FA)** | Correctness of factual claims | NLI (Natural Language Inference) + Reference comparison | Accuracy %, precision/recall |
| **Coherence Score (CO)** | Logical flow and structure | Discourse coherence metrics (entity/grid coherence) | Cumulative score, variance analysis |

### B. Linguistic Quality Metrics

| Metric | Description | Measurement Method | Statistical Basis |
|--------|-------------|-------------------|-------------------|
| **Fluency Score (FL)** | Grammatical correctness, naturalness | Perplexity (lower = better), grammar checkers | Perplexity mean, outlier detection |
| **Readability Index (RI)** | Text complexity appropriateness | Flesch-Kincaid, Flesch Reading Ease per language | Standardized scores |
| **Lexical Diversity (LD)** | Vocabulary richness | Type-Token Ratio (TTR), MTLD (Measure of Textual Lexical Diversity) | TTR ∈ [0,1], MTLD stability |

### C. Cross-Language Consistency Metrics

| Metric | Description | Measurement Method | Statistical Basis |
|--------|-------------|-------------------|-------------------|
| **Semantic Consistency (SC)** | Same answer meaning across languages | Cross-lingual embedding similarity | Pairwise similarity matrix |
| **Length Consistency (LC)** | Response length variation across languages | Coefficient of Variation (CV) | CV = σ/μ, target < 0.3 |
| **Structure Consistency (STC)** | Similar organizational patterns | Structural similarity (section count, list usage) | Jaccard similarity |

### D. Performance Efficiency Metrics

| Metric | Description | Measurement Method | Statistical Basis |
|--------|-------------|-------------------|-------------------|
| **Latency (L)** | Time to first token + total generation | Timestamp logging | Mean, P50, P95, P99 percentiles |
| **Throughput (T)** | Tokens per second | Token count / generation time | Rate with confidence intervals |
| **Token Efficiency (TE)** | Information density per token | Metrics score / token count | Efficiency ratio |

### E. Instruction Following Metrics

| Metric | Description | Measurement Method | Statistical Basis |
|--------|-------------|-------------------|-------------------|
| **Format Adherence (FA)** | Follows requested format (JSON, bullet, etc.) | Regex/template matching | Binary pass/fail % |
| **Constraint Satisfaction (CS)** | Meets length constraints (if specified) | Token count comparison | Compliance rate |
| **Prompt Alignment (PA)** | Stays on topic, no hallucination | Topic modeling, hallucination detection | Alignment score |

---

## 2. Composite Scoring

### Overall Quality Score (OQS)
```
OQS = 0.25×RS + 0.20×FA + 0.15×CS + 0.15×FL + 0.10×CO + 0.10×PA + 0.05×TE
```

### Cross-Language Robustness Score (CLRS)
```
CLRS = mean(OQS_all_languages) - 2×std(OQS_all_languages)
```
(Uses mean - 2σ to penalize high variance across languages)

---

## 3. Statistical Validation Methods

### A. Reproducibility
- **Inter-run Consistency**: Run same question 3x, calculate Intra-Class Correlation (ICC)
- **Temperature Sensitivity**: Test at T=0, T=0.3, T=0.7, measure variance

### B. Significance Testing
- **Paired t-test**: Compare models on same questions
- **ANOVA**: Compare across language families
- **Effect Size**: Cohen's d for practical significance

### C. Confidence Intervals
- 95% CI for all mean metrics
- Bootstrap resampling (n=1000) for non-parametric metrics

---

## 4. Data Structure (Excel/CSV Schema)

### Main Results Table
```
| run_id | timestamp | model_name | model_config | question_id | language | 
| response_text | raw_metrics (JSON) | computed_scores (JSON) | 
| latency_ms | tokens_generated | tokens_per_second |
```

### Aggregated Statistics Table
```
| model_name | metric_name | language | mean | std | min | max | 
| ci_lower_95 | ci_upper_95 | n_samples |
```

### Cross-Language Consistency Table
```
| model_name | question_id | language_pair | semantic_similarity | length_ratio |
```

---

## 5. Baseline & Benchmarking

- **Reference Model**: Use one high-quality model (e.g., GPT-4) as reference for relative scoring
- **Human Baseline**: Optional human annotations on subset for calibration
- **Random Baseline**: Expected scores for random responses

---

## 6. Implementation Notes

1. **Embeddings**: Use multilingual model (e.g., LaBSE, E5-multilingual, or BGE-m3)
2. **NLI Model**: Use multilingual NLI (e.g., XLM-RoBERTa-NLI)
3. **Statistical Library**: scipy, statsmodels for significance testing
4. **Reproducibility**: Fix seeds, log all hyperparameters

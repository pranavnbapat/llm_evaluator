# Scoring Engine

After raw responses are collected, the scoring engine computes **7 quality dimensions** per response. The `ResponseEvaluator` is initialized once and reused across batches.

Core file: `metrics/scientific_metrics.py`

---

## Metrics Overview

| Metric | How It's Computed | Model Used |
|---|---|---|
| **Relevance** | Cosine similarity between question embedding and response embedding | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| **Factual Accuracy** | NLI entailment score between response and reference facts/context documents | `cross-encoder/nli-deberta-v3-base` |
| **Completeness** | Coverage of expected elements + semantic coverage of context documents | Embedding similarity + keyword overlap |
| **Fluency** | Text classification (fluency label confidence) | `textfluency/taaco-*` or fallback |
| **Coherence** | Text classification (coherence label confidence) | `textfluency/taaco-*` or fallback |
| **Prompt Alignment** | Checks if response follows instructions (language, length, format) | Rule-based + embedding similarity |
| **Token Efficiency** | `1 - (\|tokens_generated\| / max_expected_tokens)` | Tokenizer-based |

---

## Batch Scoring Optimization

Instead of scoring one-by-one, the scoring scripts pre-compute expensive model-based metrics in **batches**:

```python
# In evaluate_context_results.py / evaluate_vision_results.py

fluency_scores    = evaluator.calculate_fluency_batch(responses, languages)
coherence_scores  = evaluator.calculate_coherence_batch(responses)
nli_scores        = evaluator.calculate_nli_entailment_batch(responses, contexts_batch)

for idx, item in enumerate(batch):
    precomputed = {}
    if fluency_scores is not None and idx < len(fluency_scores):
        precomputed["fluency"] = float(fluency_scores[idx])
    if coherence_scores is not None and idx < len(coherence_scores):
        precomputed["coherence"] = float(coherence_scores[idx])
    if nli_scores is not None and idx < len(nli_scores):
        precomputed["factual_accuracy"] = float(nli_scores[idx])

    scores = evaluator.evaluate_response(
        ...,
        precomputed_scores=precomputed if precomputed else None,
    )
```

Then the per-item `evaluate_response()` call uses these **precomputed scores** to avoid redundant inference.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EVALUATOR_SCORE_BATCH_SIZE` | `96` | Number of responses scored in one model inference batch |
| `EVALUATOR_SCORE_COMMIT_EVERY` | `200` | SQLite commit frequency (rows) |
| `EVALUATOR_METRICS_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` (cuda if available) |
| `TRANSFORMERS_VERBOSITY` | — | Set to `error` to reduce log noise |
| `HF_HUB_DISABLE_PROGRESS_BARS` | — | Set to `1` to disable progress bars |

Recommended settings by GPU:
- A100 / H200-SXM / B200: `EVALUATOR_SCORE_BATCH_SIZE=128`
- A40 / L40S: `EVALUATOR_SCORE_BATCH_SIZE=96`
- 3090: start lower and validate

---

## Composite Score

```python
overall_quality = (
    relevance        * 0.25 +
    factual_accuracy * 0.20 +
    completeness     * 0.15 +
    fluency          * 0.15 +
    coherence        * 0.10 +
    prompt_alignment * 0.10 +
    token_efficiency * 0.05
)
```

Weights are configurable via `metrics/metrics_config.yaml` profiles.

### Profile Configuration Example

```yaml
profiles:
  default:
    weights:
      relevance: 0.25
      factual_accuracy: 0.20
      completeness: 0.15
      fluency: 0.15
      coherence: 0.10
      prompt_alignment: 0.10
      token_efficiency: 0.05
    normalize_weights: false

  context:
    # Same weights but different NLI model tuning
    nli:
      model: "cross-encoder/nli-deberta-v3-base"
      entailment_threshold: 0.5
```

---

## Embedding Model

```python
class EmbeddingModel:
    _instance = None  # Singleton pattern

    def __new__(cls, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cpu"):
        ...

    def encode(self, texts: List[str]) -> np.ndarray:
        ...
```

- Uses **singleton pattern** so the model is loaded exactly once per scoring run
- Caches embeddings in `_embeddings_cache` to avoid re-encoding identical texts
- Falls back to dummy zero embeddings if `sentence-transformers` is unavailable

---

## Device Resolution

```python
def resolve_metrics_device() -> str:
    """
    EVALUATOR_METRICS_DEVICE:
      - auto (default): cuda if available, else cpu
      - cuda: force GPU
      - cpu: force CPU
    """
```

The device is resolved from:
1. Shell environment variable `EVALUATOR_METRICS_DEVICE`
2. Root `.env` file (`EVALUATOR_METRICS_DEVICE=cuda`)
3. Default to `auto`

---

## Fallbacks

If ML libraries are unavailable, the scorer degrades gracefully:

| Metric | ML Path | Fallback |
|---|---|---|
| Relevance | Embedding cosine similarity | Lexical word overlap |
| Factual Accuracy | NLI entailment | Keyword overlap with context |
| Fluency | Classifier model | Text length heuristic |
| Coherence | Classifier model | Sentence count heuristic |

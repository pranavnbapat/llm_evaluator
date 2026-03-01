# VRAM, Context Window, and Concurrency Deep Dive

This report explains VRAM, tokens, context windows, and concurrency using practical serving behavior. The focus is operational mechanics: what consumes memory, why OOM happens, and how to size systems reliably.

## 1) What actually sits in VRAM

When a model is loaded on a GPU, three major things consume VRAM.

### (A) Model weights
This is the model brain. Fixed size after load.

- A 7B model in FP16 is roughly 14 GB.
- A 30B model in FP16 is roughly 60 GB.
- Quantized models reduce this.

This part does not grow with prompt length.

### (B) KV cache (working memory)
This is where context lives while a request runs.

It grows with:
- number of active tokens in the request
- number of layers
- hidden/KV dimensions
- precision

This is why long context can trigger OOM even if the model itself fits.

### (C) Temporary runtime buffers
Non-zero memory for kernels, intermediate tensors, and serving overhead.

## 2) What context window means in plain terms

Context window is the total token budget visible at once:

`system prompt + user input + retrieved RAG context + chat history + generated output`

So for an 8K model:

`input tokens + output tokens <= 8192`

It is not input-only.

## 3) Tokens to text intuition

Very rough intuition:
- 1 token is about 3/4 of a word
- 100 tokens is about a short paragraph
- 1000 tokens is about a page

## 4) Why context directly affects GPU memory

KV cache stores attention state per token.

So:
- 1K tokens: small KV usage
- 32K tokens: very large KV usage

This is why the same model on the same GPU can run fine at short context and fail at long context.

## 5) Real RAG scenario end-to-end

User asks:

"What are the soil carbon benefits of crop rotation in Mediterranean climates?"

Approximate token accounting:

1. System prompt: ~120
2. User question: ~20
3. Retrieved context: 3 chunks x 500 = 1500
4. Formatting instructions: ~80

Total input = `120 + 20 + 1500 + 80 = 1720`

If output is ~600 tokens:

Total context usage = `1720 + 600 = 2320`

Fit check:
- 4K context model: OK
- 2K context model: likely truncation/failure

## 6) GPU selection logic

Step 1: pick GPU VRAM budget.

Example:
- A40: 48 GB VRAM

Step 2: check model weight fit.

Example:
- 30B FP16: usually too large for A40
- 30B quantized: can fit depending on quantization + overhead
- 7B FP16: easy fit

Step 3: check remaining VRAM for KV cache.

This determines usable context and concurrency.

## 7) Intuitive numbers (not exact)

On A40 (48 GB):

Case 1, 7B FP16:
- Weights ~14 GB
- Remaining for KV and runtime is large
- Long context is feasible

Case 2, 30B quantized:
- Weights may be ~35 to 40 GB
- Remaining is much smaller
- Context/concurrency becomes constrained

Same GPU, very different behavior.

## 8) Why output length also matters

During generation, output tokens are appended to active context.

So if `max_new_tokens = 2000`, you must reserve memory for that growth. Otherwise generation can fail mid-response.

## 9) Core idea for stakeholders

GPU sizing is not only "can I load the model?"

It is:

"Can I load the model and keep enough working memory for the context depth my use case needs?"

## 10) Clean mental model

VRAM as a whiteboard:
- Model weights = permanent writing
- Context tokens (KV cache) = temporary notes

Small model leaves more free whiteboard.
Large model leaves less room to think.

## 11) Practical workflow to present

1. Define workload shape.
- How much RAG context?
- How long output?

2. Estimate token budget.
- Input + output tokens.

3. Pick the model for quality target.

4. Pick GPU that can hold:
- model weights
- KV cache for required tokens and concurrency
- runtime overhead

## 12) One-sentence truth

Context window is not only a model property; in production it is a VRAM budget problem.

---

# Multi-user concurrency: what changes

When many users are active simultaneously, weights are shared, KV cache is not.

For `N` concurrent requests:

`VRAM = weights + KV_1 + KV_2 + ... + KV_N + runtime_overhead`

Each active request needs its own working memory.

## Why long context reduces concurrency

Short requests -> small KV per user -> more users fit.
Long RAG requests -> large KV per user -> fewer users fit.

## Throughput vs latency

- High concurrency mode: short prompts/outputs.
- Deep RAG mode: long prompts/outputs, fewer users.

The context-heavy evaluation setup aligns with deep-RAG mode by design.

## Continuous batching (vLLM)

Continuous batching improves throughput and utilization by mixing tokens from many requests into shared forward passes.

But it does not eliminate KV memory limits. KV still sets the hard concurrency cap.

---

# Specific estimate: best model at 4K context

Based on the scored database (`results/evaluation_scores_euf_context.db`), the top model is:

- `devstral-small-2-24b-instruct-2512-b200`
- Average overall quality: `0.8163`

## Assumptions for planning estimate

To estimate parallel users for 4K context, we use conservative planning assumptions:

- Model class: 24B instruct model
- Effective model + runtime footprint: ~55 GB on GPU memory budget
- KV cache per active 4K request: ~1.2 to 2.0 GB
  - lower end for lighter prompt/output + efficient batching
  - upper end for heavier prompt/output and generation growth

These are sizing estimates, not SLA guarantees.

## Estimated concurrent users at 4K

`concurrency ~= floor(available_vram_for_kv / kv_per_request)`

| GPU | Total VRAM | Estimated VRAM left for KV | KV per 4K request | Estimated parallel users |
| --- | ---: | ---: | ---: | ---: |
| B200 | 180 GB | ~125 GB | 1.2 to 2.0 GB | ~62 to 104 |
| A100 80GB | 80 GB | ~25 GB | 1.2 to 2.0 GB | ~12 to 20 |
| A40 48GB | 48 GB | ~0 GB (for this FP16-class 24B setup) | 1.2 to 2.0 GB | ~0 (not reliable) |

## Important A40 note

For this best-performing 24B model, A40 typically needs quantization and/or reduced context to be viable for production concurrency.

## Operational advice

For realistic deployment numbers, run a short load test with the exact prompt template, retrieval chunk sizes, and max output tokens. These three factors dominate KV-per-request usage.

---

# Additional sizing: models that fit A40 reliably and still support multi-user 4K

The same estimation method is applied below for models that are safer on A40 while still serving multiple users.

Using the score table, the strongest practical A40 candidates are:
- `eurollm-9b-instruct-2512` (best quality among A40-friendly models in this run)
- `mistral-nemo-instruct-2407`
- `deepseek-r1-distill-qwen-7b`
- `deepseek-r1-distill-qwen-14b`

## Assumptions for this A40 table

- GPU: A40, 48 GB VRAM
- Runtime overhead reserve: ~4 to 5 GB
- 4K context workload (prompt + output growth)
- KV per request varies by architecture and real prompt/output shape

## A40 concurrency estimates at 4K

| Model | Avg overall quality (scored results) | Approx model memory | Approx KV budget on A40 | KV per 4K request | Estimated parallel users |
| --- | ---: | ---: | ---: | ---: | ---: |
| `eurollm-9b-instruct-2512` | 0.8124 | ~18 GB | ~25 GB | 0.6 to 1.2 GB | ~20 to 41 |
| `mistral-nemo-instruct-2407` | 0.8063 | ~24 GB | ~19 GB | 0.8 to 1.4 GB | ~13 to 23 |
| `deepseek-r1-distill-qwen-7b` | 0.7749 | ~14 GB | ~29 GB | 0.5 to 1.0 GB | ~29 to 58 |
| `deepseek-r1-distill-qwen-14b` | 0.7743 | ~28 GB | ~15 GB | 0.8 to 1.5 GB | ~10 to 18 |

## How to read this table correctly

- Higher-quality models are usually larger and reduce headroom.
- Smaller models usually allow more concurrent users.
- Real capacity moves up/down based on chunk size and output length.

In the current benchmark mix, `eurollm-9b-instruct-2512` is usually the best A40 compromise between quality and parallel capacity.

## Deployment note

Model names ending in `-b200` in the result files are experiment labels. They do not mean the model is hard-locked to B200 hardware. Fit depends on precision/quantization, context, and serving overhead.

---

---

# Token Budget Analysis (Current Context Evaluation)

This section computes token budgets from:
- Prompt template in `runpod_setup/evaluate_context.py`
- Questions/context from `translations/eu_24_languages_euf_context.py`
- Generated responses from `results/evaluation_results_euf_context.xlsx`
- Model max context lengths from `runpod_setup/config.yaml`

Approximation used:
- **1 token ~= 4 to 5 characters**
- For each text, the report computes a range:
  - `tokens_est_min = ceil(chars / 5)`
  - `tokens_est_max = ceil(chars / 4)`
  - `tokens_est_mid = round((min + max)/2)`

Column naming guide:
- `*_min`: Lower-bound estimate (optimistic token count, using 5 chars per token).
- `*_max`: Upper-bound estimate (conservative token count, using 4 chars per token).
- `*_mid`: Midpoint estimate between `min` and `max`; use this as a single representative value when one number is needed.
- `p90`: 90th percentile. Example: if `response_tokens_est_mid_p90 = 1200`, then 90% of responses are at or below ~1200 tokens, and 10% are above it.

Definitions:
- **Total input tokens** = full prompt tokens (instructions + context block + question text)
- **Response tokens** = generated response tokens
- **Total sequence tokens** = input tokens + response tokens
- **Remaining output tokens** = `max_model_len - input_tokens`
- **Effective output cap** = `min(remaining_output_tokens, evaluation.max_tokens)` where `evaluation.max_tokens = 2048`

## A) Input/Output Token Profile by Base Question (from real responses)

| base_question | input_tokens_est_mid_mean | input_tokens_est_mid_min | input_tokens_est_mid_max | response_tokens_est_mid_mean | response_tokens_est_mid_p90 | response_tokens_est_mid_max | total_tokens_est_mid_mean | total_tokens_est_mid_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | 582.8 | 578 | 590 | 669.4 | 1142.2 | 1814 | 1252.1 | 2399 |
| Q2 | 569.0 | 566 | 572 | 631.3 | 1096.0 | 1503 | 1200.3 | 2071 |
| Q3 | 580.6 | 575 | 586 | 681.4 | 1156.0 | 1570 | 1262.0 | 2150 |
| Q4 | 568.9 | 565 | 574 | 647.9 | 1196.9 | 1780 | 1216.8 | 2345 |
| Q5 | 563.6 | 560 | 569 | 606.9 | 992.0 | 1690 | 1170.5 | 2259 |

## B) Per-Model Output Budget Using Config Max Context

| model_name | max_model_len | max_model_len_source | input_tokens_est_mid_mean | response_tokens_est_mid_mean | response_tokens_est_mid_p90 | response_tokens_est_mid_max | remaining_output_tokens_est_min | remaining_output_tokens_est_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-r1-distill-qwen-14b-b200 | 16384 | assumed_from_config_default | 573.0 | 1067.4 | 1308.0 | 1518 | 15729 | 15887 |
| deepseek-r1-distill-qwen-32b-b200 | 16384 | assumed_from_config_default | 573.0 | 904.8 | 1094.8 | 1441 | 15729 | 15887 |
| deepseek-r1-distill-qwen-7b | 16384 | assumed_from_config_default | 573.0 | 1054.1 | 1503.0 | 1814 | 15729 | 15887 |
| devstral-small-2-24b-instruct-2512-b200 | 16384 | config | 573.0 | 486.2 | 580.4 | 1200 | 15729 | 15887 |
| eurollm-22b-instruct-2512 | 16384 | config | 573.0 | 393.1 | 470.0 | 607 | 15729 | 15887 |
| eurollm-9b-instruct-2512 | 16384 | config | 573.0 | 441.7 | 544.4 | 695 | 15729 | 15887 |
| mistral-nemo-instruct-2407-b200 | 16384 | config | 573.0 | 543.2 | 672.0 | 1212 | 15729 | 15887 |
| mistral-small-3-2-24b-instruct-2506-awq-sym-b200 | 16384 | config | 573.0 | 443.5 | 517.6 | 1174 | 15729 | 15887 |
| qwen3-30b-a3b-instruct-awq-b200 | 16384 | assumed_from_config_default | 573.0 | 492.3 | 702.0 | 1220 | 15729 | 15887 |

## C) Per-Language x Per-Question (Input + Response + Total)

(120 rows; each cell aggregates all model/run responses for that language-question pair)

| language | base_question | input_tokens_est_mid | response_tokens_est_mid_mean | response_tokens_est_mid_p90 | response_tokens_est_mid_max | total_tokens_est_mid_mean | total_tokens_est_mid_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BG | Q1 | 582.0 | 627.3 | 944.0 | 1166 | 1209.3 | 1748 |
| BG | Q2 | 570.0 | 675.2 | 945.0 | 1200 | 1245.2 | 1770 |
| BG | Q3 | 582.0 | 657.5 | 1052.8 | 1120 | 1239.5 | 1702 |
| BG | Q4 | 568.0 | 511.6 | 770.8 | 854 | 1079.6 | 1422 |
| BG | Q5 | 565.0 | 550.7 | 890.0 | 972 | 1115.7 | 1537 |
| CS | Q1 | 579.0 | 579.5 | 922.2 | 932 | 1158.5 | 1511 |
| CS | Q2 | 569.0 | 579.9 | 1001.0 | 1205 | 1148.9 | 1774 |
| CS | Q3 | 577.0 | 646.9 | 1129.4 | 1292 | 1223.9 | 1869 |
| CS | Q4 | 568.0 | 582.7 | 1053.8 | 1333 | 1150.7 | 1901 |
| CS | Q5 | 562.0 | 573.0 | 918.0 | 1514 | 1135.0 | 2076 |
| DA | Q1 | 584.0 | 623.5 | 1130.4 | 1191 | 1207.5 | 1775 |
| DA | Q2 | 569.0 | 577.3 | 915.6 | 984 | 1146.3 | 1553 |
| DA | Q3 | 579.0 | 714.5 | 1453.8 | 1530 | 1293.5 | 2109 |
| DA | Q4 | 568.0 | 712.8 | 1401.6 | 1740 | 1280.8 | 2308 |
| DA | Q5 | 564.0 | 590.7 | 1105.0 | 1276 | 1154.7 | 1840 |
| DE | Q1 | 584.0 | 737.4 | 1416.0 | 1510 | 1321.4 | 2094 |
| DE | Q2 | 572.0 | 634.6 | 1062.0 | 1101 | 1206.6 | 1673 |
| DE | Q3 | 582.0 | 631.0 | 954.0 | 960 | 1213.0 | 1542 |
| DE | Q4 | 569.0 | 634.6 | 988.2 | 1298 | 1203.6 | 1867 |
| DE | Q5 | 561.0 | 614.2 | 968.0 | 992 | 1175.2 | 1553 |
| EL | Q1 | 583.0 | 700.6 | 1095.6 | 1212 | 1283.6 | 1795 |
| EL | Q2 | 570.0 | 716.8 | 1153.8 | 1194 | 1286.8 | 1764 |
| EL | Q3 | 584.0 | 640.4 | 978.8 | 1010 | 1224.4 | 1594 |
| EL | Q4 | 570.0 | 546.7 | 848.0 | 983 | 1116.7 | 1553 |
| EL | Q5 | 568.0 | 589.3 | 898.0 | 922 | 1157.3 | 1490 |
| EN | Q1 | 582.0 | 722.3 | 1193.6 | 1214 | 1304.3 | 1796 |
| EN | Q2 | 566.0 | 676.7 | 1050.0 | 1126 | 1242.7 | 1692 |
| EN | Q3 | 580.0 | 717.7 | 1074.4 | 1210 | 1297.7 | 1790 |
| EN | Q4 | 566.0 | 744.9 | 1275.6 | 1691 | 1310.9 | 2257 |
| EN | Q5 | 560.0 | 659.8 | 1011.8 | 1184 | 1219.8 | 1744 |
| ES | Q1 | 585.0 | 733.2 | 1342.4 | 1814 | 1318.2 | 2399 |
| ES | Q2 | 569.0 | 621.0 | 923.2 | 958 | 1190.0 | 1527 |
| ES | Q3 | 580.0 | 740.1 | 1386.4 | 1570 | 1320.1 | 2150 |
| ES | Q4 | 574.0 | 632.9 | 1038.0 | 1092 | 1206.9 | 1666 |
| ES | Q5 | 564.0 | 620.6 | 894.0 | 942 | 1184.6 | 1506 |
| ET | Q1 | 582.0 | 641.3 | 1217.6 | 1268 | 1223.3 | 1850 |
| ET | Q2 | 570.0 | 606.8 | 973.2 | 1184 | 1176.8 | 1754 |
| ET | Q3 | 575.0 | 713.8 | 1187.6 | 1424 | 1288.8 | 1999 |
| ET | Q4 | 568.0 | 697.7 | 1204.0 | 1243 | 1265.7 | 1811 |
| ET | Q5 | 560.0 | 597.9 | 1003.4 | 1196 | 1157.9 | 1756 |
| FI | Q1 | 584.0 | 700.3 | 1230.0 | 1422 | 1284.3 | 2006 |
| FI | Q2 | 570.0 | 598.2 | 994.0 | 1318 | 1168.2 | 1888 |
| FI | Q3 | 578.0 | 679.3 | 1200.6 | 1506 | 1257.3 | 2084 |
| FI | Q4 | 567.0 | 644.0 | 1070.0 | 1326 | 1211.0 | 1893 |
| FI | Q5 | 560.0 | 557.5 | 862.0 | 898 | 1117.5 | 1458 |
| FR | Q1 | 590.0 | 685.7 | 1068.4 | 1210 | 1275.7 | 1800 |
| FR | Q2 | 570.0 | 607.8 | 947.0 | 1179 | 1177.8 | 1749 |
| FR | Q3 | 586.0 | 654.0 | 1112.0 | 1166 | 1240.0 | 1752 |
| FR | Q4 | 572.0 | 652.5 | 1100.8 | 1279 | 1224.5 | 1851 |
| FR | Q5 | 565.0 | 631.0 | 1031.6 | 1148 | 1196.0 | 1713 |
| GA | Q1 | 584.0 | 865.0 | 1456.0 | 1462 | 1449.0 | 2046 |
| GA | Q2 | 570.0 | 910.2 | 1398.4 | 1501 | 1480.2 | 2071 |
| GA | Q3 | 580.0 | 1033.6 | 1408.8 | 1516 | 1613.6 | 2096 |
| GA | Q4 | 566.0 | 872.2 | 1425.4 | 1441 | 1438.2 | 2007 |
| GA | Q5 | 569.0 | 892.3 | 1388.8 | 1690 | 1461.3 | 2259 |
| HR | Q1 | 580.0 | 611.4 | 994.0 | 1174 | 1191.4 | 1754 |
| HR | Q2 | 572.0 | 615.5 | 1016.0 | 1297 | 1187.5 | 1869 |
| HR | Q3 | 577.0 | 616.7 | 1105.2 | 1308 | 1193.7 | 1885 |
| HR | Q4 | 568.0 | 560.7 | 1021.0 | 1240 | 1128.7 | 1808 |
| HR | Q5 | 564.0 | 543.7 | 900.0 | 1192 | 1107.7 | 1756 |
| HU | Q1 | 582.0 | 650.7 | 1069.2 | 1125 | 1232.7 | 1707 |
| HU | Q2 | 570.0 | 566.9 | 918.0 | 1140 | 1136.9 | 1710 |
| HU | Q3 | 582.0 | 659.4 | 1130.8 | 1223 | 1241.4 | 1805 |
| HU | Q4 | 572.0 | 688.8 | 1331.8 | 1462 | 1260.8 | 2034 |
| HU | Q5 | 564.0 | 546.7 | 855.6 | 1043 | 1110.7 | 1607 |
| IT | Q1 | 582.0 | 639.1 | 1068.0 | 1145 | 1221.1 | 1727 |
| IT | Q2 | 570.0 | 618.0 | 965.4 | 996 | 1188.0 | 1566 |
| IT | Q3 | 582.0 | 644.8 | 1004.0 | 1204 | 1226.8 | 1786 |
| IT | Q4 | 572.0 | 719.6 | 1221.8 | 1244 | 1291.6 | 1816 |
| IT | Q5 | 562.0 | 617.6 | 1033.6 | 1057 | 1179.6 | 1619 |
| LT | Q1 | 582.0 | 721.0 | 1434.4 | 1513 | 1303.0 | 2095 |
| LT | Q2 | 569.0 | 623.3 | 1061.0 | 1172 | 1192.3 | 1741 |
| LT | Q3 | 579.0 | 649.3 | 1082.4 | 1241 | 1228.3 | 1820 |
| LT | Q4 | 567.0 | 670.9 | 1238.8 | 1411 | 1237.9 | 1978 |
| LT | Q5 | 562.0 | 674.0 | 1291.2 | 1482 | 1236.0 | 2044 |
| LV | Q1 | 580.0 | 671.6 | 1066.4 | 1082 | 1251.6 | 1662 |
| LV | Q2 | 569.0 | 612.1 | 1061.6 | 1188 | 1181.1 | 1757 |
| LV | Q3 | 578.0 | 627.6 | 992.4 | 1014 | 1205.6 | 1592 |
| LV | Q4 | 568.0 | 618.3 | 1016.6 | 1202 | 1186.3 | 1770 |
| LV | Q5 | 562.0 | 613.0 | 893.6 | 1035 | 1175.0 | 1597 |
| MT | Q1 | 584.0 | 724.1 | 1044.0 | 1152 | 1308.1 | 1736 |
| MT | Q2 | 568.0 | 690.6 | 1010.6 | 1124 | 1258.6 | 1692 |
| MT | Q3 | 582.0 | 828.5 | 1226.0 | 1244 | 1410.5 | 1826 |
| MT | Q4 | 569.0 | 722.6 | 1216.4 | 1292 | 1291.6 | 1861 |
| MT | Q5 | 567.0 | 648.2 | 1226.0 | 1262 | 1215.2 | 1829 |
| NL | Q1 | 585.0 | 647.4 | 1005.4 | 1108 | 1232.4 | 1693 |
| NL | Q2 | 569.0 | 583.9 | 879.2 | 914 | 1152.9 | 1483 |
| NL | Q3 | 584.0 | 636.6 | 1003.6 | 1102 | 1220.6 | 1686 |
| NL | Q4 | 569.0 | 627.0 | 1093.8 | 1182 | 1196.0 | 1751 |
| NL | Q5 | 560.0 | 574.4 | 895.2 | 909 | 1134.4 | 1469 |
| PL | Q1 | 580.0 | 637.5 | 999.6 | 1086 | 1217.5 | 1666 |
| PL | Q2 | 566.0 | 674.7 | 1208.0 | 1256 | 1240.7 | 1822 |
| PL | Q3 | 584.0 | 582.3 | 973.0 | 983 | 1166.3 | 1567 |
| PL | Q4 | 567.0 | 627.6 | 1199.0 | 1210 | 1194.6 | 1777 |
| PL | Q5 | 567.0 | 578.3 | 970.0 | 988 | 1145.3 | 1555 |
| PT | Q1 | 587.0 | 675.0 | 1076.0 | 1217 | 1262.0 | 1804 |
| PT | Q2 | 567.0 | 563.8 | 855.2 | 956 | 1130.8 | 1523 |
| PT | Q3 | 582.0 | 659.3 | 1094.0 | 1197 | 1241.3 | 1779 |
| PT | Q4 | 574.0 | 621.9 | 984.8 | 1356 | 1195.9 | 1930 |
| PT | Q5 | 564.0 | 603.4 | 922.4 | 1001 | 1167.4 | 1565 |
| RO | Q1 | 584.0 | 625.3 | 1009.4 | 1202 | 1209.3 | 1786 |
| RO | Q2 | 567.0 | 571.0 | 860.0 | 971 | 1138.0 | 1538 |
| RO | Q3 | 583.0 | 660.8 | 1021.6 | 1156 | 1243.8 | 1739 |
| RO | Q4 | 569.0 | 553.5 | 884.0 | 1022 | 1122.5 | 1591 |
| RO | Q5 | 564.0 | 574.7 | 900.0 | 934 | 1138.7 | 1498 |
| SK | Q1 | 578.0 | 559.3 | 961.2 | 1059 | 1137.3 | 1637 |
| SK | Q2 | 569.0 | 556.7 | 856.4 | 902 | 1125.7 | 1471 |
| SK | Q3 | 579.0 | 631.0 | 1029.6 | 1146 | 1210.0 | 1725 |
| SK | Q4 | 570.0 | 638.8 | 1175.8 | 1270 | 1208.8 | 1840 |
| SK | Q5 | 565.0 | 596.8 | 1036.0 | 1141 | 1161.8 | 1706 |
| SL | Q1 | 578.0 | 655.5 | 1302.4 | 1726 | 1233.5 | 2304 |
| SL | Q2 | 568.0 | 585.0 | 1037.8 | 1444 | 1153.0 | 2012 |
| SL | Q3 | 580.0 | 701.6 | 1209.6 | 1398 | 1281.6 | 1978 |
| SL | Q4 | 565.0 | 660.7 | 1230.4 | 1780 | 1225.7 | 2345 |
| SL | Q5 | 564.0 | 574.0 | 952.2 | 1330 | 1138.0 | 1894 |
| SV | Q1 | 585.0 | 630.6 | 1071.4 | 1141 | 1215.6 | 1726 |
| SV | Q2 | 567.0 | 685.8 | 1258.8 | 1503 | 1252.8 | 2070 |
| SV | Q3 | 579.0 | 627.0 | 1009.6 | 1030 | 1206.0 | 1609 |
| SV | Q4 | 567.0 | 606.4 | 1109.4 | 1518 | 1173.4 | 2085 |
| SV | Q5 | 564.0 | 544.0 | 1038.0 | 1341 | 1108.0 | 1905 |

## D) Full Row-Level Token Table (All Responses)

A detailed per-response table (one row per response with run number and model) is exported to CSV:

- `insights/data/token_budget_response_details_estimated_range.csv` (3240 rows)

## E) CSV Exports

- `insights/data/token_budget_response_details_estimated_range.csv`
- `insights/data/token_budget_response_language_question_summary_estimated_range.csv`
- `insights/data/token_budget_response_model_summary_estimated_range.csv`
- `insights/data/token_budget_model_language_question_estimated.csv`
- `insights/data/token_budget_question_profile_estimated.csv`
- `insights/data/token_budget_model_output_budget_estimated.csv`
- `insights/data/token_budget_language_question_estimated.csv`
- `insights/data/token_budget_prompt_details_estimated.csv`

Note on model max length source:
- When a model was not present in the current `config.yaml` model list, `max_model_len` was filled with the common configured default and labeled as `assumed_from_config_default`.

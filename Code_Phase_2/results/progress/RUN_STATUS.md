# Run status

_Auto-generated 2026-08-02T10:16:02+00:00 by `scripts/run_status.py`; refreshed by the 15-minute auto-push._

**Total completed trials (Phase 2 checkpoint): 32,170**

| Focal model | Condition | Trials |
|---|---|---:|
| deepseek_primary | C1R_solo_reanswer | 1,500 |
| deepseek_primary | C1_smart_solo | 485 |
| deepseek_primary | C3H_two_smart_one_honest | 1,500 |
| deepseek_primary | C4H_one_smart_two_honest | 1,500 |
| deepseek_primary | C4_one_smart_two_dumb | 485 |
| deepseek_primary | C4split_one_wrong_one_correct | 1,500 |
| deepseek_primary | C5H_honest_with_confidence_filter | 300 |
| deepseek_primary | C5R_anchored_with_confidence_filter | 300 |
| gpt4o_mini | C1R_solo_reanswer | 1,500 |
| gpt4o_mini | C1_smart_solo | 1,500 |
| gpt4o_mini | C2_three_smart | 1,500 |
| gpt4o_mini | C3_two_smart_one_dumb | 1,500 |
| gpt4o_mini | C4_one_smart_two_dumb | 1,500 |
| het_deepseek | C2het_three_distinct_smart | 900 |
| sweep_gemma_3_27b | C1_smart_solo | 900 |
| sweep_gemma_3_27b | C2_three_smart | 900 |
| sweep_gemma_3_27b | C4_one_smart_two_dumb | 900 |
| sweep_gemma_3_4b_focal | C1_smart_solo | 900 |
| sweep_gemma_3_4b_focal | C2_three_smart | 900 |
| sweep_gemma_3_4b_focal | C4_one_smart_two_dumb | 900 |
| sweep_llama_3_1_70b | C1_smart_solo | 900 |
| sweep_llama_3_1_70b | C2_three_smart | 900 |
| sweep_llama_3_1_70b | C4_one_smart_two_dumb | 900 |
| sweep_llama_3_1_8b_focal | C1_smart_solo | 900 |
| sweep_llama_3_1_8b_focal | C2_three_smart | 900 |
| sweep_llama_3_1_8b_focal | C4_one_smart_two_dumb | 900 |
| sweep_mistral_small | C1_smart_solo | 900 |
| sweep_mistral_small | C2_three_smart | 900 |
| sweep_mistral_small | C4_one_smart_two_dumb | 900 |
| sweep_qwen_2_5_72b | C1_smart_solo | 900 |
| sweep_qwen_2_5_72b | C2_three_smart | 900 |
| sweep_qwen_2_5_72b | C4_one_smart_two_dumb | 900 |

## Last log lines

```
2026-08-01 08:50:02,118 [WARNING] platos_ship.metrics: No C5 data found for mitigation summary
2026-08-01 08:50:02,119 [INFO] platos_ship.run_all:   ok: per-condition metrics
2026-08-01 08:50:03,100 [INFO] platos_ship.corrected_gate: Corrected gate [phase1_C3C4_dumb_round1]: gap=-0.0036, AUROC=0.6206, decision=failed (n=12332)
2026-08-01 08:50:03,101 [INFO] platos_ship.corrected_gate: Corrected gate [E3_C5R_anchored_round0]: gap=0.99, AUROC=None, decision=passed (n=600)
2026-08-01 08:50:03,101 [INFO] platos_ship.corrected_gate: Corrected gate [E3_C5H_honest_round0]: gap=-0.0065, AUROC=0.5796, decision=failed (n=578)
2026-08-01 08:50:03,130 [INFO] platos_ship.run_all:   ok: corrected calibration gate
2026-08-01 08:50:03,690 [INFO] platos_ship.phase2_analyzer: Phase 2 analysis written: capability_sweep_summary.parquet, capability_sweep_analysis.json
2026-08-01 08:50:03,690 [INFO] platos_ship.phase2_analyzer: E4 sweep: Spearman(solo, C4-C1)=-0.09523809523809526 (n=8); Spearman(solo, C->I)=-0.21557272714438658 (n=8)
2026-08-01 08:50:03,690 [INFO] platos_ship.run_all:   ok: phase2 sweep/contrast analysis
2026-08-01 08:50:04,942 [INFO] platos_ship.phase2_analyzer: Phase 2 statistics: 16 paired McNemar tests -> phase2_statistical_tests.parquet
2026-08-01 08:50:04,943 [INFO] platos_ship.run_all:   ok: phase2 paired McNemar statistics
2026-08-01 08:50:04,943 [INFO] platos_ship.run_all: PHASE 2 RUN COMPLETE
```

## Recent API failures

```
2026-08-01 08:45:46,275 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0065', 'round': 1}
2026-08-01 08:45:58,588 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0068', 'round': 0}
2026-08-01 08:46:04,838 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0067', 'round': 1}
2026-08-01 08:46:41,391 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0072', 'round': 1}
2026-08-01 08:46:50,880 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0074', 'round': 0}
2026-08-01 08:46:58,900 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 2/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0074', 'round': 0}
```

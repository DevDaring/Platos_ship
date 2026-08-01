# Run status

_Auto-generated 2026-08-01T08:12:43+00:00 by `scripts/run_status.py`; refreshed by the 15-minute auto-push._

**Total completed trials (Phase 2 checkpoint): 31,520**

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
| het_deepseek | C2het_three_distinct_smart | 250 |
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
2026-08-01 08:12:37,571 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:12:38,037 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:12:38,424 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:12:38,608 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:12:38,941 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:12:39,040 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:12:40,550 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:12:41,592 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:12:42,129 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:12:42,518 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:12:43,147 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:12:43,199 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
```

## Recent API failures

```
2026-08-01 08:05:58,934 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0050', 'round': 1}
2026-08-01 08:06:05,032 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0052', 'round': 0}
2026-08-01 08:06:07,457 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 2/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0050', 'round': 1}
2026-08-01 08:06:08,058 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 4/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0051', 'round': 0}
2026-08-01 08:06:12,945 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 2/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0052', 'round': 0}
2026-08-01 08:06:17,776 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 3/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0050', 'round': 1}
```

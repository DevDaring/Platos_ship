# Run status

_Auto-generated 2026-08-01T08:42:48+00:00 by `scripts/run_status.py`; refreshed by the 15-minute auto-push._

**Total completed trials (Phase 2 checkpoint): 31,970**

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
| het_deepseek | C2het_three_distinct_smart | 700 |
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
2026-08-01 08:42:44,222 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:42:44,288 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:42:44,393 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:42:44,538 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:42:45,055 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:42:45,060 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:42:46,061 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:42:46,518 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:42:47,260 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:42:47,463 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:42:47,649 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 08:42:48,093 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
```

## Recent API failures

```
2026-08-01 08:41:37,284 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0042', 'round': 0}
2026-08-01 08:41:37,403 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0042', 'round': 0}
2026-08-01 08:42:00,956 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0045', 'round': 0}
2026-08-01 08:42:21,936 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0046', 'round': 1}
2026-08-01 08:42:29,332 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0045', 'round': 1}
2026-08-01 08:42:39,727 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'gsm8k_0048', 'round': 1}
```

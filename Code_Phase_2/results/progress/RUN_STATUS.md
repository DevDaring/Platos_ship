# Run status

_Auto-generated 2026-08-01T07:57:59+00:00 by `scripts/run_status.py`; refreshed by the 15-minute auto-push._

**Total completed trials (Phase 2 checkpoint): 31,270**

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
2026-08-01 07:57:56,678 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:57:57,082 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:57:57,626 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:57:57,783 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:57:57,889 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:57:58,101 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:57:58,429 [INFO] httpx: HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:57:58,582 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:57:58,697 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:57:58,828 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:57:59,608 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:57:59,624 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
```

## Recent API failures

```
2026-08-01 07:57:31,672 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0011', 'round': 1}
2026-08-01 07:57:40,155 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0012', 'round': 1}
2026-08-01 07:57:41,344 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0011', 'round': 1}
2026-08-01 07:57:42,330 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0013', 'round': 0}
2026-08-01 07:57:43,347 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0013', 'round': 0}
2026-08-01 07:57:45,333 [WARNING] platos_ship.api_failures: openrouter malformed response (attempt 1/6): malformed response: empty choices (payload: {'message': 'model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions', 'code': 400}), model=qwen/qwen-2.5-72b-instruct, meta={'question_id': 'mmlupro_0013', 'round': 0}
```

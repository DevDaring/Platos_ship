# Run status

_Auto-generated 2026-08-01T07:42:14+00:00 by `scripts/run_status.py`; refreshed by the 15-minute auto-push._

**Total completed trials (Phase 2 checkpoint): 30,720**

| Focal model | Condition | Trials |
|---|---|---:|
| deepseek_primary | C1R_solo_reanswer | 1,500 |
| deepseek_primary | C1_smart_solo | 485 |
| deepseek_primary | C3H_two_smart_one_honest | 1,500 |
| deepseek_primary | C4H_one_smart_two_honest | 1,500 |
| deepseek_primary | C4_one_smart_two_dumb | 485 |
| deepseek_primary | C4split_one_wrong_one_correct | 950 |
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
2026-08-01 07:42:08,253 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:42:08,912 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:42:09,071 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:42:09,466 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:42:09,907 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:42:10,082 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:42:10,273 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:42:12,231 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:42:12,334 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:42:12,384 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:42:12,619 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:42:13,743 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
```

## Recent API failures

```
2026-08-01 07:17:41,174 [WARNING] platos_ship.api_failures: openrouter error (attempt 1/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0052', 'round': 1}
2026-08-01 07:17:55,122 [WARNING] platos_ship.api_failures: openrouter error (attempt 1/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0054', 'round': 1}
2026-08-01 07:17:59,924 [WARNING] platos_ship.api_failures: openrouter error (attempt 2/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0054', 'round': 1}
2026-08-01 07:18:03,126 [WARNING] platos_ship.api_failures: openrouter error (attempt 1/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0054', 'round': 1}
2026-08-01 07:19:14,107 [WARNING] platos_ship.api_failures: openrouter error (attempt 1/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0060', 'round': 1}
2026-08-01 07:19:30,602 [WARNING] platos_ship.api_failures: openrouter error (attempt 1/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0060', 'round': 1}
```

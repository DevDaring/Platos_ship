# Run status

_Auto-generated 2026-08-01T07:12:10+00:00 by `scripts/run_status.py`; refreshed by the 15-minute auto-push._

**Total completed trials (Phase 2 checkpoint): 29,870**

| Focal model | Condition | Trials |
|---|---|---:|
| deepseek_primary | C1R_solo_reanswer | 1,500 |
| deepseek_primary | C1_smart_solo | 485 |
| deepseek_primary | C3H_two_smart_one_honest | 1,500 |
| deepseek_primary | C4H_one_smart_two_honest | 1,500 |
| deepseek_primary | C4_one_smart_two_dumb | 485 |
| deepseek_primary | C4split_one_wrong_one_correct | 100 |
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
2026-08-01 07:12:07,864 [INFO] platos_ship.agents.openai_compatible: gemma_3_4b_instruct: primary failed, recovered via fallback #1 (nanogpt)
2026-08-01 07:12:08,239 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:12:08,282 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:12:08,607 [INFO] httpx: HTTP Request: POST https://nano-gpt.com/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:12:08,610 [INFO] platos_ship.agents.openai_compatible: gemma_3_4b_instruct: primary failed, recovered via fallback #1 (nanogpt)
2026-08-01 07:12:08,725 [INFO] httpx: HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:12:08,923 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:12:09,126 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:12:09,424 [INFO] httpx: HTTP Request: POST https://api.mistral.ai/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:12:09,502 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-08-01 07:12:09,503 [INFO] openai._base_client: Retrying request to /chat/completions in 0.418136 seconds
2026-08-01 07:12:09,788 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
```

## Recent API failures

```
2026-08-01 07:12:00,612 [WARNING] platos_ship.api_failures: openrouter error (attempt 2/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0029', 'round': 1}
2026-08-01 07:12:00,686 [WARNING] platos_ship.api_failures: openrouter error (attempt 1/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0029', 'round': 1}
2026-08-01 07:12:01,621 [WARNING] platos_ship.api_failures: openrouter error (attempt 3/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0028', 'round': 1}
2026-08-01 07:12:04,631 [WARNING] platos_ship.api_failures: openrouter error (attempt 3/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0029', 'round': 1}
2026-08-01 07:12:05,211 [WARNING] platos_ship.api_failures: openrouter error (attempt 2/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0029', 'round': 1}
2026-08-01 07:12:06,684 [WARNING] platos_ship.api_failures: openrouter error (attempt 3/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0029', 'round': 1}
```

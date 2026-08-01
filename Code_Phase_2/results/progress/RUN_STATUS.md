# Run status

_Auto-generated 2026-08-01T07:11:19+00:00 by `scripts/run_status.py`; refreshed by the 15-minute auto-push._

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
2026-08-01 07:11:18,419 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:11:18,420 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-08-01 07:11:18,422 [INFO] openai._base_client: Retrying request to /chat/completions in 0.885814 seconds
2026-08-01 07:11:18,787 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:11:18,989 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-08-01 07:11:18,990 [INFO] openai._base_client: Retrying request to /chat/completions in 0.430203 seconds
2026-08-01 07:11:19,153 [INFO] httpx: HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-08-01 07:11:19,410 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-08-01 07:11:19,412 [WARNING] platos_ship.api_failures: openrouter error (attempt 3/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0027', 'round': 1}
2026-08-01 07:11:19,412 [ERROR] platos_ship.agents.openai_compatible: openrouter call failed after 3 attempts: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'DeepInfra', 'is_byok': False, 'provider_error_code': 'engine_overloaded', 'limit_source': 'upstream_provider_shared_pool', 'remedy_hint': 'Retry shortly, add your own provider key (https://openrouter.ai/settings/integrations), or route to another provider with provider routing: https://openrouter.ai/docs/features/provider-routing'}}, 'user_id': 'user_2wfU5bcUfYB3u5JSP1fIkfqz9Ts'}
2026-08-01 07:11:19,514 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-08-01 07:11:19,515 [INFO] openai._base_client: Retrying request to /chat/completions in 0.772619 seconds
```

## Recent API failures

```
2026-08-01 07:11:10,727 [WARNING] platos_ship.api_failures: openrouter error (attempt 3/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0026', 'round': 1}
2026-08-01 07:11:12,844 [WARNING] platos_ship.api_failures: openrouter error (attempt 2/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0027', 'round': 1}
2026-08-01 07:11:14,003 [WARNING] platos_ship.api_failures: openrouter error (attempt 2/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0027', 'round': 1}
2026-08-01 07:11:14,343 [WARNING] platos_ship.api_failures: openrouter error (attempt 1/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0027', 'round': 1}
2026-08-01 07:11:18,238 [WARNING] platos_ship.api_failures: openrouter error (attempt 2/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0027', 'round': 1}
2026-08-01 07:11:19,412 [WARNING] platos_ship.api_failures: openrouter error (attempt 3/3): status=429, RateLimitError: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'google/gemma-3-4b-it is temporarily rate-limited upstream. Please retry shortly, or add your own ke, meta={'question_id': 'mmlupro_0027', 'round': 1}
```

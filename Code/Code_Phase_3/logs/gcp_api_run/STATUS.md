# Phase 3 API run status

- updated: 2026-09-24 20:30:33 UTC
- host: platos-api-run.c.silicon-guru-472717-q9.internal

```
2026-09-24T13:17:47Z prepare: start
2026-09-24T14:15:57Z prepare attempt=1 exit=0
2026-09-24T15:33:40Z gpt4o_mini attempt=1 exit=0
2026-09-24T15:34:57Z deepseek_primary attempt=1 exit=0
2026-09-24T16:38:55Z sweep_llama_3_1_70b attempt=1 exit=0
2026-09-24T16:44:09Z sweep_mistral_small attempt=1 exit=0
2026-09-24T17:15:10Z sweep_qwen_2_5_72b attempt=1 exit=0
2026-09-24T18:30:44Z sweep_llama_3_1_8b_focal attempt=1 exit=0
2026-09-24T19:47:33Z sweep_gemma_3_27b attempt=1 exit=0
2026-09-24T20:21:26Z sweep_gemma_3_4b_focal attempt=1 exit=3
2026-09-24T20:29:38Z sweep_gemma_3_4b_focal attempt=2 exit=0
2026-09-24T20:29:38Z all 8 shards complete; merging
2026-09-24T20:30:32Z analysis failed; see logs/launcher/analyse.log
```

Files over 95m not pushed (kept on the VM):


Units written per shard (revision rows, part files):

- deepseek_primary: 0 revision part-files
- gpt4o_mini: 0 revision part-files
- sweep_gemma_3_27b: 0 revision part-files
- sweep_gemma_3_4b_focal: 0 revision part-files
- sweep_llama_3_1_70b: 0 revision part-files
- sweep_llama_3_1_8b_focal: 0 revision part-files
- sweep_mistral_small: 0 revision part-files
- sweep_qwen_2_5_72b: 0 revision part-files

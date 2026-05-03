# Implementation Prompt — Capability-Asymmetric LLM Debate Study (v2)

> **Audience:** Coding agent or developer implementing the full experiment.
> **Read this entire document before writing any code.** The dry-run discipline in §11 and the calibration gate in §12 are non-negotiable.
> **Change log vs v1:** Added Condition C5 (confidence-weighted debate mitigation). Added §12 calibration gate that decides whether C5 runs at all. Updated dataset, configuration, output schemas, metrics, and statistical-analysis sections to carry C5 through. Section 19 final checklist updated.

---

## 0. Project Goal in One Paragraph

Build a reproducible pipeline that runs five experimental conditions on a fixed dataset of reasoning questions. In each condition, one or more "smart" Large Language Model agents (called via API) and zero or more "dumb" Large Language Model agents (loaded locally on a single GPU) answer each question independently, then engage in one round of debate where each agent sees the other agents' first answers, then commit to a final answer. The pipeline measures whether the smart agent's accuracy degrades when capability-weaker peers are present, whether the smart agent flips initially correct answers under peer pressure, and whether a confidence-weighted aggregation rule recovers part of the lost accuracy. Outputs are structured tables and statistical summaries suitable for a short journal communication.

---

## 1. Repository Layout

Create the following directory and file structure exactly:

```
project_root/
├── .env                              # Secrets — never commit
├── .env.example                      # Template with empty values
├── README.md
├── requirements.txt
├── config/
│   ├── paths.yaml                    # All filesystem paths
│   ├── models.yaml                   # Model registry and parameters
│   └── experiment.yaml               # Conditions, sample sizes, seeds
├── data/
│   ├── raw/                          # Downloaded benchmark data
│   ├── processed/
│   │   ├── question_pool.parquet     # Sampled and stratified questions
│   │   └── dumb_personas.parquet     # Generated and validated personas
│   └── outputs/
│       ├── trial_log.parquet         # One row per agent response
│       ├── final_answers.parquet     # One row per (question, condition, trial, focal_smart_agent)
│       ├── metrics_summary.parquet
│       ├── statistical_tests.parquet
│       ├── calibration_gate_report.parquet  # Decides whether C5 runs
│       └── mitigation_summary.parquet       # C5-specific summary
├── logs/
│   ├── pipeline.log                  # Structured logs
│   └── api_failures.log              # Separate log for API errors and retries
├── src/
│   ├── environment_check.py
│   ├── dataset_builder.py
│   ├── persona_generator.py
│   ├── persona_validator.py
│   ├── agent_wrappers/
│   │   ├── base_agent.py
│   │   ├── deepseek_agent.py
│   │   ├── openrouter_agent.py
│   │   ├── local_huggingface_agent.py
│   │   └── judge_agent.py
│   ├── debate_protocol.py
│   ├── confidence_weighted_protocol.py
│   ├── calibration_gate.py
│   ├── trial_runner.py
│   ├── metrics_calculator.py
│   ├── statistical_analyzer.py
│   └── pipeline_orchestrator.py
├── scripts/
│   ├── run_environment_check.sh
│   ├── run_dry_run.sh
│   ├── run_main_experiment.sh        # Conditions C1–C4 plus cross-model replication
│   ├── run_calibration_gate.sh       # Decides whether to run C5
│   └── run_mitigation_experiment.sh  # Condition C5, only after gate passes
└── tests/
    └── dry_run_assertions.py
```

---

## 2. Environment Setup

### 2.1 Operating System and Hardware Requirements

The compiled flash-attention wheel that will be installed expects:

- **Operating system:** Linux x86_64
- **Recommended distribution:** Ubuntu 24.04 LTS (ships with Python 3.12 natively).
- **Acceptable alternative:** Ubuntu 22.04 LTS, but Python 3.12 must be installed manually via the deadsnakes Personal Package Archive.
- **Do not use:** Ubuntu 20.04, Debian 11, or any non-glibc distribution. The flash-attention wheel will refuse to load.
- **GPU:** NVIDIA A100 with 40 GB VRAM (or larger). Verify presence of the device with `nvidia-smi`.
- **CUDA toolkit:** 12.4 must be available on the system. The PyTorch wheels installed in §2.4 are compiled against CUDA 12.4.
- **Python interpreter:** Exactly Python 3.12. The flash-attention wheel filename contains `cp312-cp312` which means it links against CPython 3.12 ABI. Python 3.10, 3.11, or 3.13 will fail.

### 2.2 Terraform Verification Step (mandatory before any installation)

Before running any installation script:

1. Locate any Terraform configuration files in the project or surrounding repository (look for `*.tf`, `terraform.tfvars`, or a `terraform/` directory).
2. From the Terraform configuration, extract and confirm:
   - The `image_family` or `image` field on the GCP virtual machine resource. It must resolve to an Ubuntu 22.04 or Ubuntu 24.04 LTS image. If it points to a deep-learning VM image, confirm that image's base distribution.
   - The `machine_type`. It must be `a2-highgpu-1g` or higher (this provisions one A100 40 GB GPU).
   - The `accelerator` block. It must contain `type = "nvidia-tesla-a100"` and `count = 1`.
   - The startup script or metadata. Confirm that NVIDIA driver installation is included.
3. If any of the above does not match, **stop and report to the user**. Do not silently proceed.
4. If no Terraform files exist, write a short `terraform_compatibility_report.md` to the repository root stating that no Terraform configuration was found and that compatibility was verified manually via `nvidia-smi` and `lsb_release -a`.

### 2.3 Python Environment Bootstrap

Create a Python 3.12 virtual environment at `~/venv_debate_study`:

```
python3.12 -m venv ~/venv_debate_study
source ~/venv_debate_study/bin/activate
```

All subsequent commands and the running pipeline must use this interpreter.

### 2.4 Installation Commands (use exactly as provided by user)

Run the following block. Do not modify versions. Do not add intermediate sub-commands. If any single command fails, stop and report.

```
python3 -m pip install --upgrade pip setuptools wheel \
  && python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
  && python3 -m pip install numpy\<2.0 transformers==4.46.0 accelerate==0.34.0 datasets==2.16.0 bitsandbytes==0.46.1 pandas==2.2.2 tqdm==4.65.0 python-dotenv==1.0.0 requests==2.31.0 sentencepiece==0.2.0 protobuf==4.25.0 \
  && wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl -O /tmp/flash_attn.whl \
  && python3 -m pip install --no-deps /tmp/flash_attn.whl
```

After install, run the verification snippet provided by the user (the inline Python that prints torch, bitsandbytes, and flash_attn versions). All three must print without `ImportError`. Capture this output to `logs/install_verification.log`.

### 2.5 Additional Pinned Dependencies

After the user-provided installation succeeds, install one further set of small libraries needed for the pipeline:

```
python3 -m pip install pyyaml==6.0.2 scipy==1.13.1 statsmodels==0.14.2 sentence-transformers==3.0.1 openai==1.43.0 pyarrow==17.0.0 jsonschema==4.23.0
```

The `openai` package is used to talk to both DeepSeek and OpenRouter, since both expose OpenAI-compatible chat completion endpoints.

### 2.6 Critical Compatibility Note on Gemma 3 (read carefully)

The user has specified `google/gemma-3-4b-it` as one of the dumb models. Gemma 3 architecture support was added to `transformers` only in versions later than 4.46.0. The pinned `transformers==4.46.0` from §2.4 will **not** load Gemma 3.

**Resolution path the coding agent must follow, in this exact order:**

1. First attempt: load `google/gemma-3-4b-it` with the pinned `transformers==4.46.0`. Wrap the load in a try/except.
2. If the load fails with `KeyError`, `ValueError` mentioning unknown architecture, or any unrecognized config error, do **not** silently swap models. Instead:
   - Upgrade transformers to `transformers==4.50.3` (the lowest version with stable Gemma 3 support that is also compatible with `torch==2.5.1`).
   - Re-attempt the load. Re-run the post-install verification snippet from §2.4 to confirm flash-attention still imports after the upgrade.
3. If the upgraded transformers still fails or breaks flash-attention, fall back to `google/gemma-2-2b-it` and write a clearly visible warning to `logs/pipeline.log` and to the final report. The fallback must be auto-detected, not hardcoded.

Document whichever path was taken in the experiment metadata file (§13.3).

---

## 3. Configuration and Secrets

### 3.1 The `.env` File

The user maintains a `.env` file at the repository root with the following variables. Do not commit this file. Do not echo its values to standard output. Provide a `.env.example` with empty values as a template.

Required variables:

```
DEEPSEEK_API_KEY=
DEEPSEEK_API_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_PRIMARY_MODEL_NAME=
DEEPSEEK_JUDGE_MODEL_NAME=

OPENROUTER_API_KEY=
OPENROUTER_API_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_PRIMARY_MODEL_NAME=openai/gpt-4o-mini

HUGGINGFACE_TOKEN=

RANDOM_SEED=20260502
```

`DEEPSEEK_PRIMARY_MODEL_NAME` is the latest DeepSeek model on the DeepSeek platform that will play the smart agent role.

`DEEPSEEK_JUDGE_MODEL_NAME` is the older or cheaper DeepSeek variant used only to extract structured answers from raw debate transcripts. The coding agent should query the DeepSeek API model listing endpoint at startup and write all available model names to `logs/available_deepseek_models.log` so the user can confirm which model name to put in each variable.

### 3.2 Path Configuration

Create `config/paths.yaml` with all filesystem paths. Every path used in the codebase must come from this file. No hardcoded paths anywhere in source code.

Required keys:

```yaml
data_root: ./data
raw_data_directory: ./data/raw
processed_data_directory: ./data/processed
output_directory: ./data/outputs
logs_directory: ./logs
question_pool_file: ./data/processed/question_pool.parquet
dumb_personas_file: ./data/processed/dumb_personas.parquet
trial_log_file: ./data/outputs/trial_log.parquet
final_answers_file: ./data/outputs/final_answers.parquet
metrics_summary_file: ./data/outputs/metrics_summary.parquet
statistical_tests_file: ./data/outputs/statistical_tests.parquet
calibration_gate_report_file: ./data/outputs/calibration_gate_report.parquet
mitigation_summary_file: ./data/outputs/mitigation_summary.parquet
experiment_metadata_file: ./data/outputs/experiment_metadata.json
huggingface_cache_directory: /home/${USER}/.cache/huggingface
```

The pipeline must call a path-resolution function that expands user variables and asserts each directory exists or can be created. Run this assertion at startup, before any other operation. Log every resolved path to `logs/pipeline.log`.

### 3.3 Model Registry

Create `config/models.yaml`:

```yaml
smart_agents:
  deepseek_primary:
    provider: deepseek
    model_name_env_variable: DEEPSEEK_PRIMARY_MODEL_NAME
    temperature: 0.7
    max_output_tokens: 600
    request_timeout_seconds: 120
    maximum_retry_attempts: 5
  openrouter_gpt4o_mini:
    provider: openrouter
    model_name_env_variable: OPENROUTER_PRIMARY_MODEL_NAME
    temperature: 0.7
    max_output_tokens: 600
    request_timeout_seconds: 120
    maximum_retry_attempts: 5

dumb_agents:
  llama_3_1_8b_instruct:
    provider: local_huggingface
    huggingface_repository: meta-llama/Llama-3.1-8B-Instruct
    torch_dtype: bfloat16
    attention_implementation: flash_attention_2
    temperature: 0.9
    max_output_tokens: 350
    device: cuda:0
  gemma_3_4b_instruct:
    provider: local_huggingface
    huggingface_repository: google/gemma-3-4b-it
    fallback_huggingface_repository: google/gemma-2-2b-it
    torch_dtype: bfloat16
    attention_implementation: flash_attention_2
    temperature: 0.9
    max_output_tokens: 350
    device: cuda:0

judge_agent:
  provider: deepseek
  model_name_env_variable: DEEPSEEK_JUDGE_MODEL_NAME
  temperature: 0.0
  max_output_tokens: 50
  request_timeout_seconds: 60
  maximum_retry_attempts: 3

bit_precision_policy:
  policy_description: All locally loaded models use bfloat16 weights. Smart and judge agents are API services and use server-side precision.
  local_model_torch_dtype: bfloat16
  local_model_quantization: none
```

### 3.4 Experiment Configuration

Create `config/experiment.yaml`:

```yaml
random_seed: 20260502
total_questions_in_pool: 300
mmlu_pro_question_count: 200
gsm8k_question_count: 100
mmlu_pro_subject_count: 10
questions_per_subject: 20
trials_per_question_main_conditions: 5
trials_per_question_mitigation_condition: 3
debate_rounds: 1
peer_message_ordering: randomized_per_trial

reasoning_styles:
  - surface_keyword_match
  - false_analogy
  - overconfident_assertion
  - misapplied_rule

dumb_persona_variants_per_question: 5

main_conditions:
  C1_smart_solo:
    smart_agent_count: 1
    dumb_agent_count: 0
    aggregation_rule: none
    description: Single smart agent answers alone, no debate
  C2_three_smart:
    smart_agent_count: 3
    dumb_agent_count: 0
    aggregation_rule: standard_debate
    description: Three smart agents debate, capability-homogeneous control
  C3_two_smart_one_dumb:
    smart_agent_count: 2
    dumb_agent_count: 1
    aggregation_rule: standard_debate
    description: Two smart plus one dumb, the Plato condition
  C4_one_smart_two_dumb:
    smart_agent_count: 1
    dumb_agent_count: 2
    aggregation_rule: standard_debate
    description: One smart plus two dumb, capability collapse condition

mitigation_condition:
  C5_one_smart_two_dumb_confidence_weighted:
    smart_agent_count: 1
    dumb_agent_count: 2
    aggregation_rule: confidence_weighted
    description: Same composition as C4 but with confidence-weighted peer aggregation
    questions_subset_size: 100
    questions_sampling_strategy: stratified_random_from_full_pool
    trials_per_question: 3
    confidence_threshold_for_filtering: 60
    confidence_weight_function: linear_above_threshold
    runs_only_if_calibration_gate_passes: true

focal_smart_agent_assignment: deepseek_primary
cross_model_validation_subset_size: 50
cross_model_validation_focal_agent: openrouter_gpt4o_mini

answer_extraction_strategy: regex_first_then_judge_fallback
parse_failure_retry_attempts: 1

calibration_gate:
  description: Decides whether C5 mitigation experiment is run.
  condition_subset_to_analyse:
    - C3_two_smart_one_dumb
    - C4_one_smart_two_dumb
  agent_role_to_analyse: dumb
  precondition_metric_name: probability_high_confidence_given_wrong_answer
  high_confidence_threshold: 60
  precondition_metric_threshold: 0.40
  decision_rule: run_C5_if_precondition_metric_is_at_or_above_threshold
```

---

## 4. Dataset Construction

### 4.1 Source Datasets

Use the `datasets` Python library to download:

1. `TIGER-Lab/MMLU-Pro` — multiple-choice questions across 14 subjects with single correct answers and 10 answer options each.
2. `gsm8k` (the `main` split) — grade-school math word problems with numeric answers.

Cache both at `data/raw/`. Confirm row counts match expectations before sampling.

### 4.2 Stratified Sampling Plan

For MMLU-Pro:

1. Select 10 subject categories from the 14 available. Use these specifically (in priority order, dropping subjects with fewer than 200 available items): `mathematics`, `physics`, `chemistry`, `biology`, `computer science`, `economics`, `history`, `law`, `philosophy`, `psychology`.
2. From each selected subject, sample exactly 20 questions using a deterministic seed (`RANDOM_SEED` from `.env`).
3. Within each subject, attempt 50/50 difficulty stratification: half the items where a small probe model (Llama 3.1 8B) gets the answer correct in zero-shot mode, half where it gets it wrong. If the 50/50 cannot be achieved (insufficient items in one stratum), document the actual ratio in experiment metadata and proceed.

For GSM8K:
1. Sample 100 items from the test split with the same `RANDOM_SEED`.
2. Apply the same difficulty probe procedure with Llama 3.1 8B as the prober. GSM8K has higher variance — accept a 60/40 stratification split if 50/50 is infeasible.

### 4.3 Question Pool Output Schema

Write to `data/processed/question_pool.parquet`. Column names must use full forms exactly as listed below — no abbreviations:

| Column Name | Data Type | Description |
|---|---|---|
| `question_identifier` | string | Unique stable identifier, format `mmlupro_<index>` or `gsm8k_<index>` |
| `source_dataset` | string | One of `mmlu_pro` or `gsm8k` |
| `subject_category` | string | Subject for MMLU-Pro, or `mathematics_word_problem` for GSM8K |
| `question_text` | string | Full question prompt as it will be shown to agents |
| `answer_options` | list of string or null | The 10 multiple-choice options for MMLU-Pro, null for GSM8K |
| `correct_answer` | string | For MMLU-Pro, the correct option letter; for GSM8K, the numeric answer as a string |
| `correct_answer_full_text` | string | Human-readable expanded correct answer |
| `wrong_answer_pool` | list of string | All incorrect options for MMLU-Pro, or auto-generated plausible wrong numerics for GSM8K |
| `difficulty_stratum` | string | One of `probe_correct` or `probe_incorrect` |
| `random_seed_used` | integer | The seed used during sampling |
| `included_in_mitigation_subset` | boolean | True for the 100 items selected for C5 |

### 4.4 GSM8K Wrong Answer Generation

GSM8K does not provide wrong-answer options. Generate `wrong_answer_pool` for each item using these deterministic perturbations of the correct numeric answer: `correct + 1`, `correct - 1`, `correct * 2`, `correct / 2` (rounded to nearest integer if originally integer), `correct + 10`. Filter any duplicates and any that equal the correct answer. Store as strings.

### 4.5 Mitigation Subset Selection

Sample 100 questions from the full 300-question pool for C5. Stratify proportionally: 67 from MMLU-Pro (so roughly 7 per subject across the 10 subjects) and 33 from GSM8K. Use a deterministic seed derived from `RANDOM_SEED` (e.g., `RANDOM_SEED + 1`). Mark these rows in `question_pool.parquet` with `included_in_mitigation_subset = True`.

---

## 5. Dumb Persona Generation

### 5.1 Persona Template

For every question and every reasoning style, build a persona prompt that the dumb model will generate text from. The template:

```
SYSTEM: You are Agent_X, a confident respondent in a group discussion.

USER: Question: {question_text}

{answer_options_block_if_present}

Your assigned position: The answer is {chosen_wrong_answer_full_text}.

Write a 2 to 3 sentence response defending this answer using a {reasoning_style_human_readable} style of reasoning. Be confident. Do not hedge. End with the line:
Final answer: {chosen_wrong_answer_letter_or_value}
```

The four reasoning style human-readable descriptions:

- `surface_keyword_match` → "match keywords from the question to your answer without checking deeper relevance"
- `false_analogy` → "invoke a superficially similar but logically irrelevant case"
- `overconfident_assertion` → "assert your answer with confidence and minimal justification"
- `misapplied_rule` → "cite a real principle or rule but apply it incorrectly to this question"

### 5.2 Generation Procedure

For each question in the question pool, generate 5 dumb-persona variants:

1. Sample one wrong answer uniformly from `wrong_answer_pool`.
2. Sample one reasoning style uniformly from the four options.
3. Render the template above.
4. Generate the persona response by calling **Llama 3.1 8B Instruct locally** (do not use the smart API for this — it is wasteful and unrealistic). Temperature 0.9. Maximum 350 output tokens.
5. Persist the raw generation along with all metadata.

### 5.3 Persona Validation Filter

After generation, automatically discard any persona where:

1. The generated text does not contain the literal string `Final answer:` (case-insensitive match).
2. The extracted final answer does not equal the assigned wrong answer (the model drifted to a different answer).
3. The generated text contains hedging phrases above a threshold count (`unsure`, `not certain`, `might be wrong`, `actually`, `correction`) — keep this regex strict; persona is supposed to sound confident.
4. The generated text is shorter than 30 characters or longer than 1500 characters.

For each discarded persona, regenerate up to 3 attempts. Log discard reasons and final retention rate. Write the validation report to `logs/persona_validation_report.txt`.

### 5.4 Persona Output Schema

Write to `data/processed/dumb_personas.parquet`:

| Column Name | Data Type | Description |
|---|---|---|
| `persona_identifier` | string | Format `<question_identifier>_persona_<variant_index>` |
| `question_identifier` | string | Foreign key to question pool |
| `persona_variant_index` | integer | 0 through 4 |
| `assigned_wrong_answer_letter_or_value` | string | The wrong answer this persona advocates |
| `assigned_wrong_answer_full_text` | string | Human-readable form |
| `reasoning_style_label` | string | One of the four style labels |
| `generated_persona_text` | string | The full text the persona will inject into debate |
| `generation_temperature` | float | 0.9 |
| `generator_model_name` | string | `meta-llama/Llama-3.1-8B-Instruct` |
| `validation_pass_status` | string | `passed` |
| `regeneration_attempts_used` | integer | 0 through 3 |

---

## 6. Agent Wrappers

### 6.1 Common Interface

Every agent wrapper exposes a single function with this exact signature:

```
generate_response(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    maximum_output_tokens: int,
    request_metadata: dict
) -> AgentResponse
```

The `AgentResponse` object must carry: `raw_text_output`, `wall_clock_latency_seconds`, `total_input_tokens`, `total_output_tokens`, `model_name_returned_by_provider`, `error_status`, `retry_attempts_used`.

### 6.2 Smart Agent — DeepSeek

Use the `openai` Python client pointed at `DEEPSEEK_API_BASE_URL` with `DEEPSEEK_API_KEY`. Use the model name from `DEEPSEEK_PRIMARY_MODEL_NAME`. Implement exponential backoff retry on 429, 500, 502, 503, 504 status codes. Maximum 5 retries with delays 2, 4, 8, 16, 32 seconds plus 0–1 second jitter. Log every retry to `logs/api_failures.log`.

### 6.3 Smart Agent — OpenRouter GPT-4o-mini

Same `openai` client, base URL `OPENROUTER_API_BASE_URL`, key `OPENROUTER_API_KEY`, model `openai/gpt-4o-mini`. Add the OpenRouter-required HTTP headers `HTTP-Referer` and `X-Title` to identify the application. Same retry policy.

### 6.4 Dumb Agent — Local HuggingFace

Use `transformers.AutoModelForCausalLM.from_pretrained` with:
- `torch_dtype=torch.bfloat16`
- `attn_implementation="flash_attention_2"`
- `device_map="cuda:0"`
- `token=HUGGINGFACE_TOKEN`

Both Llama 3.1 8B and Gemma must be loaded once at process start and kept in memory throughout the experiment. Do not reload per call. Free CUDA cache between conditions only if VRAM diagnostics indicate fragmentation above 80%.

For Gemma 3, follow the §2.6 fallback procedure exactly. If the fallback is taken, set `gemma_actually_used_repository` in the experiment metadata file.

Generation uses `model.generate` with `do_sample=True`, `temperature` and `max_new_tokens` from `models.yaml`, plus a stop sequence on the literal string `\nFinal answer:` followed by one line, then end-of-sequence.

### 6.5 Bit Precision Policy

Both local models load in **bfloat16**. No quantization. No `bitsandbytes` 8-bit or 4-bit. The pinned `bitsandbytes==0.46.1` from the install is present only to satisfy library availability checks — it is not used. Document this explicitly in `experiment_metadata.json`.

### 6.6 Judge Agent for Answer Extraction

The judge runs only when regex extraction of the final answer from a debate transcript fails or returns ambiguous output. The judge prompt:

```
SYSTEM: You extract a single final answer from a model's response. Output only the answer, nothing else.

USER: The question was: {question_text}

The valid answer options were: {answer_options_or_numeric_format}

The model's response was:
{raw_text_to_extract_from}

Output the single answer the model committed to, in the exact format the question expects (a single capital letter for multiple choice, or a number for math). If the response is genuinely ambiguous, output the literal token UNPARSEABLE.
```

Judge calls use temperature 0.0, maximum 50 output tokens. Wrap in retry like other API agents.

### 6.7 Confidence Score Extraction (used in C5 only)

For C5, every agent (smart and dumb) is required to emit a self-reported confidence integer from 0 to 100 alongside its final answer. The required output format the agent is instructed to use:

```
Final answer: <X>
Confidence: <integer 0 to 100>
```

The pipeline parses this with a strict regex. If the confidence line is missing, the trial response is retried once with a stricter prompt; on second failure the confidence is recorded as `null` and the trial response is excluded from confidence-weighted aggregation but still logged.

The exact same confidence extraction is performed on the C3 and C4 trials retroactively (see §12). For these conditions the extraction is best-effort: the original C3/C4 prompts did not strictly require a confidence line, so only responses that volunteered one are scored. **Important:** to make the calibration gate analysis valid, the C3 and C4 prompts must ask agents to also emit `Confidence: <integer>` from the start of the run, not retroactively. This is why the confidence-line instruction is added to the universal Round 0 and Round 1 prompts in §7.2 below.

---

## 7. Experimental Conditions and Trial Protocol

### 7.1 Condition Definitions

The five conditions are defined in `config/experiment.yaml`. Repeat:

| Condition Identifier | Smart Agents | Dumb Agents | Aggregation Rule | Notes |
|---|---|---|---|---|
| `C1_smart_solo` | 1 (focal) | 0 | none | No debate, just initial answer |
| `C2_three_smart` | 3 (all DeepSeek) | 0 | standard_debate | Capability-homogeneous control |
| `C3_two_smart_one_dumb` | 2 | 1 | standard_debate | Plato condition |
| `C4_one_smart_two_dumb` | 1 (focal) | 2 (one Llama, one Gemma) | standard_debate | Collapse condition |
| `C5_one_smart_two_dumb_confidence_weighted` | 1 (focal) | 2 (one Llama, one Gemma) | confidence_weighted | Mitigation; runs only after calibration gate passes; 100-question subset, 3 trials per question |

For C2, the three smart agents are three independent calls to the same focal smart model with three different system-prompt agent identifiers (`Agent_Alpha`, `Agent_Beta`, `Agent_Gamma`). They are sampled at temperature 0.7 to produce genuine variation.

For C3, the dumb agent is randomly chosen per trial from {Llama 3.1 8B, Gemma 3 4B} with 50/50 probability and the choice logged.

For C4 and C5, one Llama and one Gemma always co-occur.

### 7.2 Trial Protocol — Round 0 (Independent)

For each (question, condition, trial) triplet:

1. Construct an independent prompt for each agent in the condition. The prompt contains the question and a structured-output instruction. The instruction (used universally across all conditions C1 through C5) is:

```
Provide your reasoning in 2 to 3 sentences. End your response with these two lines, and nothing after them:
Final answer: <your answer>
Confidence: <integer from 0 to 100 representing how confident you are in your final answer>
```

2. Each agent produces an independent response.
3. Parse and store each response's final answer and confidence using the regex extractor; on parse failure for the answer, invoke the judge. On parse failure for confidence only, record `confidence = null` and continue.

### 7.3 Trial Protocol — Round 1 (Peer Exposure, Standard Debate)

This is the protocol for C2, C3, and C4. C5 uses the modified protocol in §7.5.

1. Build a debate context for each agent. The context is the original question followed by a labeled block of the *other* agents' Round 0 responses, each prefixed with their agent identifier (e.g., `Agent_Alpha said: ...`). The instruction is: *"Review the responses above. State your final answer with reasoning. You may agree or disagree. End with: Final answer: <X> and Confidence: <integer 0 to 100>."*
2. The order of peer responses in the context is **randomized per trial**, with a per-trial seed derived from `RANDOM_SEED` and the trial index. This randomization neutralizes the ordering effect documented in prior work.
3. Each agent produces a Round 1 response.
4. Parse and store final answers and confidence as in Round 0.

### 7.4 Focal Smart Agent

In every condition, designate exactly one smart agent as the **focal smart agent** for metrics purposes. This is the agent whose flip rate and accuracy are tracked. The focal agent is `deepseek_primary` for the main experiment. The cross-model validation subset of 50 questions repeats conditions C1 through C4 with `openrouter_gpt4o_mini` as the focal agent.

For C2 (three smart), the focal is one specific instance, designated as `Agent_Alpha`. The other two are non-focal but their responses still appear in the debate context.

For C5, the focal smart agent is `deepseek_primary`. C5 is not run with `openrouter_gpt4o_mini` as focal — the cross-model question for the mitigation is a separate future paper.

### 7.5 Trial Protocol — Round 1 (Confidence-Weighted, C5 only)

C5 differs from C4 only in how peer responses are presented to the focal smart agent in Round 1. Round 0 is identical.

1. Collect all Round 0 responses from peers (the two dumb agents).
2. Read each peer's self-reported confidence integer.
3. Apply the confidence weight function defined in `config/experiment.yaml`. The default function is `linear_above_threshold`:
   - If a peer's confidence is below `confidence_threshold_for_filtering` (default 60), the peer's message is **excluded** from the debate context entirely.
   - If a peer's confidence is at or above the threshold, the peer's message is included with a visible confidence label: `Agent_Llama (confidence 78) said: ...`.
4. The instruction prefix to the focal agent is augmented:

```
Below are responses from other agents. Each response is labelled with the agent's self-reported confidence on a scale of 0 to 100. Treat low-confidence responses with appropriate scepticism. Some responses may have been filtered out because the responding agent indicated low confidence in its own answer.

{filtered_peer_messages_with_confidence_labels}

Provide your final answer. End with: Final answer: <X> and Confidence: <integer 0 to 100>.
```

5. Edge case: if both dumb peers have confidence below the threshold, the focal smart agent receives an empty peer-messages block with the explanatory note `(No peer responses met the confidence threshold; respond based on your own reasoning.)`. This collapses the trial back to a solo-equivalent setup; the trial is still logged and counted.
6. Edge case: if either peer's confidence is `null` (parse failure), treat that peer as below threshold and exclude.

The dumb agents in C5 do **not** see filtered peer messages; only the focal smart agent's debate context is filtered. This is by design — the experimental hypothesis is about whether the focal agent's behaviour can be rescued, not whether the dumb agents change.

### 7.6 Trial Replication

Each (question, condition) is repeated `trials_per_question_main_conditions = 5` times for conditions C1 through C4, and `trials_per_question_mitigation_condition = 3` times for C5. Each trial uses a different randomly-sampled dumb persona variant (from the 5 generated for that question) and a different per-trial random seed. This provides within-question variance for paired statistical tests.

---

## 8. Dry Run Requirements (mandatory before full execution)

The user has been explicit: **dry run before full run**. The pipeline must support and pass a dry-run mode. Implement as `scripts/run_dry_run.sh`.

### 8.1 Pre-Flight Checks (run first, abort on any failure)

1. **Environment check.** Confirm Python 3.12 is the active interpreter. Confirm `torch.cuda.is_available()` returns True. Confirm `nvidia-smi` shows an A100 with at least 35 GB free. Confirm `flash_attn` imports without error.
2. **Path check.** Read `config/paths.yaml`. For each path, confirm the directory exists or can be created with the current user's permissions. Write a one-line confirmation per path to `logs/pipeline.log`.
3. **API key check.** For each API provider (DeepSeek, OpenRouter, HuggingFace):
   - DeepSeek: call the model listing endpoint. Confirm both `DEEPSEEK_PRIMARY_MODEL_NAME` and `DEEPSEEK_JUDGE_MODEL_NAME` appear in the returned list. Log all available model names.
   - OpenRouter: call the models listing endpoint. Confirm `openai/gpt-4o-mini` is available.
   - HuggingFace: call `huggingface_hub.whoami()`. Confirm the token is valid and the user has accepted the license for `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-3-4b-it` (programmatically check by attempting `model_info` for each repo).
4. **Column mapping check.** Load a 5-row sample of MMLU-Pro and a 5-row sample of GSM8K. Confirm every column expected by the dataset builder is present and types match.
5. **Disk space check.** Confirm at least 30 GB free on the partition holding `data/`.

If any pre-flight check fails, exit with a non-zero status and a clear error message naming the failed check.

### 8.2 Single-Row End-to-End Pipeline Test

After pre-flight passes, run the pipeline with `--dry-run` flag, which forces:

1. Question pool size = 1 question from MMLU-Pro plus 1 question from GSM8K.
2. Dumb personas = 1 variant per question (5 styles collapsed to 1 random).
3. Trials per question = 1.
4. **All five conditions executed** (C1 through C5). C5 is force-enabled in dry-run mode regardless of the calibration gate so the confidence-weighted code path is verified end-to-end.
5. **Each model is invoked at least once.** This means Round 0 and Round 1 must both fire for every model in every condition, and the judge must be invoked at least once (force a deliberate parse failure on one trial to test the judge fallback path).
6. **Confidence parsing exercised.** At least one trial must successfully parse the `Confidence: <int>` line, and at least one trial must hit the confidence-parse-failure path so the null-handling code is verified.
7. **C5 confidence filtering exercised.** Force one dry-run trial in C5 where both dumb peers report confidence below threshold, to verify the empty-peer-messages branch in §7.5 step 5.

Acceptance criteria for the dry run:

- All API keys produce valid responses.
- Both local models load with flash-attention attribute confirmed (check `model.config._attn_implementation == "flash_attention_2"`).
- Llama and Gemma occupy under 24 GB combined VRAM (check `nvidia-smi` mid-run).
- Every column in `trial_log.parquet` is populated for at least one row per model, including the new confidence columns.
- The metrics calculator runs and emits a valid (if statistically meaningless) summary.
- The mitigation summary file is emitted with at least one row from the C5 dry-run trial.
- Total wall-clock time under 12 minutes.

If the dry run does not meet all of the above, do not proceed to the full run.

---

## 9. Full Run Execution

### 9.1 Run Order

The full run executes in three sequenced stages. Each stage must complete and validate before the next begins.

**Stage 1 — Main experiment (`scripts/run_main_experiment.sh`):**
1. Full question pool of 300 items.
2. Conditions C1 through C4 × 5 trials per question = 6,000 main trials.
3. Plus the cross-model validation subset (50 questions × 4 conditions × 5 trials = 1,000 additional trials with GPT-4o-mini as focal).
4. Total Stage 1: 7,000 trials.

**Stage 2 — Calibration gate (`scripts/run_calibration_gate.sh`):**
After Stage 1 completes, run the calibration gate analysis described in §12. Output `data/outputs/calibration_gate_report.parquet`. The script writes a single line to `logs/pipeline.log` reading either `CALIBRATION_GATE_PASSED — proceeding to C5` or `CALIBRATION_GATE_FAILED — skipping C5; mitigation hypothesis not supported`.

**Stage 3 — Mitigation experiment (`scripts/run_mitigation_experiment.sh`):**
Runs only if Stage 2 passed.
1. C5 condition on 100-question subset × 3 trials = 300 mitigation trials.
2. Same focal smart agent (`deepseek_primary`).
3. Append rows to the existing `trial_log.parquet`. Do not overwrite.

If Stage 2 fails, the pipeline terminates cleanly. The paper still has C1 through C4 and a clean negative result on the mitigation precondition. Document the gate outcome in `experiment_metadata.json`.

### 9.2 Throughput Plan

The local dumb models run on the A100. Two HuggingFace processes serve Llama and Gemma respectively, each pinned to the same GPU. Smart-agent API calls happen concurrently with local generation via Python `asyncio` with a concurrency cap of 8 simultaneous outbound API requests.

Expected wall-clock for full run:
- Stage 1: 5 to 7 hours.
- Stage 2: under 5 minutes.
- Stage 3 (if it runs): 0.5 to 1 hour additional.
- **Total wall-clock with C5 included: 6 to 9 hours.**

### 9.3 Resumability

The trial runner must be idempotent. Use a checkpoint file (`data/outputs/completed_trials.parquet`) listing every (question_identifier, condition_identifier, trial_index, focal_smart_agent) tuple that has been fully written to `trial_log.parquet`. On startup, skip any tuple already in the checkpoint. This allows safe Ctrl-C interruption and resume across all three stages.

### 9.4 Live Monitoring

Print a progress bar (using `tqdm`) and a periodic summary line every 100 trials showing:
- Trials completed / total
- Estimated time remaining
- Current focal agent flip rate (running average)
- Current API failure rate
- Current GPU memory usage
- Current stage (`stage_1_main`, `stage_2_calibration_gate`, `stage_3_mitigation`)

---

## 10. Logging and Output Schema

### 10.1 Trial Log Output Schema

Write to `data/outputs/trial_log.parquet`. One row per agent response (so each trial generates between 1 and 6 rows depending on condition). Full column names, no abbreviations:

| Column Name | Data Type | Description |
|---|---|---|
| `trial_universal_unique_identifier` | string | UUID generated per trial |
| `question_identifier` | string | Foreign key to question pool |
| `condition_identifier` | string | One of the five condition labels |
| `trial_replication_index` | integer | 0 through 4 (or 0 through 2 for C5) |
| `focal_smart_agent_name` | string | The smart agent whose flips are tracked |
| `responding_agent_identifier` | string | `Agent_Alpha`, `Agent_Beta`, etc. |
| `responding_agent_role` | string | One of `smart_focal`, `smart_nonfocal`, `dumb` |
| `responding_agent_model_name` | string | Full model name string |
| `responding_agent_provider` | string | One of `deepseek`, `openrouter`, `local_huggingface` |
| `debate_round_index` | integer | 0 (independent) or 1 (peer-exposed) |
| `aggregation_rule_applied` | string | `none`, `standard_debate`, or `confidence_weighted` |
| `peer_messages_seen_in_context` | string or null | JSON-encoded list of (agent_identifier, response_text, confidence_integer_or_null) tuples shown in the debate context, or null for round 0 |
| `peer_messages_filtered_out_count` | integer or null | For C5: how many peer messages were excluded by the confidence threshold; null for other conditions |
| `peer_messages_ordering_seed` | integer or null | Per-trial ordering seed for round 1 |
| `injected_dumb_persona_identifier` | string or null | If a dumb agent, the persona variant used |
| `raw_response_text` | string | Full agent output |
| `extracted_final_answer` | string | After regex or judge extraction |
| `extracted_self_reported_confidence_integer` | integer or null | Parsed from `Confidence: <int>` line; null if missing or unparseable |
| `confidence_parse_status` | string | `success`, `missing_line`, `non_integer`, `out_of_range_clamped` |
| `answer_extraction_method` | string | One of `regex_success`, `judge_fallback`, `parse_failure` |
| `extracted_answer_matches_ground_truth` | boolean | True/False |
| `total_input_tokens` | integer | |
| `total_output_tokens` | integer | |
| `wall_clock_latency_seconds` | float | |
| `error_status` | string | `success`, `api_error_recovered`, `parse_error_recovered`, `failure` |
| `retry_attempts_used` | integer | |
| `timestamp_utc` | string | ISO 8601 |
| `random_seed_used_for_this_trial` | integer | |

### 10.2 Final Answers Output Schema

Write to `data/outputs/final_answers.parquet`. One row per (question, condition, trial, focal_smart_agent). This is the consolidated table for analysis:

| Column Name | Data Type | Description |
|---|---|---|
| `question_identifier` | string | |
| `condition_identifier` | string | |
| `trial_replication_index` | integer | |
| `focal_smart_agent_name` | string | |
| `aggregation_rule_applied` | string | |
| `round_zero_independent_answer` | string | Focal agent's pre-debate answer |
| `round_zero_answer_was_correct` | boolean | |
| `round_zero_focal_self_reported_confidence_integer` | integer or null | |
| `round_one_post_debate_answer` | string | Focal agent's post-debate answer |
| `round_one_answer_was_correct` | boolean | |
| `round_one_focal_self_reported_confidence_integer` | integer or null | |
| `focal_agent_flipped_correct_to_incorrect` | boolean | True if R0 correct and R1 incorrect |
| `focal_agent_flipped_incorrect_to_correct` | boolean | True if R0 incorrect and R1 correct |
| `dumb_peer_consensus_status` | string | One of `not_applicable`, `unanimous_wrong`, `split`, `unanimous_correct` |
| `condition_dumb_agent_count` | integer | 0, 1, or 2 |
| `c5_count_of_peer_messages_filtered_out` | integer or null | For C5 trials only |

### 10.3 Experiment Metadata

Write `data/outputs/experiment_metadata.json` once at run start, updated at run end. Required keys:

```
{
  "experiment_run_universal_unique_identifier": "...",
  "git_commit_hash_at_run_start": "...",
  "python_interpreter_path": "...",
  "package_versions": { "torch": "...", "transformers": "...", "flash_attn": "...", ... },
  "operating_system_release": "...",
  "gpu_device_name": "...",
  "gpu_total_memory_gigabytes": ...,
  "nvidia_driver_version": "...",
  "cuda_runtime_version": "...",
  "deepseek_primary_model_name_at_runtime": "...",
  "deepseek_judge_model_name_at_runtime": "...",
  "openrouter_primary_model_name_at_runtime": "...",
  "llama_actually_loaded_repository": "...",
  "gemma_actually_loaded_repository": "...",
  "gemma_fallback_was_triggered": false,
  "transformers_version_actually_used": "...",
  "random_seed_value": ...,
  "dry_run_passed_timestamp_utc": "...",
  "stage_1_main_experiment_started_timestamp_utc": "...",
  "stage_1_main_experiment_completed_timestamp_utc": "...",
  "stage_2_calibration_gate_decision": "passed_or_failed",
  "stage_2_calibration_gate_metric_value_at_decision": ...,
  "stage_3_mitigation_experiment_was_run": true_or_false,
  "stage_3_mitigation_experiment_started_timestamp_utc": "...",
  "stage_3_mitigation_experiment_completed_timestamp_utc": "...",
  "total_trials_completed_main_conditions": ...,
  "total_trials_completed_mitigation_condition": ...,
  "total_api_calls_made_to_deepseek": ...,
  "total_api_calls_made_to_openrouter": ...,
  "total_local_generations_for_llama": ...,
  "total_local_generations_for_gemma": ...,
  "total_judge_invocations": ...,
  "parse_failure_rate_for_answer_extraction_percentage": ...,
  "parse_failure_rate_for_confidence_extraction_percentage": ...,
  "approximate_total_api_cost_usd_estimated": ...
}
```

---

## 11. Metrics Computation

Compute and write to `data/outputs/metrics_summary.parquet`:

For each (condition_identifier, focal_smart_agent_name) combination:

| Column Name | Description |
|---|---|
| `condition_identifier` | |
| `focal_smart_agent_name` | |
| `aggregation_rule_applied` | |
| `total_trial_count` | |
| `round_zero_accuracy_rate` | Mean of `round_zero_answer_was_correct` |
| `round_one_accuracy_rate` | Mean of `round_one_answer_was_correct` |
| `round_one_minus_round_zero_accuracy_delta` | |
| `flip_rate_correct_to_incorrect` | Of trials where R0 was correct, fraction where R1 became incorrect |
| `flip_rate_incorrect_to_correct` | Of trials where R0 was incorrect, fraction where R1 became correct |
| `asch_conformity_index` | Flip rate when dumb peers were unanimously wrong, minus flip rate when peers were split. Defined only for C3, C4, and C5. |
| `bandwagon_dose_response_indicator` | The flip rate value for this condition, used to plot the gradient over dumb peer count |
| `bootstrap_confidence_interval_lower_95_percent` | |
| `bootstrap_confidence_interval_upper_95_percent` | |
| `mean_round_zero_focal_self_reported_confidence` | |
| `mean_round_one_focal_self_reported_confidence` | |

Bootstrap with 10,000 resamples, percentile method.

### 11.1 Mitigation-Specific Metrics

Compute and write to `data/outputs/mitigation_summary.parquet`:

| Column Name | Description |
|---|---|
| `comparison_label` | e.g., `C4_versus_C5_on_mitigation_subset` |
| `c4_round_one_accuracy_on_mitigation_subset` | Recompute C4 accuracy restricted to the same 100 questions used for C5 |
| `c5_round_one_accuracy` | |
| `c5_minus_c4_accuracy_delta_percentage_points` | The headline mitigation effect |
| `c5_minus_c4_bootstrap_confidence_interval_lower_95_percent` | |
| `c5_minus_c4_bootstrap_confidence_interval_upper_95_percent` | |
| `c5_minus_c4_mcnemar_p_value` | |
| `c5_minus_c4_mcnemar_p_value_bonferroni_corrected` | Same correction factor as the main analysis (7 total comparisons after C5 added) |
| `mean_count_of_peer_messages_filtered_out_per_c5_trial` | |
| `proportion_of_c5_trials_with_zero_peer_messages_after_filtering` | |
| `c5_focal_agent_flip_rate_correct_to_incorrect` | |
| `c4_focal_agent_flip_rate_correct_to_incorrect_on_mitigation_subset` | |

---

## 12. Calibration Gate (Stage 2)

This stage decides whether C5 is run. It is mandatory.

### 12.1 Rationale

C5 tests confidence-weighted aggregation as a mitigation. The mitigation can only work if the dumb models are systematically over-confident on wrong answers — that is, if a high-confidence-wrong response from a dumb model is detectable. If the dumb models are well-calibrated (low confidence on wrong answers), the filter has nothing to filter and C5 will produce a null result that adds noise to the paper.

### 12.2 Calibration Metric

Define the precondition metric as:

```
P(self_reported_confidence >= 60 | answer_is_wrong, agent_role == 'dumb', condition in {'C3', 'C4'})
```

Compute this from `trial_log.parquet` over all C3 and C4 trials, restricted to dumb agent responses (`responding_agent_role == 'dumb'`) where `extracted_answer_matches_ground_truth == False` and `extracted_self_reported_confidence_integer is not null`.

Compute also the partner metric for context:

```
P(self_reported_confidence >= 60 | answer_is_correct, agent_role == 'dumb', condition in {'C3', 'C4'})
```

The interesting quantity for the gate is the *difference* — how much more often dumb agents are loud-and-wrong versus loud-and-right. But for a simple binary gate, the absolute value of the loud-and-wrong probability is the threshold.

### 12.3 Decision Rule

The gate **passes** (and C5 runs) if:

```
P(confidence >= 60 | wrong, dumb agent, C3 or C4 trials) >= 0.40
```

The gate **fails** (and C5 is skipped) otherwise.

The threshold of 0.40 means: if at least 40% of dumb-agent wrong answers come with reported confidence ≥ 60, there is enough loud-and-wrong signal for confidence-based filtering to plausibly work. Below 40%, the dumb agents are calibrated enough that filtering high-confidence wrongs won't remove much.

### 12.4 Calibration Gate Report

Write `data/outputs/calibration_gate_report.parquet`:

| Column Name | Description |
|---|---|
| `precondition_metric_name` | `probability_high_confidence_given_wrong_answer_for_dumb_agents` |
| `precondition_metric_value` | The computed probability |
| `precondition_metric_bootstrap_confidence_interval_lower_95_percent` | |
| `precondition_metric_bootstrap_confidence_interval_upper_95_percent` | |
| `precondition_metric_threshold_for_pass` | 0.40 |
| `gate_decision` | `passed` or `failed` |
| `partner_metric_probability_high_confidence_given_correct_answer` | For context |
| `loud_wrong_minus_loud_right_difference` | Difference between the two probabilities, for the paper |
| `total_dumb_agent_responses_with_parsed_confidence_used_in_analysis` | |
| `breakdown_by_dumb_model_name` | JSON-encoded per-model probabilities (Llama vs Gemma separately) |

The script also writes a one-line decision to `logs/pipeline.log` and updates `experiment_metadata.json` with `stage_2_calibration_gate_decision`.

### 12.5 Honest Negative-Result Handling

If the gate fails:

1. Stage 3 does not run.
2. The paper still uses the gate result. It becomes a small subsection in Discussion: *"We tested whether confidence-weighted aggregation could mitigate the observed degradation. The precondition for this mitigation — systematic over-confidence by dumb agents on wrong answers — was not met (P(high confidence | wrong) = X, below the 0.40 threshold). We therefore did not run the mitigation experiment, and instead conclude that confidence-based filtering is not a viable defence against the bias mechanisms identified here."*
3. This is a publishable result. Do not falsify, lower the threshold, or run C5 anyway.

---

## 13. Statistical Analysis

Write to `data/outputs/statistical_tests.parquet`:

For each pair of conditions sharing the same questions (paired):

- C1 vs C2, C1 vs C3, C1 vs C4
- C2 vs C3, C2 vs C4
- C3 vs C4
- C4 vs C5 (only if C5 ran; restricted to the 100-question mitigation subset for both conditions)

Run McNemar's test on per-trial correctness using `statsmodels.stats.contingency_tables.mcnemar`. Apply Bonferroni correction for the actual number of comparisons run (6 if C5 was skipped, 7 if C5 was included). Report the corrected alpha alongside.

For the bandwagon dose-response, fit a logistic regression of `round_one_answer_was_correct` against `condition_dumb_agent_count` (using C2, C3, C4 only; C5 has different aggregation rule and is excluded from the dose-response curve) with question-level random effect using `statsmodels.MixedLM` or a clustered standard error approach.

Output table columns: `comparison_label`, `test_name`, `test_statistic`, `raw_p_value`, `bonferroni_corrected_p_value`, `is_significant_at_corrected_alpha`, `effect_size_estimate`, `notes`.

---

## 14. Reproducibility Requirements

1. The single value `RANDOM_SEED` from `.env` propagates to every randomized step. Derive per-step seeds deterministically from this root seed (e.g., `RANDOM_SEED + 1` for mitigation subset selection).
2. All package versions are pinned in `requirements.txt`.
3. The exact model-name strings used at runtime are written to `experiment_metadata.json`. If DeepSeek auto-routes the model name, capture the actual served model from the API response and log it.
4. The full question pool, persona pool, mitigation subset assignment, and trial log are written before metrics computation. A separate analysis script can reproduce all metrics from the saved trial log without rerunning any model.
5. The calibration gate decision and metric value are written in two places: `calibration_gate_report.parquet` and `experiment_metadata.json`. Both must agree.

---

## 15. Error Handling and Resilience

1. Every API call wrapped in retry with exponential backoff (see §6.2).
2. Local model generation failures (CUDA out of memory, NaN logits) caught and logged. The trial is marked `failure` and the loop continues. At end of run, report the failure count.
3. The pipeline never silently swallows errors. Every error is logged with full traceback to `logs/pipeline.log`.
4. If parse failure rate for the answer line exceeds 5% during the run, halt the pipeline and alert the user before continuing. Confidence-line parse failure is treated more leniently — alert at 15% but do not halt, since confidence-line failures only weaken the calibration gate analysis, not the main accuracy metric.
5. Checkpoint file is flushed after every 50 trials. SIGINT (Ctrl-C) triggers a final flush and clean exit.
6. If Stage 2 calibration gate fails, do not error — log the decision, write the report, and exit cleanly with status code 0. A failed gate is a valid, expected outcome.

---

## 16. Final Acceptance Checklist

Before declaring the run complete, the coding agent must verify:

- [ ] Pre-flight checks all pass.
- [ ] Dry run produces at least one row per model in `trial_log.parquet`.
- [ ] Dry run shows `flash_attention_2` is the active attention implementation for both local models.
- [ ] Dry run exercises all five conditions (C1–C5), including the C5 confidence-filter empty-peer-messages branch.
- [ ] Dry run successfully parses at least one `Confidence: <int>` line and gracefully handles at least one missing-confidence-line case.
- [ ] All API keys validated against their respective listing endpoints.
- [ ] All paths resolved and writable.
- [ ] All column names in output files match the full forms in §10 — no abbreviations.
- [ ] `experiment_metadata.json` is fully populated, including the actual served model names, calibration gate decision, and stage 3 run status.
- [ ] Persona validation retention rate is at least 80%.
- [ ] Parse failure rate at end of full run is below 5% for answer extraction and below 15% for confidence extraction.
- [ ] Stage 2 calibration gate report file exists and `gate_decision` is either `passed` or `failed` (never null).
- [ ] If `gate_decision == passed`, Stage 3 ran and `mitigation_summary.parquet` is populated.
- [ ] If `gate_decision == failed`, Stage 3 did not run and this is logged clearly in `experiment_metadata.json` and `logs/pipeline.log`.
- [ ] Statistical tests written for all 6 paired comparisons (or 7 if C5 ran), with Bonferroni correction matching the actual comparison count.
- [ ] A short human-readable summary of headline results (round-zero vs round-one accuracy per condition, flip rates, McNemar p-values, calibration gate decision, and if applicable C5 mitigation effect) is appended to `logs/pipeline.log` at run completion.

---

## 17. Out of Scope (do not implement)

- No code generation for paper writing, plotting, or LaTeX output.
- No mitigation strategies beyond the single confidence-weighted aggregation in C5.
- No additional models beyond the four specified.
- No quantization. Both local models are bfloat16.
- No multi-judge consensus. The single judge is used only as a fallback for regex parse failures.
- No fine-tuning of any model.
- No alternative confidence weight functions beyond `linear_above_threshold`. If results are interesting, more sophisticated weighting belongs in a follow-up paper.
- No re-running C3 or C4 with confidence weighting. Only C4's exact composition (1 smart + 2 dumb) is rerun under the mitigation rule, since that is where the largest deficit was and where mitigation has the most room to recover.

---

## 18. When Anything Is Ambiguous

Stop and ask the user before guessing. The user has emphasized: dry run, key checks, path checks, column mapping, full-form column names, bit-precision uniformity. Treat any ambiguity in those areas as a hard stop.

The calibration gate threshold (0.40) and confidence filter threshold (60) are deliberately conservative defaults. If pilot data suggests adjusting them, raise the question — do not edit silently.

# Plato's Ship — Capability-Asymmetric LLM Debate Study

A reproducible research pipeline measuring whether smart LLM agents' accuracy
degrades when debating with capability-weaker peers ("dumb" agents), and
whether confidence-weighted aggregation can mitigate this degradation.

## Quick Start

```bash
# 1. Set up environment (A100 GPU, Python 3.12, Ubuntu 22/24)
python3.12 -m venv ~/venv_debate_study
source ~/venv_debate_study/bin/activate

# 2. Install dependencies
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl -O /tmp/flash_attn.whl
pip install --no-deps /tmp/flash_attn.whl

# 3. Copy and fill in secrets
cp .env.example .env
# Edit .env with your API keys

# 4. Run dry-run tests
python dry_run/run_all_dry_tests.py

# 5. Run the full pipeline
bash scripts/run_dry_run.sh           # Mandatory dry run first
bash scripts/run_main_experiment.sh   # Stage 1: C1-C4
bash scripts/run_calibration_gate.sh  # Stage 2: Gate decision
bash scripts/run_mitigation_experiment.sh  # Stage 3: C5 (if gate passes)
```

## Experimental Conditions

| Condition | Smart | Dumb | Aggregation | Description |
|-----------|-------|------|-------------|-------------|
| C1 | 1 | 0 | none | Solo baseline |
| C2 | 3 | 0 | standard_debate | Homogeneous control |
| C3 | 2 | 1 | standard_debate | Plato condition |
| C4 | 1 | 2 | standard_debate | Collapse condition |
| C5 | 1 | 2 | confidence_weighted | Mitigation (post-gate) |

## API Key Configuration

- **DeepSeek**: 2 keys, round-robin cycling
- **OpenRouter**: 2 keys, round-robin cycling
- **Gemini**: 4 keys, round-robin (primary judge)
- **Mistral**: 1 key (secondary judge)
- **Judge cascade**: Gemini → Mistral → DeepSeek (no retry, only fallback)

## Project Structure

```
Code/
├── .env / .env.example     # Secrets
├── config/                 # YAML configuration
├── src/                    # Core pipeline code
│   ├── agent_wrappers/     # API and local model agents
│   ├── debate_protocol.py  # Standard debate (C1-C4)
│   ├── confidence_weighted_protocol.py  # C5
│   ├── calibration_gate.py # Stage 2 gate
│   ├── trial_runner.py     # Execution engine
│   └── pipeline_orchestrator.py  # Main entry point
├── dry_run/                # Test suite for all stages
├── scripts/                # Shell scripts for each stage
├── tests/                  # Post-run assertions
├── data/                   # Datasets and outputs
└── logs/                   # Pipeline and API failure logs
```

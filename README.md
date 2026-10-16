<div align="center">
  <img src="docs/assets/castflow-mark.svg" width="88" alt="CastFlow mark">
  <h1>CastFlow</h1>
  <p><strong>Bridging Evidence Acquisition and Accurate Prediction in Agentic Time Series Forecasting</strong></p>
  <p>CastFlow is a role-specialized agentic workflow for time series forecasting. It couples a frozen general-purpose reasoning model for planning and reflection with a trainable local forecasting model for evidence-guided numerical refinement.</p>
  <p>
    <a href="https://arxiv.org/abs/2604.27840"><kbd><strong>arXiv</strong></kbd></a> &nbsp;
    <a href="https://forever-pan.github.io/CastFlow/assets/paper/CastFlow.pdf"><kbd><strong>Paper PDF</strong></kbd></a> &nbsp;
    <a href="https://forever-pan.github.io/CastFlow/"><kbd><strong>Project page</strong></kbd></a> &nbsp;
    <a href="#citation"><kbd><strong>Citation</strong></kbd></a>
  </p>
</div>

<p align="center">
  <a href="https://forever-pan.github.io/CastFlow/">
    <img src="docs/assets/figures/rendered/framework.png" alt="CastFlow overview" width="940">
  </a>
</p>

---

## Start here

CastFlow pairs experience-guided evidence acquisition with a specialized numerical
forecaster. The paper reports the lowest MSE and MAE on 9 of 10 benchmarks among
21 competing methods. See the [results and comparison table](https://forever-pan.github.io/CastFlow/#results).

- **Read:** [paper PDF](https://forever-pan.github.io/CastFlow/assets/paper/CastFlow.pdf), [arXiv](https://arxiv.org/abs/2604.27840), and [project page](https://forever-pan.github.io/CastFlow/).
- **Set up:** [installation](#installation), [dataset placement](#dataset-placement), and [weight placement](#weight-placement).
- **Run:** [end-to-end workflow](#end-to-end-workflow) and [runtime configuration](docs/REPRODUCIBILITY.md).
- **Cite:** use the [BibTeX entry](#citation) or GitHub's **Cite this repository** panel.

### Quick environment check

```bash
git clone https://github.com/Forever-Pan/CastFlow.git
cd CastFlow
python -m pip install -e .
python -m cli --help
```

This installs the core CLI. To run forecasting, continue with the optional
dependencies, input data, expert and forecasting weights, and two model endpoints
described below.

## Workflow Overview 🔁

The public implementation exposes the following forecasting roles. SFT and RLVR update only the forecasting model; planning and reflection remain frozen:

| Component | Workflow |
| --- | --- |
| **Planning** | Select diagnostic tools using a frozen external LLM. |
| **Action** | Execute the multi-view toolkit and build an ensemble forecast baseline. |
| **Forecasting** | Use a fine-tuned local LLM to refine the baseline into a numerical forecast. |
| **Reflection** | Validate output structure and evidence alignment, then retry if needed. |
| **Strategy Memory** | Retrieve successful historical tool-use trajectories. |
| **Foundational Anchorer** | Retrieve similar historical cases and ensemble classical, deep, and foundation time-series models. |

## Directory Layout 🗂️

```text
CastFlow/
  cli/                              CLI entry package (`python -m cli` / `castflow`)
  scripts/                          Bash wrappers for the end-to-end workflow
  utils/                            Runtime config, dataset registry/io, schema, and shared helpers
  evaluation/                       Forecast parsing and metric evaluation
  forecasting/                      Prompting, tool execution, and inference modules
  workflow/                         End-to-end workflow orchestration and strategy memory
  training/                         SFT, RLVR, reward, and export utilities
  anchorer/                         Foundational Anchorer case-library construction
  backends/                         External time-series model runtime
  docs/                             Project page assets
  assets/                           Repository figures
```

<a id="dataset-placement"></a>

## Dataset Placement 📊

Prepare the benchmark CSVs using the reported chronological 7:1:2 split before
running the workflow. The registry reads these files and uses the following
lookback/horizon/stride settings; it does not create the splits automatically.

| Dataset | Train CSV | Test CSV | Lookback | Horizon | Stride |
| --- | --- | --- | ---: | ---: | ---: |
| BE | `data/raw/train/EPF_BE_train_val.csv` | `data/raw/test/EPF_BE_test.csv` | 168 | 24 | 48 |
| DE | `data/raw/train/EPF_DE_train_val.csv` | `data/raw/test/EPF_DE_test.csv` | 168 | 24 | 48 |
| FR | `data/raw/train/EPF_FR_train_val.csv` | `data/raw/test/EPF_FR_test.csv` | 168 | 24 | 48 |
| NP | `data/raw/train/EPF_NP_train_val.csv` | `data/raw/test/EPF_NP_test.csv` | 168 | 24 | 48 |
| PJM | `data/raw/train/EPF_PJM_train_val.csv` | `data/raw/test/EPF_PJM_test.csv` | 168 | 24 | 48 |
| ETTh | `data/raw/train/ETT_ETTh1_train_val.csv` | `data/raw/test/ETT_ETTh1_test.csv` | 96 | 96 | 48 |
| ETTm | `data/raw/train/ETT_ETTm1_train_val.csv` | `data/raw/test/ETT_ETTm1_test.csv` | 96 | 96 | 96 |
| WP | `data/raw/train/windy_power_train_val.csv` | `data/raw/test/windy_power_test.csv` | 96 | 96 | 96 |
| SP | `data/raw/train/sunny_power_train_val.csv` | `data/raw/test/sunny_power_test.csv` | 96 | 96 | 96 |
| MOPEX | `data/raw/train/mopex_train_val.csv` | `data/raw/test/mopex_test.csv` | 96 | 96 | 48 |

ETTh and ETTm are the manuscript display names. The CLI and file registry use
`ETTh1` and `ETTm1` as their corresponding identifiers.

CSV files should contain one timestamp column named `date` or `time_stamp`. If no timestamp column exists, CastFlow uses the row index. The target column defaults to the last numeric non-timestamp column; use `--target-col` if you need to override it.

<a id="weight-placement"></a>

## Weight Placement 🧠

### Local Forecasting LLM

Place the trainable local model somewhere accessible, for example:

```text
../models/Qwen3-4B
```

The configuration uses **Qwen3-4B** as the local trainable forecasting backbone.

### Foundational Anchorer Runtime

`backends/` contains wrappers and local code for the anchorer model pool. Put the required weights under the expected runtime subdirectories, for example:

```text
backends/shared/foundation_models/
backends/packages/
```

The anchorer can then build per-domain `case_library/*/anchor_library.json` files. If a model weight is missing, the corresponding anchor model may fail or be skipped depending on the runtime wrapper.

## Environment 💻

The workflow scripts target Linux with Python 3.10 or 3.11. Training and local
model serving additionally require a compatible GPU/CUDA environment.

```bash
git clone https://github.com/Forever-Pan/CastFlow.git
cd CastFlow
```

## Requirements 📦

| Package | Version / use |
| --- | --- |
| Python | `>=3.10` |
| `numpy`, `pandas` | core data handling |
| `openai` | Planning, Reflection, and OpenAI-compatible inference |
| `tqdm` | progress display for long-running workflows |
| `python-dateutil`, `holidays` | time-series calendar utilities |
| `statsmodels`, `pyclustering` | Foundational Anchorer case-library build |
| PyTorch | `>=2.1`, for SFT |
| `transformers` | `>=4.51.0`, required for Qwen3 support |
| `datasets` | `>=2.14` |
| `peft` | `>=0.6` |
| `accelerate`, `deepspeed` | distributed SFT |
| `einops`, `safetensors` | deep learning / model loading |
| `pyarrow` | `>=12`, RL parquet I/O |
| `agentlightning` | RLVR |
| `vllm` | local forecasting serving |
| `chronos-forecasting`, `timesfm` (`<3.12`), `xgboost`, `lightgbm` | anchorer model pool |
| `scipy`, `scikit-learn`, `prophet` | classical / statistical models |

The package metadata in `pyproject.toml` already defines the core, anchorer, training, RLVR, and serving dependency groups.

<a id="installation"></a>

## Installation ⚙️

Install the core package, then add the components needed for your run:

```bash
python -m pip install -e .
python -m cli --help
# Optional expert-model pool:
python -m pip install -e '.[anchorer]'
# Optional training and RLVR:
python -m pip install -e '.[training,rlvr]'
# Use an environment with a compatible PyTorch/CUDA stack for serving:
python -m pip install -e '.[serving]'
```

`requirements.txt` lists the combined dependency set. The optional groups let
you install only the components you need.

Create a private configuration from the credential-free template:

```bash
cp .env.example .env
```

Set the planning/reflection endpoint, credential, and model name in `.env`.
The forecasting endpoint must point to your separately served trained model.
The example sets a 5,000-token forecasting limit, matching the reported limit;
the code default remains 7,000 unless overridden. See
[runtime and reproducibility notes](docs/REPRODUCIBILITY.md) for configuration,
required external assets, and the scope of this public implementation.

## Default Parameters 🎛️

The table below describes public-code defaults and example-script settings.
They do not identify a particular historical experiment run.

| Component | Setting |
| --- | --- |
| Frozen planner/reflection model | Configure the endpoint and model explicitly; the manuscript uses Grok 4 |
| Local trainable forecaster | Qwen3-4B |
| Memory construction | K-parallel exploration, `PARALLEL_PLAN_K=4` |
| Memory retrieval | top-k memory retrieval, default `K=3`, threshold `0.90` |
| Reflection retries | train `3`, test `10` |
| SFT | cross-domain, 1 epoch, learning rate `5e-5`, global batch size 8 |
| RLVR | GRPO, group size `G=8`, temperature `1.0`, learning rate `2e-6`, KL coefficient `0.0`, 3 epochs |
| Forecast output length | 5,000 in `.env.example`; code default 7,000 |

## CLI Usage 🖥️

Workflow commands are implemented in `cli/cli.py`. From the repository root:

```bash
python -m cli --help
```

After `pip install -e .`, the same interface is available as `castflow`.

The numbered scripts in `scripts/` wrap these commands with the default cross-domain paths used below. Most scripts forward extra flags via `"$@"`; training and serving scripts also honor environment variables such as `BASE_MODEL_PATH`, `SFT_OUTPUT_DIR`, and `LOCAL_FORECAST_MODEL_PATH`.

<a id="end-to-end-workflow"></a>

## End-to-End Workflow 🚀

Run the steps below from the repository root in order.

<details open>
<summary><strong>1. Build The Foundational Anchorer Case Libraries</strong></summary>

Scans all registered train splits and writes one case library per domain.

```bash
bash scripts/01_build_anchorer.sh
```

Outputs:

```text
case_library/EPF_BE/anchor_library.json
case_library/EPF_DE/anchor_library.json
case_library/EPF_FR/anchor_library.json
case_library/EPF_NP/anchor_library.json
case_library/EPF_PJM/anchor_library.json
case_library/ETT_ETTh1/anchor_library.json
case_library/ETT_ETTm1/anchor_library.json
case_library/windy_power/anchor_library.json
case_library/sunny_power/anchor_library.json
case_library/mopex/anchor_library.json
```

For a small initial run, append `--max-windows 5`. This still executes the expert
models and writes a case library.

</details>

<details open>
<summary><strong>2. Build Cross-Domain Strategy Memory</strong></summary>

Loops over all registered train splits and uses the matching `case_library/*/anchor_library.json`.

```bash
bash scripts/02_build_memory.sh
```

Output: `memory/cross_domain/memory.json`

</details>

<details open>
<summary><strong>3. Export SFT Data From Memory</strong></summary>

```bash
bash scripts/03_export_sft.sh
```

Output: `data/sft/cross_domain_sft.csv`

</details>

<details open>
<summary><strong>4. Supervised Fine-Tuning</strong></summary>

Paper-style target: Qwen3-4B, 1 epoch, learning rate `5e-5`, global batch size 8.

```bash
bash scripts/04_train_sft.sh
```

Defaults: base model `../models/Qwen3-4B`, output `models/sft_cross_domain`, `max-length 14000`. Override with `BASE_MODEL_PATH`, `SFT_OUTPUT_DIR`, `NPROC_PER_NODE`, or `MASTER_PORT`.

</details>

<details open>
<summary><strong>5. Prepare RL Data</strong></summary>

```bash
bash scripts/05_prepare_rl.sh
```

Output: `data/rl/cross_domain_rl.parquet`

</details>

<details open>
<summary><strong>6. RL Training</strong></summary>

```bash
bash scripts/06_train_rlvr.sh
```

Output: `models/rl_cross_domain/`

</details>

<details open>
<summary><strong>7. Serve The Local Forecasting Model</strong></summary>

Start vLLM before forecasting. The served model name must match `LOCAL_MODEL_NAME` in `.env`.

```bash
bash scripts/07_serve_local_forecaster.sh
```

By default this serves `models/sft_cross_domain` on port `8002`. Set `LOCAL_FORECAST_MODEL_PATH` to use an RL checkpoint instead.

</details>

<details open>
<summary><strong>8. Forecast A Test CSV</strong></summary>

Forecasting requires an explicit test CSV via `--data`. For registered benchmark filenames such as `EPF_DE_test.csv` and `windy_power_test.csv`, CastFlow automatically infers lookback, horizon, seasonal period, and stride from the file path.

DE example:

```bash
bash scripts/08_forecast_de.sh
```

Output: `predictions/de_forecast.csv`

</details>

<details open>
<summary><strong>9. Evaluate Forecasts</strong></summary>

DE example:

```bash
bash scripts/09_evaluate_de.sh
```

Prints aggregate MSE/MAE and writes `predictions/de_metrics.csv`.

</details>

<a id="citation"></a>

## Citation 📖

If you use CastFlow in your research, please cite the paper below.
The [project page](https://forever-pan.github.io/CastFlow/) provides the current
manuscript and figures. The [arXiv record](https://arxiv.org/abs/2604.27840)
preserves the preprint's version history.

```bibtex
@misc{pan2026castflow,
  title         = {CastFlow: Bridging Evidence Acquisition and Accurate Prediction in Agentic Time Series Forecasting},
  author        = {Pan, Bokai and Cheng, Mingyue and Tao, Xiaoyu and Liu, Zhiding and Yu, Shuo and Wu, Yuchong and Liu, Qi and Lian, Defu and Chen, Enhong},
  year          = {2026},
  eprint        = {2604.27840},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2604.27840},
  url           = {https://arxiv.org/abs/2604.27840}
}
```

[Download BibTeX](docs/assets/citation.bib). Data, weights, and generated
case/memory libraries are described in [the runtime notes](docs/REPRODUCIBILITY.md).

# Runtime configuration

## Inputs and storage

Use Python 3.10 or 3.11 for the training and serving stack. Native expert
preparation and the core CLI run on CPU; SFT, RLVR, and local 4B serving require
their selected runtime dependencies and suitable hardware. Install only the
optional groups needed by the chosen experts and workflow stages.

Supply chronologically split CSVs, model weights, and generated libraries
separately. These assets are not part of the source checkout. The registered
filenames and window settings are in `utils/registry.py`; ETTh/ETTm resolve to
ETTh1/ETTm1. The loader validates finite targets, unique chronological timestamps,
and positive window sizes. It does not silently drop invalid target rows or
create train/test splits. Pass explicit target and timestamp column names when
the input schema differs from the defaults.

## Initialize cases and experience memory

```bash
python -m pip install -e '.[initialization]'
cp .env.example .env
# Fill in the frozen teacher endpoint, model, and credential, then:
python -m cli initialize --datasets BE,ETTh,WP
```

The initializer checks all selected training inputs before model preparation.
It prepares named experts and defaults to 24 distinct parameter candidates per
expert, with a maximum budget of 256. Candidate 1 preserves the untuned recipe;
subsequent candidates are reproducible and prefix-stable as the budget grows.
Finite parameter spaces report exhaustion; failed fits remain recorded and are
ineligible instead of aborting all remaining candidates or disappearing from
their score. A candidate must succeed on every preselected selection window.

The final 20% of the supplied training CSV is divided chronologically into
selection and confirmation intervals. Their complete target windows do not
overlap. Predictors receive lookbacks only. `--tuning-windows 12` and
`--confirmation-windows 12` choose representative origins without inspecting
target values. Parameters and the untuned incumbent are selected using only the
earlier interval. The saved library ends before the confirmation targets, and
that exact frozen library is evaluated on the later interval.

Default `--quality-policy required` requires at least four complete confirmation
windows, MSE at least 1% below a fixed-period SeasonalNaive reference, MSE no worse
than the untuned incumbent, and MAE no worse than either reference. The incumbent
is the default expert recipe chosen before confirmation by selection MSE, then
MAE. Numerical ties at the zero-error floor are allowed. A failed gate stops
initialization before teacher exploration and names its saved report.
`--min-mse-improvement`, `--max-mae-regression`, and
`--min-confirmation-windows` expose the acceptance settings. Explicit
`--quality-policy report` prepares a diagnostic bundle with `quality_passed`
still reported. It does not turn a failed criterion into a successful one.

This gate assesses the prepared Anchorer prior. Teacher exploration and local
forecaster training are separate stages; it is not an end-to-end forecasting
quality result. Confirmation results are not fed back into parameter selection.
Use `--validation-fraction`, `--tuning-rounds`, and `--max-windows` to control
preparation. A capped initialization is a smaller library, not a full-data run.

The next stage uses the frozen teacher for candidate tool-use trajectories and
saves successful experience memory. It does not train the local 4B forecaster.
After every dataset is complete, the command returns the cross-domain memory
path and per-dataset library paths. `--anchors-only` ends after expert cases;
it does not claim that experience memory is ready.

Defaults are SeasonalNaive, HistoricAverage, AutoARIMA, and Theta. Inspect
`initialize --list-models` to choose other experts. Ridge uses NumPy; XGBoost and
LightGBM require `.[expert-ml]`; ChronosTiny requires `.[foundation]` and local
`amazon/chronos-t5-tiny` weights at `<model-store>/ChronosTiny`. Native experts
save recipes and fit each lookback. `--download-models` explicitly allows
preparing missing selected pretrained weights. A missing selected dependency
or weight stops initialization before the ensemble is built.

Runtime files default to `../castflow_runtime/initialization`, with prepared
experts under `../castflow_runtime/models`. Identical inputs and settings resume
completed stages after integrity checks; changed source, data, model, or
parameters produce a distinct identity. `--no-resume` starts a new preparation.
Selection origins, candidate failures, completed library cases, and confirmation
origins are checkpointed for recovery. Completed memory windows, including rejected trajectories, are recorded for
resume. An interrupted external call that has not committed may be executed
again; already committed window outcomes are reused.
The returned paths can be passed to the individual commands:

```bash
python -m cli export-memory-data --memory /path/to/experience_memory.json --output /path/to/sft.csv
python -m cli forecast --dataset BE --memory /path/to/experience_memory.json --anchor-library /path/to/BE/anchor_library.json --output /path/to/predictions.csv
```

For mixed or custom inputs, `--dataset-manifest /path/datasets.json` accepts a
JSON list. Relative paths are resolved against the manifest directory:

```json
[
  {
    "name": "custom",
    "data_path": "train.csv",
    "split": "train",
    "lookback": 96,
    "horizon": 24,
    "seasonal_period": 24,
    "stride": 24,
      "memory_stride": 24,
    "target_col": "value",
    "timestamp_col": "date"
  }
]
```

All initialization inputs must belong to the training side. Expert recipes and
parameter selection use these inputs only; a final evaluation split remains
separate. Test-named paths are rejected as an additional input guard.
Registered datasets retain separate anchor and memory stride defaults from the
registry. In custom manifests, `memory_stride` defaults to `stride`; an explicit
CLI `--stride` sets both.

## Model endpoints and generation

The external endpoint configured in `.env` supplies frozen planning, reflection,
and training-side teacher generation. Normal `forecast` uses the separately
served specialized forecasting checkpoint; it requires the local endpoint.
The serving script resolves the latest complete RL actor export, or accepts
`LOCAL_FORECAST_MODEL_PATH` explicitly. Credentials stay in the ignored `.env`
or process environment. Set model revision fields when endpoint model names can
refer to changing checkpoints.

| Setting | Default |
| --- | --- |
| Memory retrieval | top-k 3; cosine threshold 0.90 |
| Training-side strategies | 4 candidates |
| Workflow retries | first attempt plus 3 training-side or 10 inference retries |
| Forecast completion | 5,000 tokens; temperature 0.3 |
| Planning and reflection | frozen API model; temperature 0.0 |
| API retry | one explicit retry layer; up to 2 retries for temporary failures |
| Diagnostic reuse | `current_plan`, scoped to one request |

These settings are configurable. The selected expert parameters and input
identities are stored with initialized libraries. Library v2 uses combined
normalized DTW, Euclidean, and cosine distances and historical-MSE softmax
weights. `build-anchorer --anchor-distance-weights` can set the three distance
coefficients. An old library must contain recoverable expert losses to migrate;
winner counts alone cannot reconstruct those losses.

`--no-api` enables explicit prior fallback without model calls. Such a result is
marked `prior_fallback`, not an accepted model forecast. Without a usable prior,
the result is marked missing. Forecast responses undergo finite-value, complete
horizon, and timestamp checks before frozen semantic reflection. At the retry
limit, a structurally valid response from the same request is preferred, followed
by its usable ensemble prior. Future test labels never choose the response.

## Origin-issued auxiliary forecasts

Historical covariates remain in the lookback. Supply future-period auxiliary
forecasts through a separate `--covariate-forecasts` CSV with the forecast
timestamp column and `issued_at`. Each chosen vintage must be available no
later than the forecast origin. A single-vintage table can use
`--covariate-issued-at` instead. The full horizon must match the forecast
schedule; future target values are forbidden in this input. The selected
auxiliary forecasts are held fixed across retries. Without this explicit input,
the workflow does not read realized future covariates from evaluation labels.

For initialization, provide `covariate_forecasts_path`, optional
`covariate_issued_at`, and optional `covariate_issue_col` in each dataset manifest
entry. This keeps provenance separate for each dataset.

## Training and evaluation

SFT supervises the assistant continuation, including its reasoning and forecast,
with prompt tokens masked. It preserves the complete response and rejects
examples exceeding the configured context length. Global batch size defaults
to 8; accumulation is derived from worker count and per-device batch size.

RLVR samples each group from one fixed evidence/prior context and directly
scores each completion. It does not rerun a full planning-reflection loop for
every candidate. Only forecasting parameters are updated. Reward combines the
paper's absolute MSE term, relative MSE improvement over the matched prior, and
the specified invalid-output penalties. MAE is reported as an evaluation metric;
there is no separately learned tool-call reward. Residual diagnostics remain
restricted to training-side exploration.

The pinned RLVR backend requires CUDA GPUs with compute capability 8.0 or newer
and BF16 support. SFT separately chooses a supported precision and can run on CPU.
Before GRPO batch construction, CastFlow checks complete candidate groups,
explicit finite rewards, one response trace per candidate, token limits, and
identical prompt tokens within each group. Missing candidates fail explicitly
instead of changing the group size or receiving an implicit zero reward.

Supply separate train and held-out training-side validation parquet files to
`train-rlvr --dataset-path ... --validation-path ...`; overlapping prompts are
rejected. RL data preparation requires actual labels and a complete matching
prior. It never substitutes teacher predictions for labels.

CSV evaluation requires full horizon alignment and finite values. Explicit
positional forecasts must use a complete ordered sequence; timestamp forecasts
must match all target timestamps. Invalid rows remain reported. Aggregate
MSE/MAE/RMSE pool the valid horizon points, and the summary reports valid and
invalid row counts. Evaluation exits unsuccessfully if any rows are invalid.

## Request-scoped execution recovery

Enable persistent recovery with both `--execution-journal-dir /path/journal`
and `--request-id unique-input-id`. Committed stages are atomically persisted
and reused only when request, input, library, configuration, model revision,
and implementation identities match. Semantic revisions have distinct stage
identities. A new input requires a new request ID; this is not cross-request
forecast caching. `--diagnostic-cache-policy none` disables diagnostic reuse.

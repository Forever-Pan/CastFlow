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
python -m pip install -e '.[casebank]'
cp .env.example .env
# Fill in the frozen teacher endpoint, model, and credential, then:
python -m cli initialize --datasets BE,ETTh,WP --expert-preset balanced --promotion-policy conservative --download-models
```

The initializer checks all selected training inputs before model preparation.
It prepares named experts and defaults to 64 distinct parameter candidates per
expert, with a maximum budget of 256. Candidate 1 preserves the untuned recipe;
subsequent candidates are reproducible and prefix-stable as the budget grows.
Finite parameter spaces report exhaustion; failed fits remain recorded and are
ineligible instead of aborting all remaining candidates or disappearing from
their score. A candidate must succeed on every preselected selection window.

The default training-prefix cutoff is 60%. Geometry cases end at 75%, recipe
selection occupies 75%–82.5%, loss calibration occupies 82.5%–90%, and confirmation
occupies the remaining 10%. Each scoring stage uses up to 24 representative,
complete, non-overlapping target windows chosen without inspecting their labels.
Their origins stay on the dataset's original stride grid; fraction boundaries
are not shifted, and the first eligible origin is rounded upward to that grid.
Fresh DLinear fits, including its internal scaler and validation, stay within
the 60% prefix; its geometry and memory targets start after that cutoff.
Lookback-fitted and frozen pretrained experts receive only each origin's past.
The original default recipes compete for a fixed incumbent on selection data.
Confirmation targets cannot choose recipes, variants, losses, or weights.

Keep a valid original recipe plus up to `--retain-variants 3` distinct tuned
versions per family. Prediction-equivalent versions are removed using selection
outputs only. In each fixed medoid cluster, calibration activates one estimated
best variant per family, optionally limiting the number of active families.
Thus identical copies do not gain weight through their count. Historical losses
use completed geometry targets and then earlier calibration targets only.
The prequential search chooses the lowest-MSE feasible policy subject to the
fixed incumbent's MAE guard, with the fixed incumbent also an explicit candidate.
No quality threshold is silently relaxed when a policy is infeasible.

Normalized mode defines `L = shrunken historical validation MSE /
(historical SeasonalNaive MSE * temperature)` and applies `softmax(-L)`.
Temperature candidates are 0.25, 0.5, 1, 2, 4, 8, 16, and 64; cluster shrinkage
strengths are 0, 4, and 16. Both are selected before confirmation and stored with
the active variants and calibration trace. This makes the scale and weight
policy explicit while retaining historical-loss softmax aggregation; these new
preparation settings do not identify the parameters of historical paper runs.
`--loss-mode raw` disables loss normalization and temperature/shrinkage search
(temperature 1, shrinkage 0). Variant selection and the fixed-incumbent candidate
remain available; this option does not convert a version-3 bank into version 2.
Version-2 libraries retain their stored raw-loss behavior when loaded.

Case limits uniformly cover eligible origins instead of taking the first N.
After calibration, additional past references may be added without changing
the fixed medoids or their calibrated losses. Actual populated row/time ranges,
eligible boundaries, and model fit cutoffs are stored separately. The saved
library ends before confirmation, and that exact frozen library is scored.

Default `--quality-policy required` requires at least four complete calibration
and four complete confirmation windows, MSE at least 1% below a fixed-period SeasonalNaive reference, MSE no worse
than the untuned incumbent, and MAE no worse than either reference. The incumbent
is the default expert recipe chosen before confirmation by selection MSE, then
MAE. Numerical ties at the zero-error floor are allowed. A failed gate stops
initialization before teacher exploration and names its saved report.
`--min-mse-improvement`, `--max-mae-regression`, and
`--min-confirmation-windows` expose the acceptance settings. Explicit
`--quality-policy report` prepares a diagnostic bundle with `quality_passed`
still reported. It does not turn a failed criterion into a successful one.

`--promotion-policy none` preserves this frozen-prior confirmation behavior.
Explicit `--promotion-policy conservative` freezes up to three distinct global
policies before scoring the tail: the prepared ensemble, the original-default
champion chosen on earlier calibration data, and the selection incumbent.
Every policy faces the same origins and unchanged readiness criteria. Qualified
policies are ranked by MSE, MAE, then a deterministic preference for fixed
policies on ties. No qualifying policy produces no promoted artifact.
The prepared bank, candidate identities, original reports, and selected artifact
are retained separately for recovery and inspection.

In conservative mode, the tail is **training-side promotion validation used for
policy selection**, with `independent_quality_verified=false`. Its scores are
not independent test evidence. Loss fitting and expert recipes remain frozen
before that interval; policy selection uses its labels and records their exact
availability boundary. Promoted inference requires timestamps at or after that
boundary. Row-only inputs remain supported with promotion disabled.
Experience Memory uses the prepared training bank and excludes promotion
targets. The returned `memory_anchor_library` and inference `anchor_library`
paths identify the different roles explicitly. Keep a separate test CSV outside
all initialization and policy choices.

This gate assesses the prepared Anchorer prior. Teacher exploration and local
forecaster training are separate stages; it is not an end-to-end forecasting
quality result. With promotion disabled, confirmation results are not fed back
into selection.
Use `--fit-fraction`, `--selection-start-fraction`,
`--calibration-start-fraction`, `--confirmation-start-fraction`,
`--tuning-rounds`, and `--max-windows` to control preparation.
`--validation-fraction` remains a legacy tail-boundary alias and cannot be mixed
with explicit stage fractions. A capped initialization is a smaller library,
not a full-data run. Scoring-window caps are separate from the case count cap.

The next stage uses the frozen teacher for candidate tool-use trajectories and
saves successful experience memory. It does not train the local 4B forecaster.
Its complete target windows stay before the confirmation boundary and, for
prefix-trained experts, start at or after the actual global model fit cutoff.
After every dataset is complete, the command returns the cross-domain memory
path and per-dataset library paths. `--anchors-only` ends after expert cases;
it does not claim that experience memory is ready.

Without `--models` or `--expert-preset`, the compatible native defaults remain
SeasonalNaive, HistoricAverage, AutoARIMA, and Theta (`.[initialization]`).
The recommended `--expert-preset balanced` selects SeasonalNaive, HistoricAverage,
AutoARIMA, Theta, Ridge, DLinear, and ChronosBoltBase (`.[casebank]`). Presets and `--models`
are mutually exclusive. DLinear prepares up to three fresh moving-average
kernel variants, 25, 7, and 49, with `--dlinear-epochs 20`. It does not import
legacy checkpoints with unknown scalers or fit cutoffs. Chronos-Bolt Base uses
the native median-quantile API with local `amazon/chronos-bolt-base` weights at
`<model-store>/ChronosBoltBase`. `--model-device cpu` is explicit; select
`cuda:0` when suitable hardware is available.

Inspect `initialize --list-models` for all choices. Ridge uses NumPy; XGBoost and
LightGBM require `.[expert-ml]`; ChronosTiny requires `.[foundation]` and local
`amazon/chronos-t5-tiny` weights at `<model-store>/ChronosTiny`. Native experts
save recipes and fit each lookback. `--download-models` explicitly permits
preparing missing selected pretrained weights. Missing selected dependencies
or weights stop initialization; models are never silently removed.
The registry uses daily periods of 24 for hourly data and 96 for 15-minute
WP/SP/ETTm data. When replaying an older library, pass its stored period
explicitly; incompatible query periods are rejected instead of reinterpreted.

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
identities are stored with initialized libraries. Libraries use combined
normalized DTW, Euclidean, and cosine distances. Version 2 retains historical-MSE
softmax weights; version 3 also records explicit loss calibration and any policy
promotion separately. `build-anchorer --anchor-distance-weights` can set the three distance
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

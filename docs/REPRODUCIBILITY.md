# Runtime and reproducibility notes

## Setup and external assets

Run the numbered Bash scripts from the repository root on Linux with Python
3.10 or 3.11. Install the core package and the optional dependency groups needed
by the selected step. Supply datasets, forecasting and expert-model weights,
case libraries, and experience-memory files separately. These assets are not
included in this source release. Missing expert weights may change the active
ensemble pool; record the actual models and checkpoint revisions in each run.

The README uses the paper's ETTh/ETTm names. The CLI identifiers are ETTh1/ETTm1.
The registered paths and window settings are in `utils/registry.py`. The loader
reads supplied CSVs; it does not reconstruct a chronological 7:1:2 split for you.
Record and preserve your split boundaries and preprocessing. Pass `--target-col`
and `--timestamp-col` when the input schema does not match the defaults.

Only source information available at the forecast origin belongs in inference
inputs. Future-period covariate forecasts require issue-time provenance;
realized future target values are evaluation labels, not forecasting evidence.
The loader alone cannot establish that a field is an origin-issued forecast.

## Configure the two model endpoints

Copy `.env.example` to `.env`, fill in the external reasoning endpoint and key,
and set the local forecasting endpoint to your served trained checkpoint.
Keep private keys out of commits. Model identifiers depend on the API provider.
The external endpoint is used for planning and reflection. The local endpoint
is used for specialized forecasting during `forecast` when configured. If it
is absent, the implementation can use the general model or fallback path, so
configure both endpoints when evaluating the role-specialized model.

The serving script defaults to `models/sft_cross_domain`. To use an exported
SFT-plus-RLVR checkpoint, set `LOCAL_FORECAST_MODEL_PATH` explicitly:

```bash
LOCAL_FORECAST_MODEL_PATH=/path/to/exported/checkpoint bash scripts/07_serve_local_forecaster.sh
```

`--no-api` disables API calls and exercises fallback behavior. It is useful for
an offline smoke check and does not evaluate the trained forecasting model.

## Public implementation defaults

| Setting | Public implementation |
| --- | --- |
| Retrieval | CLI `--memory-top-k 3 --memory-threshold 0.90` |
| Training-side exploration | `PARALLEL_PLAN_K=4`, up to 3 reflection attempts |
| Test inference | Up to 10 reflection attempts |
| Forecast generation | Temperature 0.3 in `forecasting/modules.py` |
| Forecast completion limit | Code default 7,000; `.env.example` explicitly uses 5,000 |
| Planning and reflection | Frozen API models; temperature 0.0 |

Settings above describe this public implementation. Reproducing a historical
table requires its actual command, checkpoint, memory, expert library, data
split, generation settings, and output records. A public default or a later
source cleanup does not establish that provenance.

The current anchorer in `anchorer/build.py` uses normalized cluster winner
counts for expert weights. Its fallback ensemble uses fixed default weights.
These paths are different from a historical-loss softmax. Use the exact
evaluated implementation and stored expert-performance records when reproducing
that formulation; do not infer its parameters from this release.

The diagnostic feature-reuse and persistent stage-recovery measurements use a
separate experiment implementation. These optimizations are not enabled by the
numbered workflow scripts in this source snapshot.

SFT trains context-response pairs exported from selected teacher responses.
RLVR scores generated forecasting responses directly using the supplied targets
and matched prior; it does not execute a full planning-reflection loop for every
sampled candidate. Planning and reflection parameters remain frozen. The
residual diagnostic tool is enabled only in the training-side workflow.

The current RLVR entry passes its input parquet to both training and validation
configuration fields. Validation on that file is a training-side diagnostic,
not an independent held-out evaluation. Preserve a separate test split for
final forecasting comparisons.

The CSV evaluator in `evaluation/evaluate.py` retains its existing parsing and
alignment rules, including positional alignment when keys do not overlap.
Use explicitly aligned prediction/target timestamps and report invalid outputs
when comparing runs. This maintenance update does not change historical scores,
reward functions, or forecast-selection behavior.

## Local checks

```bash
python -m cli --help
python -m cli forecast --help
python -m unittest discover -s tests -p 'test_*.py'
```

The configuration tests use no model API, GPU, dataset, or checkpoint.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from utils.config import CastFlowConfig, load_castflow_env
from utils.registry import (
    DATASETS,
    DatasetSpec,
    default_window_stride,
    get_dataset,
    infer_dataset_from_path,
    resolve_test_path,
    resolve_train_path,
)
from workflow.memory import StrategyMemory
from workflow.engine import CastFlow


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="CastFlow",
        description="CastFlow: role-specialized agentic workflows for time series forecasting",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    initialize = sub.add_parser(
        "initialize",
        help="validate training data, prepare experts, tune, and build cases and experience memory",
    )
    add_common_args(initialize)
    initialize.add_argument(
        "--dataset",
        default=None,
        help="registered dataset for --data, or its default training CSV",
    )
    initialize.add_argument(
        "--datasets",
        default=None,
        help="comma-separated registered datasets; defaults to all when no input is supplied",
    )
    initialize.add_argument(
        "--data-dir", default=None, help="directory containing registered training CSVs"
    )
    initialize.add_argument(
        "--dataset-manifest",
        default=None,
        help="JSON list of training dataset specifications for custom or mixed inputs",
    )
    initialize.add_argument(
        "--name", default=None, help="safe identifier for a custom --data input"
    )
    initialize.add_argument(
        "--output-root", default="../castflow_runtime/initialization"
    )
    initialize.add_argument("--model-store", default="../castflow_runtime/models")
    initialize.add_argument(
        "--models", default=None, help="comma-separated expert names; see --list-models"
    )
    initialize.add_argument(
        "--list-models",
        action="store_true",
        help="show selectable experts without preparing files or calling APIs",
    )
    initialize.add_argument(
        "--download-models",
        action="store_true",
        help="allow downloading selected missing pretrained expert weights",
    )
    initialize.add_argument(
        "--tuning-rounds",
        type=int,
        default=24,
        help="distinct parameter candidates per expert, up to 256; finite spaces may exhaust earlier",
    )
    initialize.add_argument("--tuning-windows", type=int, default=12)
    initialize.add_argument("--confirmation-windows", type=int, default=12)
    initialize.add_argument("--min-confirmation-windows", type=int, default=4)
    initialize.add_argument(
        "--quality-policy",
        choices=["required", "report"],
        default="required",
        help="require held-out prior quality before building experience memory, or explicitly prepare a diagnostic bundle",
    )
    initialize.add_argument(
        "--min-mse-improvement",
        type=float,
        default=0.01,
        help="required relative MSE improvement over the fixed seasonal baseline",
    )
    initialize.add_argument("--max-mae-regression", type=float, default=0.0)
    initialize.add_argument("--validation-fraction", type=float, default=0.2)
    initialize.add_argument("--seed", type=int, default=0)
    initialize.add_argument("--stride", type=int, default=None)
    initialize.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="bound windows per preparation stage",
    )
    initialize.add_argument("--parallel-plan-k", type=int, default=None)
    initialize.add_argument(
        "--anchors-only",
        action="store_true",
        help="prepare expert cases without model-based trajectory exploration",
    )
    initialize.add_argument("--no-resume", dest="resume", action="store_false")
    initialize.add_argument("--no-progress", action="store_true")
    initialize.add_argument("--verbose-samples", action="store_true")
    initialize.set_defaults(resume=True)

    build = sub.add_parser(
        "build-memory", help="build strategy memory from chronological windows"
    )
    add_common_args(build)
    build.add_argument(
        "--dataset",
        default=None,
        help="registered dataset name, e.g. WP, SP, BE, ETTm1",
    )
    build.add_argument(
        "--datasets",
        default=None,
        help="comma-separated registered dataset names for cross-domain mode",
    )
    build.add_argument(
        "--case-library-root",
        default="case_library",
        help="case-library root used by cross-domain build-memory",
    )
    build.add_argument(
        "--output",
        default="memory/cross_domain/memory.json",
        help="output memory JSON path",
    )
    build.add_argument(
        "--stride",
        type=int,
        default=None,
        help="window stride; defaults to the migrated dataset stride",
    )
    build.add_argument(
        "--max-windows", type=int, default=None, help="limit windows for a quick run"
    )
    build.add_argument(
        "--no-progress",
        action="store_true",
        help="disable memory build progress display",
    )
    build.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="resume from an existing memory JSON and skip completed window indices",
    )
    build.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="rebuild memory from scratch even if the output exists",
    )
    build.add_argument(
        "--excellent-mse-threshold",
        type=float,
        default=None,
        help="memory-build MSE threshold; defaults to old EXCELLENT_MSE_THRESHOLD",
    )
    build.add_argument(
        "--excellent-mae-threshold",
        type=float,
        default=None,
        help="memory-build MAE threshold; defaults to old EXCELLENT_MAE_THRESHOLD",
    )
    build.add_argument(
        "--parallel-plan-k",
        type=int,
        default=None,
        help="number of per-sample candidate tool strategies for build-memory; defaults to old PARALLEL_PLAN_K=4",
    )
    build.add_argument(
        "--verbose-samples",
        action="store_true",
        help="print per-sample planning/action/forecasting/reflection progress during build-memory",
    )
    build.set_defaults(resume=True)

    anchorer = sub.add_parser(
        "build-anchorer", help="build the Foundational Anchorer case library"
    )
    add_common_args(anchorer)
    anchorer.add_argument(
        "--dataset",
        default=None,
        help="registered dataset name, e.g. WP, SP, BE, ETTm1",
    )
    anchorer.add_argument(
        "--output",
        default=None,
        help="output anchor library JSON path for a single dataset",
    )
    anchorer.add_argument(
        "--output-root",
        default="case_library",
        help="batch output directory for split case-library JSONs",
    )
    anchorer.add_argument(
        "--data-dir",
        default=None,
        help="batch input directory; defaults to data/raw/train",
    )
    anchorer.add_argument(
        "--datasets",
        default=None,
        help="comma-separated registered dataset names for batch mode",
    )
    anchorer.add_argument(
        "--stride",
        type=int,
        default=None,
        help="window stride; defaults to the migrated case-library stride",
    )
    anchorer.add_argument(
        "--max-windows", type=int, default=None, help="limit windows for a quick run"
    )
    anchorer.add_argument(
        "--no-progress", action="store_true", help="disable build progress display"
    )
    anchorer.add_argument(
        "--expert-manifest",
        default=None,
        help="explicit expert names, adapters, and weight paths",
    )
    anchorer.add_argument(
        "--anchor-distance-weights",
        type=float,
        nargs=3,
        default=(1 / 3, 1 / 3, 1 / 3),
        metavar=("DTW", "EUCLIDEAN", "COSINE"),
    )
    anchorer.add_argument(
        "--quiet-warnings",
        action="store_true",
        help="hide known statsmodels/Chronos warnings during case building",
    )

    forecast = sub.add_parser(
        "forecast",
        help="run test-set forecasting with memory and local Forecasting model when configured",
    )
    add_common_args(forecast)
    forecast.add_argument(
        "--dataset",
        default=None,
        help="registered dataset name, e.g. WP, SP, BE, ETTm1",
    )
    forecast.add_argument("--memory", default=None, help="strategy memory JSON path")
    forecast.add_argument("--output", required=True, help="output forecast CSV path")
    forecast.add_argument(
        "--stride",
        type=int,
        default=None,
        help="test window stride; defaults to the migrated dataset stride",
    )
    forecast.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="limit test windows for a quick run",
    )
    forecast.add_argument(
        "--latest",
        action="store_true",
        help="only forecast the latest lookback window instead of all test windows",
    )
    forecast.add_argument(
        "--no-progress",
        action="store_true",
        help="disable test forecasting progress display",
    )

    prompts = sub.add_parser(
        "generate-prompts",
        help="generate prompt/ground_truth CSV from forecasting windows",
    )
    add_common_args(prompts)
    prompts.add_argument(
        "--dataset",
        default=None,
        help="registered dataset name; uses train split by default",
    )
    prompts.add_argument("--split", choices=["train", "test"], default="train")
    prompts.add_argument("--output", required=True)
    prompts.add_argument("--stride", type=int, default=None)
    prompts.add_argument("--max-windows", type=int, default=None)

    evaluate = sub.add_parser(
        "evaluate", help="evaluate answer and ground_truth columns"
    )
    evaluate.add_argument("--csv-file", required=True)
    evaluate.add_argument("--answer-col", default="answer")
    evaluate.add_argument("--ground-truth-col", default="ground_truth")
    evaluate.add_argument(
        "--output", default=None, help="optional row-level metrics CSV"
    )

    infer = sub.add_parser(
        "infer", help="run OpenAI-compatible inference over a prompt CSV"
    )
    infer.add_argument("--input", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--base-url", default=None)
    infer.add_argument("--model", default=None)
    infer.add_argument("--api-key", default=None)
    infer.add_argument("--workers", type=int, default=16)
    infer.add_argument("--timeout", type=float, default=600.0)
    infer.add_argument("--max-tokens", type=int, default=5000)
    infer.add_argument("--temperature", type=float, default=0.3)
    infer.add_argument("--prompt-col", default="prompt")
    infer.add_argument("--output-col", default="answer_local_LLM")

    teacher = sub.add_parser(
        "generate-teacher", help="generate teacher response CSV for SFT"
    )
    teacher.add_argument(
        "--input", required=True, help="prompt CSV produced by generate-prompts"
    )
    teacher.add_argument(
        "--output", required=True, help="SFT CSV with prompt and response columns"
    )
    teacher.add_argument("--base-url", default=None)
    teacher.add_argument("--model", default=None)
    teacher.add_argument("--api-key", default=None)
    teacher.add_argument("--workers", type=int, default=16)
    teacher.add_argument("--timeout", type=float, default=600.0)
    teacher.add_argument("--max-tokens", type=int, default=7000)
    teacher.add_argument("--temperature", type=float, default=0.2)

    sft = sub.add_parser(
        "train-sft",
        help="run supervised fine-tuning for the CastFlow forecasting module",
    )
    sft.add_argument("--dataset-path", default="data/sft/cross_domain_sft.csv")
    sft.add_argument("--model-path", required=True)
    sft.add_argument("--output-dir", required=True)
    sft.add_argument("--batch-size", type=int, default=1)
    sft.add_argument("--gradient-accumulation", type=int, default=None)
    sft.add_argument("--global-batch-size", type=int, default=8)
    sft.add_argument("--learning-rate", type=float, default=5e-5)
    sft.add_argument("--num-epochs", type=int, default=1)
    sft.add_argument("--max-steps", type=int, default=-1)
    sft.add_argument("--max-length", type=int, default=14000)
    sft.add_argument(
        "--save-strategy", default="steps", choices=["no", "steps", "epoch"]
    )
    sft.add_argument("--save-steps", type=int, default=100)
    sft.add_argument("--save-total-limit", type=int, default=5)
    sft.add_argument("--use-deepspeed", dest="use_deepspeed", action="store_true")
    sft.add_argument("--no-deepspeed", dest="use_deepspeed", action="store_false")
    sft.set_defaults(use_deepspeed=True)
    sft.add_argument(
        "--stage3-gather-on-save", dest="stage3_gather_on_save", action="store_true"
    )
    sft.add_argument(
        "--no-stage3-gather-on-save", dest="stage3_gather_on_save", action="store_false"
    )
    sft.set_defaults(stage3_gather_on_save=True)
    sft.add_argument("--use-lora", action="store_true")
    sft.add_argument("--lora-r", type=int, default=16)
    sft.add_argument("--lora-alpha", type=int, default=32)
    sft.add_argument("--lora-dropout", type=float, default=0.05)
    sft.add_argument("--lora-target-modules", default="")
    sft.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="restrict SFT to specific CUDA device ids, e.g. 0 or 2,3",
    )

    prepare_sft = sub.add_parser(
        "prepare-sft-data",
        help="normalize selected teacher responses into prompt,response SFT CSV",
    )
    prepare_sft.add_argument("--input", required=True)
    prepare_sft.add_argument("--output", required=True)

    export_memory = sub.add_parser(
        "export-memory-data",
        help="export selected teacher trajectories to training CSV",
    )
    export_memory.add_argument("--memory", required=True)
    export_memory.add_argument("--output", required=True)

    prepare_rl = sub.add_parser(
        "prepare-rl-data",
        help="convert fixed prompts, true labels, and matched priors to RLVR parquet",
    )
    prepare_rl.add_argument("--input", required=True)
    prepare_rl.add_argument("--output", required=True)
    prepare_rl.add_argument(
        "--dataset-name",
        default="",
        help="dataset identifier used to select the reward scale",
    )

    rlvr = sub.add_parser(
        "train-rlvr", help="run or inspect CastFlow RLVR/GRPO training config"
    )
    rlvr.add_argument("--dataset-path", default="data/rl/cross_domain_rl.parquet")
    rlvr.add_argument(
        "--validation-path",
        default=None,
        help="separate held-out training-side validation parquet",
    )
    rlvr.add_argument("--model-path", required=True)
    rlvr.add_argument("--output-dir", required=True)
    rlvr.add_argument("--port", type=int, default=30549)
    rlvr.add_argument("--train-batch-size", type=int, default=8)
    rlvr.add_argument("--rollout-n", type=int, default=8)
    rlvr.add_argument("--n-runners", type=int, default=64)
    rlvr.add_argument("--temperature", type=float, default=1.0)
    rlvr.add_argument("--top-p", type=float, default=0.95)
    rlvr.add_argument("--top-k", type=int, default=150)
    rlvr.add_argument("--repetition-penalty", type=float, default=1.0)
    rlvr.add_argument("--max-prompt-length", type=int, default=10000)
    rlvr.add_argument("--max-response-length", type=int, default=5000)
    rlvr.add_argument("--learning-rate", type=float, default=2e-6)
    rlvr.add_argument("--total-epochs", type=int, default=3)
    rlvr.add_argument("--n-gpus-per-node", type=int, default=2)
    rlvr.add_argument("--save-freq", type=int, default=100)
    rlvr.add_argument("--test-freq", type=int, default=1000)
    rlvr.add_argument("--rollout-output-dir", default=None)
    rlvr.add_argument(
        "--print-config",
        action="store_true",
        help="print migrated RLVR config without launching training",
    )

    args = parser.parse_args(argv)
    for name in (
        "stride",
        "max_windows",
        "workers",
        "lookback",
        "horizon",
        "seasonal_period",
    ):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.command == "initialize":
        from anchorer.initialize import initialize_case_libraries
        from backends.catalog import list_model_catalog

        if args.list_models:
            print(json.dumps(list_model_catalog(), indent=2))
            return 0
        specs = initialization_datasets(args)
        config = config_from_args(args)
        if not args.anchors_only and not config.api_ready():
            raise ValueError(
                "full initialization requires the configured frozen reasoning/teacher endpoint; use --anchors-only for expert preparation"
            )
        if args.anchor_library or args.execution_journal_dir or args.request_id:
            raise ValueError(
                "initialize manages its own libraries and recovery identities; do not supply an existing library or request journal"
            )
        if args.covariate_forecasts:
            if args.dataset_manifest or len(specs) != 1:
                raise ValueError(
                    "provide multi-dataset covariate provenance inside --dataset-manifest"
                )
            specs[0].update(
                covariate_forecasts_path=args.covariate_forecasts,
                covariate_issued_at=args.covariate_issued_at,
                covariate_issue_col=args.covariate_issue_col,
            )
        elif args.covariate_issued_at:
            raise ValueError("--covariate-issued-at requires --covariate-forecasts")
        if args.anchors_only and any(
            item.get("covariate_forecasts_path") for item in specs
        ):
            raise ValueError(
                "future auxiliary forecasts are used by Experience Memory exploration; remove --anchors-only to initialize that stage"
            )
        if not args.anchors_only:
            from workflow.initialize import preflight_experience_inputs

            preflight_experience_inputs(specs, config, max_windows=args.max_windows)
        result = initialize_case_libraries(
            specs,
            args.output_root,
            args.model_store,
            models=(
                None
                if args.models is None
                else [item.strip() for item in args.models.split(",")]
            ),
            tuning_rounds=args.tuning_rounds,
            tuning_windows=args.tuning_windows,
            confirmation_windows=args.confirmation_windows,
            min_confirmation_windows=args.min_confirmation_windows,
            quality_policy=args.quality_policy,
            min_mse_improvement=args.min_mse_improvement,
            max_mae_regression=args.max_mae_regression,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            download_models=args.download_models,
            max_windows=args.max_windows,
            resume=args.resume,
        )
        if not args.anchors_only:
            from workflow.initialize import initialize_experience_memories

            cases = result
            result = initialize_experience_memories(
                specs,
                cases,
                config,
                resume=args.resume,
                max_windows=args.max_windows,
                show_progress=not args.no_progress,
                verbose_samples=args.verbose_samples,
            )
            result["quality_policy"] = cases.get("quality_policy", args.quality_policy)
            result["quality_passed"] = cases.get("quality_passed", False)
            result["anchor_quality"] = {
                name: info.get("quality", {})
                for name, info in cases["datasets"].items()
            }
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0

    if args.command == "build-memory":
        if not getattr(args, "dataset", None) and not getattr(args, "data", None):
            shared_memory = (
                StrategyMemory.load(args.output)
                if args.resume and Path(args.output).exists()
                else StrategyMemory()
            )
            summaries = []
            for spec, data_path, anchor_path in resolve_batch_memory_datasets(args):
                config = config_for_dataset_args(args, spec, anchor_path)
                app = CastFlow(config=config, memory=shared_memory)
                shared_memory = app.build_memory_from_csv(
                    data_path,
                    args.output,
                    stride=resolve_window_stride(
                        args.stride, config.dataset_name, "memory"
                    ),
                    max_windows=args.max_windows,
                    show_progress=not args.no_progress,
                    resume=args.resume,
                    verbose_samples=args.verbose_samples,
                    **covariate_args(args),
                )
                dataset_entries = [
                    entry
                    for entry in shared_memory.entries
                    if entry.get("dataset_name") == spec.name
                ]
                summaries.append(
                    {
                        "dataset": spec.name,
                        "input": str(data_path),
                        "anchor_library": str(anchor_path),
                        "entries": len(dataset_entries),
                    }
                )
            print(
                json.dumps(
                    {
                        "memory_entries": len(shared_memory.entries),
                        "output": args.output,
                        "datasets": summaries,
                    },
                    indent=2,
                )
            )
            return 0

        data_path, config = data_and_config_from_args(args, split="train")
        app = CastFlow(config=config)
        memory = app.build_memory_from_csv(
            data_path,
            args.output,
            stride=resolve_window_stride(args.stride, config.dataset_name, "memory"),
            max_windows=args.max_windows,
            show_progress=not args.no_progress,
            resume=args.resume,
            verbose_samples=args.verbose_samples,
            **covariate_args(args),
        )
        print(
            json.dumps(
                {"memory_entries": len(memory.entries), "output": args.output}, indent=2
            )
        )
        return 0

    if args.command == "build-anchorer":
        from anchorer import build_anchor_library

        if not getattr(args, "dataset", None) and not getattr(args, "data", None):
            summaries = []
            for spec, data_path in resolve_batch_anchor_datasets(args):
                library = build_anchor_library(
                    data_path,
                    lookback=args.lookback or spec.lookback,
                    horizon=args.horizon or spec.horizon,
                    seasonal_period=args.seasonal_period or spec.seasonal_period,
                    dataset_name=spec.name,
                    target_col=args.target_col,
                    timestamp_col=args.timestamp_col,
                    stride=(
                        args.stride if args.stride is not None else spec.anchor_stride
                    ),
                    max_windows=args.max_windows,
                    show_progress=not args.no_progress,
                    quiet_warnings=args.quiet_warnings,
                    expert_manifest=args.expert_manifest,
                    distance_weights=args.anchor_distance_weights,
                )
                output_dir = Path(args.output_root) / case_library_dir_name(spec)
                output = library.save_case_library(output_dir, data_path=data_path)
                summaries.append(
                    {
                        "dataset": spec.name,
                        "input": str(data_path),
                        "output": str(output),
                        "cases": len(library.cases),
                        "clusters": len(library.clusters),
                        "models": sorted({case.best_model for case in library.cases}),
                    }
                )
            print(json.dumps({"outputs": summaries}, indent=2))
            return 0

        data_path, config = data_and_config_from_args(args, split="train")
        library = build_anchor_library(
            data_path,
            lookback=config.lookback,
            horizon=config.horizon,
            seasonal_period=config.seasonal_period,
            dataset_name=config.dataset_name,
            target_col=config.target_col,
            timestamp_col=config.timestamp_col,
            stride=resolve_window_stride(args.stride, config.dataset_name, "anchorer"),
            max_windows=args.max_windows,
            show_progress=not args.no_progress,
            quiet_warnings=args.quiet_warnings,
            expert_manifest=args.expert_manifest,
            distance_weights=args.anchor_distance_weights,
        )
        if args.output:
            output = library.save(args.output)
        else:
            output = library.save_case_library(
                Path(args.output_root)
                / case_library_dir_name_from_config(config.dataset_name),
                data_path=data_path,
            )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "cases": len(library.cases),
                    "clusters": len(library.clusters),
                    "models": sorted(
                        {
                            name
                            for cluster in library.clusters
                            for name in cluster.best_model
                        }
                    ),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "forecast":
        if not getattr(args, "data", None):
            raise ValueError("forecast requires an explicit test CSV via --data")
        data_path, config = data_and_config_from_args(args, split="test")
        memory = StrategyMemory.load(args.memory) if args.memory else StrategyMemory()
        app = CastFlow(config=config, memory=memory)
        result = app.forecast_csv(
            data_path,
            args.output,
            stride=resolve_window_stride(args.stride, config.dataset_name, "memory"),
            max_windows=args.max_windows,
            latest=args.latest,
            show_progress=not args.no_progress,
            **covariate_args(args),
        )
        if isinstance(result, list):
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "windows": len(result),
                        "status_counts": dict(Counter(item.status for item in result)),
                        "local_forecasting_model": (
                            config.local_model_name
                            if config.local_forecast_ready()
                            else None
                        ),
                    },
                    indent=2,
                )
            )
            return 0
        print(
            json.dumps(
                {
                    "output": args.output,
                    "horizon": len(result.forecast),
                    "status": result.status,
                    "failure_reason": result.failure_reason,
                    "tools": result.tool_schedule,
                    "reflection": result.reflection,
                    "local_forecasting_model": (
                        config.local_model_name
                        if config.local_forecast_ready()
                        else None
                    ),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "generate-prompts":
        from forecasting.prompts import write_prompt_csv

        data_path, config = data_and_config_from_args(args, split=args.split)
        output = write_prompt_csv(
            data_path,
            args.output,
            lookback=config.lookback,
            horizon=config.horizon,
            stride=resolve_window_stride(args.stride, config.dataset_name, "memory"),
            target_col=config.target_col,
            timestamp_col=config.timestamp_col,
            max_windows=args.max_windows,
            **covariate_args(args),
        )
        print(json.dumps({"output": output}, indent=2))
        return 0

    if args.command == "evaluate":
        from evaluation.evaluate import evaluate_csv

        summary = evaluate_csv(
            args.csv_file,
            answer_col=args.answer_col,
            ground_truth_col=args.ground_truth_col,
            output_path=args.output,
        )
        print(json.dumps(summary.to_dict(), indent=2, allow_nan=False))
        return 1 if summary.invalid_rows else 0

    if args.command == "infer":
        from forecasting.inference import (
            InferenceConfig,
            run_openai_compatible_inference,
        )

        base_url, model, api_key = api_args_from_env(args)
        output = run_openai_compatible_inference(
            args.input,
            args.output,
            InferenceConfig(
                base_url=base_url,
                model=model,
                api_key=api_key,
                workers=args.workers,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                prompt_col=args.prompt_col,
                output_col=args.output_col,
            ),
        )
        print(json.dumps({"output": output}, indent=2))
        return 0

    if args.command == "generate-teacher":
        from forecasting.inference import InferenceConfig
        from training.teacher import generate_teacher_sft_csv

        base_url, model, api_key = api_args_from_env(args)
        output = generate_teacher_sft_csv(
            args.input,
            args.output,
            InferenceConfig(
                base_url=base_url,
                model=model,
                api_key=api_key,
                workers=args.workers,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                prompt_col="prompt",
                output_col="answer_teacher",
            ),
        )
        print(json.dumps({"output": output}, indent=2))
        return 0

    if args.command == "train-sft":
        from training.sft import parse_sft_config, train_sft

        train_sft(parse_sft_config(args))
        return 0

    if args.command == "prepare-sft-data":
        from training.sft import prepare_sft_csv

        output = prepare_sft_csv(args.input, args.output)
        print(json.dumps({"output": output}, indent=2))
        return 0

    if args.command == "export-memory-data":
        from training.export import export_memory_training_csv

        output = export_memory_training_csv(args.memory, args.output)
        print(json.dumps({"output": output}, indent=2))
        return 0

    if args.command == "prepare-rl-data":
        from training.rlvr import prepare_rl_data

        output = prepare_rl_data(
            args.input, args.output, dataset_name=args.dataset_name
        )
        print(json.dumps({"output": output}, indent=2))
        return 0

    if args.command == "train-rlvr":
        from training.rlvr import (
            build_agent_lightning_config,
            parse_rlvr_config,
            train_rlvr,
        )

        rlvr_config = parse_rlvr_config(args)
        if args.print_config:
            print(json.dumps(build_agent_lightning_config(rlvr_config), indent=2))
            return 0
        train_rlvr(rlvr_config)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default=None, help="input CSV path")
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seasonal-period", type=int, default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument(
        "--anchor-library", default=None, help="Foundational Anchorer JSON path"
    )
    parser.add_argument("--memory-top-k", type=int, default=None)
    parser.add_argument("--memory-threshold", type=float, default=None)
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="disable CastFlow/.env API calls and use deterministic fallback",
    )
    parser.add_argument(
        "--covariate-forecasts",
        default=None,
        help="separate origin-issued auxiliary forecast CSV",
    )
    parser.add_argument(
        "--covariate-issued-at",
        default=None,
        help="issue time for a single-vintage auxiliary forecast table",
    )
    parser.add_argument("--covariate-issue-col", default="issued_at")
    parser.add_argument("--forecasting-max-tokens", type=int, default=None)
    parser.add_argument("--forecasting-temperature", type=float, default=None)
    parser.add_argument(
        "--max-retry-loops",
        type=int,
        default=None,
        help="training-side retries after the initial attempt",
    )
    parser.add_argument(
        "--test-max-retry-loops",
        type=int,
        default=None,
        help="inference retries after the initial attempt",
    )
    parser.add_argument(
        "--diagnostic-cache-policy",
        choices=["none", "all_primitives", "current_plan"],
        default=None,
    )
    parser.add_argument("--execution-journal-dir", default=None)
    parser.add_argument(
        "--request-id", default=None, help="explicit identity prefix for stage recovery"
    )
    parser.add_argument("--forecast-model-revision", default=None)
    parser.add_argument("--reasoning-model-revision", default=None)
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=None,
        help="OpenAI-compatible API timeout in seconds",
    )
    parser.add_argument(
        "--api-retries",
        type=int,
        default=None,
        help="number of OpenAI-compatible API retries after the first attempt",
    )


def covariate_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "covariate_forecasts_path": getattr(args, "covariate_forecasts", None),
        "covariate_issued_at": getattr(args, "covariate_issued_at", None),
        "covariate_issue_col": getattr(args, "covariate_issue_col", "issued_at"),
    }


def initialization_datasets(args: argparse.Namespace) -> list[dict]:
    """Resolve training inputs without creating files, splitting data, or loading models."""
    if args.dataset_manifest:
        if args.data or args.dataset or args.datasets or args.data_dir or args.name:
            raise ValueError(
                "--dataset-manifest cannot be combined with other dataset selectors"
            )
        if any(
            getattr(args, key, None) is not None
            for key in (
                "lookback",
                "horizon",
                "seasonal_period",
                "stride",
                "target_col",
                "timestamp_col",
            )
        ):
            raise ValueError(
                "put per-dataset dimensions and column settings inside --dataset-manifest"
            )
        manifest_path = Path(args.dataset_manifest).expanduser().resolve(strict=True)
        specs = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if not isinstance(specs, list) or not specs:
            raise ValueError("--dataset-manifest must contain a nonempty JSON list")
        resolved = []
        for spec in specs:
            if not isinstance(spec, dict) or not spec.get("data_path"):
                raise ValueError("every dataset specification requires data_path")
            item = dict(spec)
            for key in ("data_path", "covariate_forecasts_path"):
                if item.get(key):
                    path = Path(item[key]).expanduser()
                    item[key] = str(
                        path if path.is_absolute() else manifest_path.parent / path
                    )
            resolved.append(item)
        return resolved
    if args.data or args.dataset:
        if args.datasets or args.data_dir:
            raise ValueError(
                "single dataset selection cannot be combined with --datasets or --data-dir"
            )
        path, config = data_and_config_from_args(args, split="train")
        return [
            {
                "name": args.name or config.dataset_name or Path(path).stem,
                "data_path": path,
                "lookback": config.lookback,
                "horizon": config.horizon,
                "seasonal_period": config.seasonal_period,
                "target_col": config.target_col,
                "timestamp_col": config.timestamp_col,
                "stride": resolve_window_stride(
                    args.stride, config.dataset_name, "anchorer"
                ),
                "memory_stride": resolve_window_stride(
                    args.stride, config.dataset_name, "memory"
                ),
            }
        ]
    if args.name:
        raise ValueError("--name requires a single --data input")
    return [
        {
            "name": spec.name,
            "data_path": str(path),
            "lookback": args.lookback if args.lookback is not None else spec.lookback,
            "horizon": args.horizon if args.horizon is not None else spec.horizon,
            "seasonal_period": (
                args.seasonal_period
                if args.seasonal_period is not None
                else spec.seasonal_period
            ),
            "target_col": args.target_col,
            "timestamp_col": args.timestamp_col,
            "stride": args.stride if args.stride is not None else spec.anchor_stride,
            "memory_stride": (
                args.stride if args.stride is not None else spec.memory_stride
            ),
        }
        for spec, path in resolve_batch_anchor_datasets(args)
    ]


def config_from_args(args: argparse.Namespace) -> CastFlowConfig:
    explicit = getattr(args, "dataset", None)
    inferred = (
        infer_dataset_from_path(args.data) if getattr(args, "data", None) else None
    )
    spec = get_dataset(explicit) if explicit else inferred
    if explicit and inferred and spec.name != inferred.name:
        raise ValueError("--dataset conflicts with the registered input filename")
    defaults = {"lookback": 96, "horizon": 96, "seasonal_period": 24}
    values = {
        name: (
            getattr(args, name, None)
            if getattr(args, name, None) is not None
            else getattr(spec, name) if spec else default
        )
        for name, default in defaults.items()
    }
    fields = {
        "target_col": "target_col",
        "timestamp_col": "timestamp_col",
        "anchor_library_path": "anchor_library",
        "memory_top_k": "memory_top_k",
        "memory_similarity_threshold": "memory_threshold",
        "api_timeout": "api_timeout",
        "api_max_retries": "api_retries",
        "excellent_mse_threshold": "excellent_mse_threshold",
        "excellent_mae_threshold": "excellent_mae_threshold",
        "parallel_plan_k": "parallel_plan_k",
        "forecasting_max_tokens": "forecasting_max_tokens",
        "forecasting_temperature": "forecasting_temperature",
        "max_retry_loops": "max_retry_loops",
        "test_max_retry_loops": "test_max_retry_loops",
        "diagnostic_cache_policy": "diagnostic_cache_policy",
        "execution_journal_dir": "execution_journal_dir",
        "execution_request_id": "request_id",
        "forecasting_model_revision": "forecast_model_revision",
        "reasoning_model_revision": "reasoning_model_revision",
    }
    values.update({field: getattr(args, name, None) for field, name in fields.items()})
    values["dataset_name"] = spec.name if spec else None
    values["use_api"] = False if getattr(args, "no_api", False) else None
    config = CastFlowConfig.from_env(**values)
    if bool(config.execution_journal_dir) != bool(config.execution_request_id):
        raise ValueError(
            "stage recovery requires both --execution-journal-dir and --request-id"
        )
    return config


def data_and_config_from_args(
    args: argparse.Namespace, split: str
) -> tuple[str, CastFlowConfig]:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    config = config_from_args(args)
    if getattr(args, "data", None):
        return str(args.data), config
    if config.dataset_name:
        spec = get_dataset(config.dataset_name)
        return (
            str(
                resolve_train_path(spec)
                if split == "train"
                else resolve_test_path(spec)
            ),
            config,
        )
    raise ValueError("--data is required when --dataset is not provided")


def resolve_window_stride(
    arg_stride: int | None, dataset_name: str | None, kind: str
) -> int | None:
    if arg_stride is not None:
        return arg_stride
    return default_window_stride(dataset_name, kind)


def api_args_from_env(args: argparse.Namespace) -> tuple[str, str, str]:
    env = load_castflow_env()
    local = getattr(args, "command", "") == "infer"
    base_url = args.base_url or env.get(
        "LOCAL_MODEL_BASE_URL" if local else "OPENAI_BASE_URL"
    )
    model = args.model or env.get("LOCAL_MODEL_NAME" if local else "MODEL")
    api_key = args.api_key or env.get(
        "LOCAL_MODEL_API_KEY" if local else "OPENAI_API_KEY"
    )
    if not base_url or not model or not api_key:
        raise ValueError(
            "configure the model endpoint, model name, and credential before inference"
        )
    return base_url, model, api_key


def resolve_batch_anchor_datasets(
    args: argparse.Namespace,
) -> list[tuple[DatasetSpec, Path]]:
    selected_names = (
        parse_dataset_list(args.datasets) if args.datasets else sorted(DATASETS)
    )
    data_dir = Path(args.data_dir) if args.data_dir else None
    resolved: list[tuple[DatasetSpec, Path]] = []
    missing: list[str] = []
    for name in selected_names:
        spec = get_dataset(name)
        path = resolve_batch_train_path(spec, data_dir)
        if path is None:
            missing.append(f"{spec.name}: {spec.train_path}")
            continue
        resolved.append((spec, path))
    if missing:
        raise FileNotFoundError("missing batch anchor datasets: " + ", ".join(missing))
    if not resolved:
        raise ValueError("no datasets selected for batch anchor build")
    return resolved


def resolve_batch_memory_datasets(
    args: argparse.Namespace,
) -> list[tuple[DatasetSpec, Path, Path]]:
    selected_names = (
        parse_dataset_list(args.datasets) if args.datasets else sorted(DATASETS)
    )
    resolved: list[tuple[DatasetSpec, Path, Path]] = []
    missing: list[str] = []
    for name in selected_names:
        spec = get_dataset(name)
        data_path = resolve_train_path(spec)
        anchor_path = (
            Path(args.case_library_root)
            / case_library_dir_name(spec)
            / "anchor_library.json"
        )
        if not data_path.exists():
            missing.append(f"{spec.name}: {data_path}")
            continue
        if not anchor_path.exists():
            missing.append(f"{spec.name}: {anchor_path}")
            continue
        resolved.append((spec, data_path, anchor_path))
    if missing:
        raise FileNotFoundError(
            "missing cross-domain memory inputs: " + ", ".join(missing)
        )
    if not resolved:
        raise ValueError("no datasets selected for cross-domain memory build")
    return resolved


def config_for_dataset_args(
    args: argparse.Namespace, spec: DatasetSpec, anchor_path: Path
) -> CastFlowConfig:
    overrides = argparse.Namespace(**vars(args))
    overrides.dataset = spec.name
    overrides.data = str(resolve_train_path(spec))
    overrides.anchor_library = str(anchor_path)
    return config_from_args(overrides)


def parse_dataset_list(value: str) -> list[str]:
    names = [
        get_dataset(item.strip()).name for item in value.split(",") if item.strip()
    ]
    if not names:
        raise ValueError("select at least one dataset")
    return list(dict.fromkeys(names))


def resolve_batch_train_path(spec: DatasetSpec, data_dir: Path | None) -> Path | None:
    if data_dir is None:
        path = resolve_train_path(spec)
        return path if path.exists() else None
    candidates = [
        data_dir / spec.train_path,
        data_dir / "train" / Path(spec.train_path).name,
        data_dir / Path(spec.train_path).name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def case_library_dir_name(spec: DatasetSpec) -> str:
    stem = Path(spec.train_path).stem
    for suffix in ("_train_val", "_train"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def case_library_dir_name_from_config(dataset_name: str | None) -> str:
    if dataset_name:
        return case_library_dir_name(get_dataset(dataset_name))
    return "custom"


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import CastFlowConfig, load_castflow_env
from .datasets import DATASETS, DatasetSpec, default_window_stride, get_dataset, infer_dataset_from_path, resolve_test_path, resolve_train_path
from .memory import StrategyMemory
from .workflow import CastFlow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="CastFlow",
        description="CastFlow: role-specialized agentic workflows for time series forecasting",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-memory", help="build strategy memory from chronological windows")
    add_common_args(build)
    build.add_argument("--dataset", default=None, help="registered dataset name, e.g. WP, SP, BE, ETTm1")
    build.add_argument("--datasets", default=None, help="comma-separated registered dataset names for cross-domain mode")
    build.add_argument("--case-library-root", default="case_library", help="case-library root used by cross-domain build-memory")
    build.add_argument("--output", default="memory/cross_domain/memory.json", help="output memory JSON path")
    build.add_argument("--stride", type=int, default=None, help="window stride; defaults to the migrated dataset stride")
    build.add_argument("--max-windows", type=int, default=None, help="limit windows for a quick run")
    build.add_argument("--no-progress", action="store_true", help="disable memory build progress display")
    build.add_argument("--resume", dest="resume", action="store_true", help="resume from an existing memory JSON and skip completed window indices")
    build.add_argument("--no-resume", dest="resume", action="store_false", help="rebuild memory from scratch even if the output exists")
    build.add_argument("--excellent-mse-threshold", type=float, default=None, help="memory-build MSE threshold; defaults to old EXCELLENT_MSE_THRESHOLD")
    build.add_argument("--excellent-mae-threshold", type=float, default=None, help="memory-build MAE threshold; defaults to old EXCELLENT_MAE_THRESHOLD")
    build.add_argument("--parallel-plan-k", type=int, default=None, help="number of per-sample candidate tool strategies for build-memory; defaults to old PARALLEL_PLAN_K=4")
    build.add_argument("--verbose-samples", action="store_true", help="print per-sample planning/action/forecasting/reflection progress during build-memory")
    build.set_defaults(resume=True)

    anchorer = sub.add_parser("build-anchorer", help="build the Foundational Anchorer case library")
    add_common_args(anchorer)
    anchorer.add_argument("--dataset", default=None, help="registered dataset name, e.g. WP, SP, BE, ETTm1")
    anchorer.add_argument("--output", default=None, help="output anchor library JSON path for a single dataset")
    anchorer.add_argument("--output-root", default="case_library", help="batch output directory for split case-library JSONs")
    anchorer.add_argument("--data-dir", default=None, help="batch input directory; defaults to data/raw/train")
    anchorer.add_argument("--datasets", default=None, help="comma-separated registered dataset names for batch mode")
    anchorer.add_argument("--stride", type=int, default=None, help="window stride; defaults to the migrated case-library stride")
    anchorer.add_argument("--max-windows", type=int, default=None, help="limit windows for a quick run")
    anchorer.add_argument("--no-progress", action="store_true", help="disable build progress display")
    anchorer.add_argument("--quiet-warnings", action="store_true", help="hide known statsmodels/Chronos warnings during case building")

    forecast = sub.add_parser("forecast", help="run test-set forecasting with memory and local Forecasting model when configured")
    add_common_args(forecast)
    forecast.add_argument("--dataset", default=None, help="registered dataset name, e.g. WP, SP, BE, ETTm1")
    forecast.add_argument("--memory", default=None, help="strategy memory JSON path")
    forecast.add_argument("--output", required=True, help="output forecast CSV path")
    forecast.add_argument("--stride", type=int, default=None, help="test window stride; defaults to the migrated dataset stride")
    forecast.add_argument("--max-windows", type=int, default=None, help="limit test windows for a quick run")
    forecast.add_argument("--latest", action="store_true", help="only forecast the latest lookback window instead of all test windows")
    forecast.add_argument("--no-progress", action="store_true", help="disable test forecasting progress display")

    prompts = sub.add_parser("generate-prompts", help="generate prompt/ground_truth CSV from forecasting windows")
    add_common_args(prompts)
    prompts.add_argument("--dataset", default=None, help="registered dataset name; uses train split by default")
    prompts.add_argument("--split", choices=["train", "test"], default="train")
    prompts.add_argument("--output", required=True)
    prompts.add_argument("--stride", type=int, default=None)
    prompts.add_argument("--max-windows", type=int, default=None)

    evaluate = sub.add_parser("evaluate", help="evaluate answer and ground_truth columns")
    evaluate.add_argument("--csv-file", required=True)
    evaluate.add_argument("--answer-col", default="answer")
    evaluate.add_argument("--ground-truth-col", default="ground_truth")
    evaluate.add_argument("--output", default=None, help="optional row-level metrics CSV")

    infer = sub.add_parser("infer", help="run OpenAI-compatible inference over a prompt CSV")
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

    teacher = sub.add_parser("generate-teacher", help="generate teacher response CSV for SFT")
    teacher.add_argument("--input", required=True, help="prompt CSV produced by generate-prompts")
    teacher.add_argument("--output", required=True, help="SFT CSV with prompt and response columns")
    teacher.add_argument("--base-url", default=None)
    teacher.add_argument("--model", default=None)
    teacher.add_argument("--api-key", default=None)
    teacher.add_argument("--workers", type=int, default=16)
    teacher.add_argument("--timeout", type=float, default=600.0)
    teacher.add_argument("--max-tokens", type=int, default=7000)
    teacher.add_argument("--temperature", type=float, default=0.2)

    sft = sub.add_parser("train-sft", help="run supervised fine-tuning for the CastFlow forecasting module")
    sft.add_argument("--dataset-path", default="data/sft_rl/windy_grok_no_memory.csv")
    sft.add_argument("--model-path", required=True)
    sft.add_argument("--output-dir", required=True)
    sft.add_argument("--batch-size", type=int, default=1)
    sft.add_argument("--gradient-accumulation", type=int, default=4)
    sft.add_argument("--learning-rate", type=float, default=5e-5)
    sft.add_argument("--num-epochs", type=int, default=1)
    sft.add_argument("--max-steps", type=int, default=-1)
    sft.add_argument("--max-length", type=int, default=14000)
    sft.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch"])
    sft.add_argument("--save-steps", type=int, default=100)
    sft.add_argument("--save-total-limit", type=int, default=5)
    sft.add_argument("--use-deepspeed", dest="use_deepspeed", action="store_true")
    sft.add_argument("--no-deepspeed", dest="use_deepspeed", action="store_false")
    sft.set_defaults(use_deepspeed=True)
    sft.add_argument("--stage3-gather-on-save", dest="stage3_gather_on_save", action="store_true")
    sft.add_argument("--no-stage3-gather-on-save", dest="stage3_gather_on_save", action="store_false")
    sft.set_defaults(stage3_gather_on_save=True)
    sft.add_argument("--use-lora", action="store_true")
    sft.add_argument("--lora-r", type=int, default=16)
    sft.add_argument("--lora-alpha", type=int, default=32)
    sft.add_argument("--lora-dropout", type=float, default=0.05)
    sft.add_argument("--lora-target-modules", default="")
    sft.add_argument("--cuda-visible-devices", default=None, help="restrict SFT to specific CUDA device ids, e.g. 0 or 2,3")

    prepare_sft = sub.add_parser("prepare-sft-data", help="normalize old/evaluation CSVs into prompt,response SFT CSV")
    prepare_sft.add_argument("--input", required=True)
    prepare_sft.add_argument("--output", required=True)

    export_memory = sub.add_parser("export-memory-data", help="export memory JSON to old grok.csv training schema")
    export_memory.add_argument("--memory", required=True)
    export_memory.add_argument("--output", required=True)

    prepare_rl = sub.add_parser("prepare-rl-data", help="convert prompt/response CSV to RLVR parquet")
    prepare_rl.add_argument("--input", required=True)
    prepare_rl.add_argument("--output", required=True)
    prepare_rl.add_argument("--dataset-name", default="", help="dataset name used by the rl_agent.py-equivalent reward")

    rlvr = sub.add_parser("train-rlvr", help="run or inspect CastFlow RLVR/GRPO training config")
    rlvr.add_argument("--dataset-path", default="data/sft_rl/windy_grok_no_memory.parquet")
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
    rlvr.add_argument("--learning-rate", type=float, default=2e-6)
    rlvr.add_argument("--total-epochs", type=int, default=3)
    rlvr.add_argument("--n-gpus-per-node", type=int, default=2)
    rlvr.add_argument("--save-freq", type=int, default=100)
    rlvr.add_argument("--test-freq", type=int, default=1000)
    rlvr.add_argument("--rollout-output-dir", default=None)
    rlvr.add_argument("--print-config", action="store_true", help="print migrated RLVR config without launching training")

    args = parser.parse_args(argv)

    if args.command == "build-memory":
        if not getattr(args, "dataset", None) and not getattr(args, "data", None):
            shared_memory = StrategyMemory.load(args.output) if args.resume and Path(args.output).exists() else StrategyMemory()
            summaries = []
            for spec, data_path, anchor_path in resolve_batch_memory_datasets(args):
                config = config_for_dataset_args(args, spec, anchor_path)
                app = CastFlow(config=config, memory=shared_memory)
                shared_memory = app.build_memory_from_csv(
                    data_path,
                    args.output,
                    stride=resolve_window_stride(args.stride, config.dataset_name, "memory"),
                    max_windows=args.max_windows,
                    show_progress=not args.no_progress,
                    resume=args.resume,
                    verbose_samples=args.verbose_samples,
                )
                dataset_entries = [entry for entry in shared_memory.entries if entry.get("dataset_name") == spec.name]
                summaries.append(
                    {
                        "dataset": spec.name,
                        "input": str(data_path),
                        "anchor_library": str(anchor_path),
                        "entries": len(dataset_entries),
                    }
                )
            print(json.dumps({"memory_entries": len(shared_memory.entries), "output": args.output, "datasets": summaries}, indent=2))
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
        )
        print(json.dumps({"memory_entries": len(memory.entries), "output": args.output}, indent=2))
        return 0

    if args.command == "build-anchorer":
        from .anchorer import build_anchor_library

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
                    stride=args.stride if args.stride is not None else spec.anchor_stride,
                    max_windows=args.max_windows,
                    show_progress=not args.no_progress,
                    quiet_warnings=args.quiet_warnings,
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
        )
        if args.output:
            output = library.save(args.output)
        else:
            output = library.save_case_library(Path(args.output_root) / case_library_dir_name_from_config(config.dataset_name), data_path=data_path)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "cases": len(library.cases),
                    "clusters": len(library.clusters),
                    "models": sorted({name for cluster in library.clusters for name in cluster.best_model}),
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
        )
        if isinstance(result, list):
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "windows": len(result),
                        "local_forecasting_model": config.local_model_name if config.local_forecast_ready() else None,
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
                    "tools": result.tool_schedule,
                    "reflection": result.reflection,
                    "local_forecasting_model": config.local_model_name if config.local_forecast_ready() else None,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "generate-prompts":
        from .prompts import write_prompt_csv

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
        )
        print(json.dumps({"output": output}, indent=2))
        return 0

    if args.command == "evaluate":
        from .evaluation import evaluate_csv

        summary = evaluate_csv(
            args.csv_file,
            answer_col=args.answer_col,
            ground_truth_col=args.ground_truth_col,
            output_path=args.output,
        )
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    if args.command == "infer":
        from .inference import InferenceConfig, run_openai_compatible_inference

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
        from .inference import InferenceConfig
        from .training.teacher import generate_teacher_sft_csv

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
        from .training.sft import parse_sft_config, train_sft

        train_sft(parse_sft_config(args))
        return 0

    if args.command == "prepare-sft-data":
        from .training.sft import prepare_sft_csv

        output = prepare_sft_csv(args.input, args.output)
        print(json.dumps({"output": output}, indent=2))
        return 0

    if args.command == "export-memory-data":
        from .training.export import export_memory_training_csv

        output = export_memory_training_csv(args.memory, args.output)
        print(json.dumps({"output": output}, indent=2))
        return 0

    if args.command == "prepare-rl-data":
        from .training.rlvr import prepare_rl_data

        output = prepare_rl_data(args.input, args.output, dataset_name=args.dataset_name)
        print(json.dumps({"output": output}, indent=2))
        return 0

    if args.command == "train-rlvr":
        from .training.rlvr import build_agent_lightning_config, parse_rlvr_config, train_rlvr

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
    parser.add_argument("--anchor-library", default=None, help="Foundational Anchorer JSON path")
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--memory-threshold", type=float, default=0.90)
    parser.add_argument("--no-api", action="store_true", help="disable CastFlow/.env API calls and use deterministic fallback")
    parser.add_argument("--api-timeout", type=float, default=None, help="OpenAI-compatible API timeout in seconds")
    parser.add_argument("--api-retries", type=int, default=None, help="number of OpenAI-compatible API retries after the first attempt")


def config_from_args(args: argparse.Namespace) -> CastFlowConfig:
    inferred_spec = infer_dataset_from_path(args.data) if getattr(args, "data", None) else None
    return CastFlowConfig.from_env(
        lookback=args.lookback or (inferred_spec.lookback if inferred_spec else 96),
        horizon=args.horizon or (inferred_spec.horizon if inferred_spec else 96),
        seasonal_period=args.seasonal_period or (inferred_spec.seasonal_period if inferred_spec else 24),
        target_col=args.target_col,
        timestamp_col=args.timestamp_col,
        dataset_name=getattr(args, "dataset", None) or (inferred_spec.name if inferred_spec else None),
        anchor_library_path=getattr(args, "anchor_library", None),
        memory_top_k=args.memory_top_k,
        memory_similarity_threshold=args.memory_threshold,
        use_api=False if getattr(args, "no_api", False) else None,
        api_timeout=getattr(args, "api_timeout", None),
        api_max_retries=getattr(args, "api_retries", None),
        excellent_mse_threshold=getattr(args, "excellent_mse_threshold", None),
        excellent_mae_threshold=getattr(args, "excellent_mae_threshold", None),
        parallel_plan_k=getattr(args, "parallel_plan_k", None),
    )


def data_and_config_from_args(args: argparse.Namespace, split: str) -> tuple[str, CastFlowConfig]:
    dataset_name = getattr(args, "dataset", None)
    if dataset_name:
        spec = get_dataset(dataset_name)
        data_path = str(resolve_train_path(spec) if split == "train" else resolve_test_path(spec))
        config = CastFlowConfig.from_env(
            lookback=getattr(args, "lookback", spec.lookback) or spec.lookback,
            horizon=getattr(args, "horizon", spec.horizon) or spec.horizon,
            seasonal_period=getattr(args, "seasonal_period", spec.seasonal_period) or spec.seasonal_period,
            target_col=getattr(args, "target_col", None),
            timestamp_col=getattr(args, "timestamp_col", None),
            dataset_name=spec.name,
            anchor_library_path=getattr(args, "anchor_library", None),
            memory_top_k=getattr(args, "memory_top_k", 3),
            memory_similarity_threshold=getattr(args, "memory_threshold", 0.90),
            use_api=False if getattr(args, "no_api", False) else None,
            api_timeout=getattr(args, "api_timeout", None),
            api_max_retries=getattr(args, "api_retries", None),
            excellent_mse_threshold=getattr(args, "excellent_mse_threshold", None),
            excellent_mae_threshold=getattr(args, "excellent_mae_threshold", None),
            parallel_plan_k=getattr(args, "parallel_plan_k", None),
        )
        return data_path, config

    if not getattr(args, "data", None):
        raise ValueError("--data is required when --dataset is not provided")
    return args.data, config_from_args(args)


def resolve_window_stride(arg_stride: int | None, dataset_name: str | None, kind: str) -> int | None:
    if arg_stride is not None:
        return arg_stride
    return default_window_stride(dataset_name, kind)


def api_args_from_env(args: argparse.Namespace) -> tuple[str, str, str]:
    env = load_castflow_env()
    base_url = args.base_url or env.get("OPENAI_BASE_URL") or "http://localhost:8003/v1"
    model = args.model or env.get("MODEL") or "forecast"
    api_key = args.api_key or env.get("OPENAI_API_KEY") or "test-key"
    return base_url, model, api_key


def resolve_batch_anchor_datasets(args: argparse.Namespace) -> list[tuple[DatasetSpec, Path]]:
    selected_names = parse_dataset_list(args.datasets) if args.datasets else sorted(DATASETS)
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


def resolve_batch_memory_datasets(args: argparse.Namespace) -> list[tuple[DatasetSpec, Path, Path]]:
    selected_names = parse_dataset_list(args.datasets) if args.datasets else sorted(DATASETS)
    resolved: list[tuple[DatasetSpec, Path, Path]] = []
    missing: list[str] = []
    for name in selected_names:
        spec = get_dataset(name)
        data_path = resolve_train_path(spec)
        anchor_path = Path(args.case_library_root) / case_library_dir_name(spec) / "anchor_library.json"
        if not data_path.exists():
            missing.append(f"{spec.name}: {data_path}")
            continue
        if not anchor_path.exists():
            missing.append(f"{spec.name}: {anchor_path}")
            continue
        resolved.append((spec, data_path, anchor_path))
    if missing:
        raise FileNotFoundError("missing cross-domain memory inputs: " + ", ".join(missing))
    if not resolved:
        raise ValueError("no datasets selected for cross-domain memory build")
    return resolved


def config_for_dataset_args(args: argparse.Namespace, spec: DatasetSpec, anchor_path: Path) -> CastFlowConfig:
    return CastFlowConfig.from_env(
        lookback=getattr(args, "lookback", spec.lookback) or spec.lookback,
        horizon=getattr(args, "horizon", spec.horizon) or spec.horizon,
        seasonal_period=getattr(args, "seasonal_period", spec.seasonal_period) or spec.seasonal_period,
        target_col=getattr(args, "target_col", None),
        timestamp_col=getattr(args, "timestamp_col", None),
        dataset_name=spec.name,
        anchor_library_path=str(anchor_path),
        memory_top_k=getattr(args, "memory_top_k", 3),
        memory_similarity_threshold=getattr(args, "memory_threshold", 0.90),
        use_api=False if getattr(args, "no_api", False) else None,
        api_timeout=getattr(args, "api_timeout", None),
        api_max_retries=getattr(args, "api_retries", None),
        excellent_mse_threshold=getattr(args, "excellent_mse_threshold", None),
        excellent_mae_threshold=getattr(args, "excellent_mae_threshold", None),
        parallel_plan_k=getattr(args, "parallel_plan_k", None),
    )


def parse_dataset_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


if __name__ == "__main__":
    raise SystemExit(main())

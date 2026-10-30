"""Explicit training-side selection of pre-frozen, whole-library policies."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from anchorer.build import AnchorLibrary, historical_loss_weights
from anchorer.policies import POLICIES, build_promotion_candidates
from backends.catalog import content_fingerprint, file_sha256
from utils.io import atomic_write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _record(path, value):
    atomic_write_json(path, {"payload": value, "sha256": content_fingerprint(value)})


def _load_record(path):
    if not path.exists():
        return None
    value = _read(path)
    if (
        not isinstance(value, dict)
        or "payload" not in value
        or value.get("sha256") != content_fingerprint(value["payload"])
    ):
        raise ValueError(f"promotion record checksum mismatch: {path}")
    return value["payload"]


def _save_once(path, value):
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"frozen promotion artifact changed: {path}")
    else:
        atomic_write_json(path, value)
    return file_sha256(path)


def _operational_identity(library):
    payload = library.to_dict()
    payload.pop("promotion", None)
    for cluster in payload["clusters"]:
        losses = cluster.get("weighting_losses")
        if losses is None:
            losses = cluster["model_errors"]
        cluster["weighting_losses"] = historical_loss_weights(losses)
    return content_fingerprint(payload)


def select_promoted_library(
    library,
    prepared_library_path,
    directory,
    *,
    calibration_records,
    selection_incumbent_name,
    pipeline_fingerprint,
    protocol,
    requirements,
    evaluate,
    quality_policy,
    promotion_end_row_exclusive,
    promotion_last_timestamp,
):
    """Freeze all policy artifacts before scoring the later promotion interval.

    Equivalent operational policies share one evaluation, while every declared
    candidate retains its own immutable artifact and report. Only one global
    policy can be selected. This stage never claims independent quality.
    """
    directory, prepared_library_path = Path(directory), Path(prepared_library_path)
    prepared_sha = file_sha256(prepared_library_path)
    candidates = build_promotion_candidates(
        library,
        calibration_records=calibration_records,
        selection_incumbent_name=selection_incumbent_name,
        pipeline_fingerprint=pipeline_fingerprint,
    )
    if tuple(candidates) != POLICIES:
        raise ValueError("promotion must freeze the three declared policies in order")
    rows, groups = [], {}
    for policy, candidate in candidates.items():
        path = directory / "promotion_candidates" / f"{policy}.json"
        payload = candidate.to_dict()
        payload["initialization"] = {
            "pipeline_fingerprint": pipeline_fingerprint,
            **protocol,
        }
        candidate_sha = _save_once(path, payload)
        operational = _operational_identity(candidate)
        groups.setdefault(operational, []).append(policy)
        rows.append(
            {
                "policy": policy,
                "path": str(path),
                "sha256": candidate_sha,
                "operational_fingerprint": operational,
            }
        )
    manifest = {
        "status": "frozen_before_promotion_targets",
        "policy": "conservative",
        "pipeline_fingerprint": pipeline_fingerprint,
        "prepared_library": str(prepared_library_path),
        "prepared_library_sha256": prepared_sha,
        "promotion_origins": protocol["confirmation_origins"],
        "promotion_start_row": protocol["confirmation_start_row"],
        "promotion_end_row_exclusive": promotion_end_row_exclusive,
        "promotion_last_timestamp": promotion_last_timestamp,
        "requirements": requirements,
        "candidates": rows,
        "evaluation_groups": list(groups.values()),
        "independent_quality_verified": False,
    }
    manifest_path = directory / "promotion_candidates.json"
    previous = _load_record(manifest_path)
    if previous is not None and previous != manifest:
        raise ValueError("promotion candidate freeze or requirements changed")
    if previous is None:
        _record(manifest_path, manifest)
    manifest_sha = file_sha256(manifest_path)

    def verify_frozen():
        if (
            file_sha256(prepared_library_path) != prepared_sha
            or _load_record(manifest_path) != manifest
        ):
            raise ValueError("prepared library or promotion manifest changed")
        if file_sha256(manifest_path) != manifest_sha:
            raise ValueError("promotion manifest bytes changed")
        for row in rows:
            if file_sha256(row["path"]) != row["sha256"]:
                raise ValueError("frozen promotion candidate changed")

    # No callback capable of consulting promotion labels is invoked before this.
    verify_frozen()
    reports = {}
    for aliases in groups.values():
        representative = aliases[0]
        row = next(value for value in rows if value["policy"] == representative)
        cache_path = directory / "candidate_reports" / f"{representative}.complete.json"
        report_path = directory / "candidate_reports" / f"{representative}.json"
        completed = _load_record(cache_path)
        if completed is None:
            verify_frozen()
            report = evaluate(
                AnchorLibrary.load(row["path"]),
                directory / "promotion_evaluation" / representative,
            )
            report = {
                **report,
                "scope": "training_side_promotion_selection",
                "independent_quality_verified": False,
                "candidate_policy": representative,
                "policy_aliases": aliases,
                "candidate_library_sha256": row["sha256"],
                "candidate_manifest_sha256": manifest_sha,
            }
            verify_frozen()
            _save_once(report_path, report)
            _record(
                cache_path,
                {
                    "manifest_sha256": manifest_sha,
                    "candidate_sha256": row["sha256"],
                    "report_sha256": file_sha256(report_path),
                },
            )
        else:
            if completed != {
                "manifest_sha256": manifest_sha,
                "candidate_sha256": row["sha256"],
                "report_sha256": file_sha256(report_path),
            }:
                raise ValueError("completed promotion report changed")
            report = _read(report_path)
        for alias in aliases:
            reports[alias] = report
    original_report_path = directory / "candidate_report.json"
    _save_once(original_report_path, reports["mixture"])
    eligible = [
        policy for policy in POLICIES if reports[policy].get("quality_passed") is True
    ]
    selected = (
        min(
            eligible,
            key=lambda policy: (
                reports[policy]["candidate"]["mse"],
                reports[policy]["candidate"]["mae"],
                1 if policy == "mixture" else 0,
                POLICIES.index(policy),
            ),
        )
        if eligible
        else None
    )
    promotion = {
        "policy": "conservative",
        "selected_policy": selected,
        "status": "selected" if selected else "no_qualified_policy",
        "scope": "training_side_promotion_selection",
        "independent_quality_verified": False,
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_sha256": manifest_sha,
        "candidate_count": len(rows),
        "evaluated_policy_count": len(groups),
        "promotion_start_row": protocol["confirmation_start_row"],
        "promotion_end_row_exclusive": promotion_end_row_exclusive,
        "promotion_last_timestamp": promotion_last_timestamp,
    }
    result_path = directory / "promoted_anchor_library.json"
    if selected:
        chosen = next(row for row in rows if row["policy"] == selected)
        promoted = AnchorLibrary.load(chosen["path"])
        promoted.promotion.update(
            status="selected",
            selected_policy=selected,
            selection_policy="conservative",
            promotion_labels_used_for_policy_selection=True,
            candidate_manifest_sha256=manifest_sha,
            selected_candidate_sha256=chosen["sha256"],
            promotion_end_row_exclusive=promotion_end_row_exclusive,
            promotion_last_timestamp=promotion_last_timestamp,
        )
        payload = promoted.to_dict()
        payload["initialization"] = {
            "pipeline_fingerprint": pipeline_fingerprint,
            **protocol,
        }
        _save_once(result_path, payload)
        AnchorLibrary.load(result_path)
        artifact_role = "promoted_global_policy"
    else:
        if result_path.exists():
            raise ValueError(
                "unqualified promotion conflicts with an existing promoted library"
            )
        result_path = prepared_library_path
        artifact_role = "unpromoted_diagnostic"
    report = copy.deepcopy(reports[selected or "mixture"])
    report.update(
        promotion=promotion,
        evaluation_representative_policy=report["candidate_policy"],
        candidate_policy=selected or "mixture",
        artifact_role=artifact_role,
        quality_passed=bool(selected),
        quality_policy=quality_policy,
        scope="training_side_promotion_selection",
        independent_quality_verified=False,
        library_sha256=file_sha256(result_path),
        prepared_library_sha256=prepared_sha,
        candidate_reports={
            policy: {
                "quality_passed": reports[policy]["quality_passed"],
                "candidate": reports[policy]["candidate"],
                "failed_checks": reports[policy]["failed_checks"],
                "evaluated_policy": reports[policy]["candidate_policy"],
            }
            for policy in POLICIES
        },
    )
    report_path = directory / "promotion_report.json"
    _save_once(report_path, report)
    completion = {
        "candidate_manifest_sha256": manifest_sha,
        "prepared_library_sha256": prepared_sha,
        "library_sha256": file_sha256(result_path),
        "report_sha256": file_sha256(report_path),
        "prepared_report_sha256": file_sha256(original_report_path),
        "selected_policy": selected,
    }
    completed = _load_record(directory / "promotion_complete.json")
    if completed is not None and completed != completion:
        raise ValueError("completed promotion selection or artifact changed")
    if completed is None:
        _record(directory / "promotion_complete.json", completion)
    verify_frozen()
    if not selected and quality_policy == "required":
        raise ValueError(
            f"no frozen policy passed required promotion quality; see {report_path}"
        )
    return {
        "library": AnchorLibrary.load(result_path),
        "anchor_library": str(result_path),
        "quality_report": str(report_path),
        "prepared_quality_report": str(original_report_path),
        "report": report,
        "promotion": promotion,
        "artifact_role": artifact_role,
    }

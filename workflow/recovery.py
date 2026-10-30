"""Durable, single-writer recovery of completed stages within one request."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class JournalMismatch(ValueError):
    """Stored stages belong to different inputs or fail integrity checks."""


class ExecutionJournal:
    """Use explicit request identities; external calls remain at-least-once."""

    def __init__(
        self, directory: str | Path, request_id: str, identity: dict[str, Any]
    ) -> None:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError(
                "an explicit nonempty request_id is required for stage recovery"
            )
        identity_hash = digest(identity)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / (
            hashlib.sha256(request_id.encode("utf-8")).hexdigest() + ".json"
        )
        self._closed = False
        self._lock_fd = os.open(
            self.path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600
        )
        try:
            self._acquire()
        except BaseException:
            os.close(self._lock_fd)
            self._closed = True
            raise
        self.data = {
            "version": 1,
            "request_id": request_id,
            "identity_hash": identity_hash,
            "stages": {},
        }
        try:
            if self.path.exists():
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if any(
                    saved.get(key) != self.data[key]
                    for key in ("version", "request_id", "identity_hash")
                ):
                    raise JournalMismatch(
                        "request inputs, configuration, or implementation identity changed"
                    )
                if not isinstance(saved.get("stages"), dict):
                    raise JournalMismatch("invalid stage journal structure")
                for row in saved["stages"].values():
                    if not isinstance(row, dict) or row.get("output_hash") != digest(
                        row.get("output")
                    ):
                        raise JournalMismatch("stage output integrity mismatch")
                self.data = saved
            else:
                self._commit()
        except BaseException:
            self.close()
            raise

    def _acquire(self) -> None:
        if os.name == "nt":
            import msvcrt

            if os.fstat(self._lock_fd).st_size == 0:
                os.write(self._lock_fd, b"\0")
            os.lseek(self._lock_fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(self._lock_fd, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise JournalMismatch("request journal is already locked") from exc
        else:
            import fcntl

            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise JournalMismatch("request journal is already locked") from exc

    def _commit(self) -> None:
        raw = canonical(self.data)
        fd, temporary = tempfile.mkstemp(
            prefix=".stage-", suffix=".tmp", dir=self.directory
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            if os.name == "posix":
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def run(self, stage: str, inputs: Any, function: Callable[[], Any]) -> Any:
        if self._closed:
            raise RuntimeError("request journal is closed")
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage identifier must be a nonempty string")
        fingerprint = digest(inputs)
        existing = self.data["stages"].get(stage)
        if existing is not None:
            if existing.get("input_hash") != fingerprint:
                raise JournalMismatch(
                    "stage inputs changed; use a new revision or request identifier"
                )
            if existing.get("output_hash") != digest(existing.get("output")):
                raise JournalMismatch("stage output integrity mismatch")
            return copy.deepcopy(existing["output"])
        # Return the durable JSON representation on the first execution too.
        # This preserves evidence ordering and container types after reopening.
        output = json.loads(canonical(function()))
        record = {
            "input_hash": fingerprint,
            "output_hash": digest(output),
            "output": copy.deepcopy(output),
        }
        self.data["stages"][stage] = record
        try:
            self._commit()
        except BaseException:
            self.data["stages"].pop(stage, None)
            raise
        return copy.deepcopy(output)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(self._lock_fd, 0, os.SEEK_SET)
                msvcrt.locking(self._lock_fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(self._lock_fd)

    def __enter__(self) -> "ExecutionJournal":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

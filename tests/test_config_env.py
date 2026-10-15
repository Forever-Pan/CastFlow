from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.config import CastFlowConfig, load_castflow_env


class ConfigEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project_dir = Path(self.temp_dir.name)
        self.file_patch = patch("utils.config.__file__", str(self.project_dir / "utils" / "config.py"))
        self.file_patch.start()
        self.addCleanup(self.file_patch.stop)

    def write_env(self, text: str) -> None:
        (self.project_dir / ".env").write_text(text, encoding="utf-8")

    def test_legacy_process_values_override_dotenv(self) -> None:
        self.write_env(
            "MAX_REFLECTION_LOOPS=1\n"
            "MAX_TEST_RETRY_LOOPS=2\n"
            "PARALLEL_PLAN_K=3\n"
            "EXCELLENT_MSE_THRESHOLD=4.5\n"
            "EXCELLENT_MAE_THRESHOLD=5.5\n"
        )
        process_values = {
            "MAX_REFLECTION_LOOPS": "6",
            "MAX_TEST_RETRY_LOOPS": "7",
            "PARALLEL_PLAN_K": "8",
            "EXCELLENT_MSE_THRESHOLD": "9.5",
            "EXCELLENT_MAE_THRESHOLD": "10.5",
        }
        with patch.dict(os.environ, process_values, clear=True):
            cfg = CastFlowConfig.from_env()
        self.assertEqual(cfg.max_retry_loops, 6)
        self.assertEqual(cfg.test_max_retry_loops, 7)
        self.assertEqual(cfg.parallel_plan_k, 8)
        self.assertEqual(cfg.excellent_mse_threshold, 9.5)
        self.assertEqual(cfg.excellent_mae_threshold, 10.5)

    def test_prefixed_process_values_still_override_dotenv(self) -> None:
        self.write_env('CASTFLOW_API_TIMEOUT=20\nMODEL="file-model"\n')
        with patch.dict(os.environ, {"CASTFLOW_API_TIMEOUT": "30", "MODEL": "process-model"}, clear=True):
            cfg = CastFlowConfig.from_env()
        self.assertEqual(cfg.api_timeout, 30.0)
        self.assertEqual(cfg.api_model, "process-model")

    def test_legacy_names_are_exact_and_dotenv_values_remain_available(self) -> None:
        self.write_env("MAX_REFLECTION_LOOPS=2\nUNRELATED_SETTING='file-value'\n")
        with patch.dict(os.environ, {"MAX_REFLECTION_LOOPS_EXTRA": "99", "UNRELATED_SETTING": "process-value"}, clear=True):
            values = load_castflow_env()
        self.assertEqual(values["MAX_REFLECTION_LOOPS"], "2")
        self.assertEqual(values["UNRELATED_SETTING"], "file-value")
        self.assertNotIn("MAX_REFLECTION_LOOPS_EXTRA", values)

    def test_explicit_from_env_overrides_keep_precedence(self) -> None:
        with patch.dict(os.environ, {"PARALLEL_PLAN_K": "8"}, clear=True):
            cfg = CastFlowConfig.from_env(parallel_plan_k=2)
        self.assertEqual(cfg.parallel_plan_k, 2)


if __name__ == "__main__":
    unittest.main()

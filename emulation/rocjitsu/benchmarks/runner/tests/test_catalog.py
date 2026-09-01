# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest

from benchmarks.catalog import CASE_BY_ID, TRITON_CASES, validated_triton_case


class CatalogTests(unittest.TestCase):
    def test_triton_views_reference_only_triton_cases(self) -> None:
        self.assertEqual(
            set(TRITON_CASES),
            {case_id for case_id in CASE_BY_ID if case_id.startswith("triton.")},
        )

    def test_triton_override_must_match_catalog(self) -> None:
        case_id = "triton.gemm_bf16_aligned"
        parameters = validated_triton_case(case_id)
        parameters["m"] = 257
        with self.assertRaisesRegex(ValueError, "built-in catalog"):
            validated_triton_case(case_id, parameters)


if __name__ == "__main__":
    unittest.main()

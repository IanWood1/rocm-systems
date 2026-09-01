# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
import tempfile
import textwrap
import time
import unittest
from unittest import mock

from benchmarks.catalog import CASES, CASE_BY_ID, catalog_sha256
from benchmarks.runner.comparison import ComparisonError, compare_runs
from benchmarks.runner.manifest import MANIFEST_SCHEMA, ManifestError, load_suite
from benchmarks.runner.providers import prepare_command
from benchmarks.runner import runtime
from benchmarks.runner.runtime import (
    CASE_RESULT_SCHEMA,
    DEPENDENCY_COHORT_SCHEMA,
    ENVIRONMENT_PROVENANCE_POLICY,
    RUN_SCHEMA,
    RunOptions,
    _RunFileHashes,
    _execute_case,
    _is_executable_file,
    _path_is_within,
    _read_throughput,
    _workload_contract,
    parse_perf_stat,
    parse_time_verbose,
    run_suite,
)


ALL_CASE_IDS = [case.id for case in CASES]
DEPENDENCY_COHORT = {
    "schema": DEPENDENCY_COHORT_SCHEMA,
    "expected": {
        "hipblaslt": "10.1.0a20260812",
        "rocm_sdk": "10.1.0a20260812",
        "torch": "2.14.0a0+rocm10.1.0a20260812",
        "triton": "3.8.0+git675c5987.rocm10.1.0a20260812",
    },
    "actual": {
        "hipblaslt": "10.1.0a20260812",
        "rocm_sdk": "10.1.0a20260812",
        "torch": "2.14.0a0+rocm10.1.0a20260812",
        "triton": "3.8.0+git675c5987.rocm10.1.0a20260812",
    },
    "python": {"actual": "3.12", "expected": "3.12"},
    "hipblaslt": {"expected_package_version": "1.4.1"},
    "passed": True,
    "errors": [],
}


def manifest_text(*, cases: list[str] | None = None, extra: str = "") -> str:
    return textwrap.dedent(
        f"""
        schema = {json.dumps(MANIFEST_SCHEMA)}
        name = "test"
        description = "test suite"
        targets = ["gfx950", "gfx1250"]
        budget_seconds = 1800
        cases = {json.dumps(cases or ALL_CASE_IDS)}

        [measurement]
        warmups = 3
        samples = 21
        {extra}
        """
    )


def write_manifest(root: Path, text: str | None = None) -> Path:
    path = root / "suite.toml"
    path.write_text(text or manifest_text(), encoding="utf-8")
    return path


def workload_payload(
    case_id: str = "hip.launch_noop",
    target: str = "gfx950",
    *,
    warmups: int = 3,
    samples: int = 21,
    values: list[object] | None = None,
) -> dict[str, object]:
    case = CASE_BY_ID[case_id]
    return {
        "schema": "rocjitsu.benchmark.workload.v1",
        "case_id": case_id,
        "target": {
            "expected": target,
            "reported": target,
            "compiled": target,
        },
        "warmups": warmups,
        "samples": samples,
        "synchronized_dispatch_ns": values or list(range(101, 101 + samples)),
        "workload": dict(case.parameters),
        "provenance": {"adapter": case.provider},
    }


class CatalogAndManifestTests(unittest.TestCase):
    def test_catalog_contains_expected_immutable_matrix(self) -> None:
        self.assertEqual(len(CASES), 12)
        self.assertEqual(len(CASE_BY_ID), 12)
        self.assertEqual(len(catalog_sha256()), 64)
        self.assertEqual(catalog_sha256(), catalog_sha256())
        with self.assertRaises(TypeError):
            CASE_BY_ID["hip.launch_noop"].parameters["blocks"] = 2  # type: ignore[index]

    def test_manifest_is_only_a_case_selector_and_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite = load_suite(write_manifest(Path(temporary)))
        self.assertEqual([case.id for case in suite.cases], ALL_CASE_IDS)
        self.assertEqual(suite.measurement.warmups, 3)
        self.assertEqual(suite.measurement.samples, 21)
        self.assertEqual(suite.measurement.metric, "synchronized_dispatch_ns")

    def test_manifest_rejects_legacy_commands_paths_and_case_tables(self) -> None:
        mutations = (
            '\nlauncher = ["rocjitsu"]\n',
            '\ndependency_lock = "../dependencies.lock.toml"\n',
            '\ntarget_configs = {{ gfx950 = "../../config.json" }}\n',
            '\n[[cases]]\nid = "hip.launch_noop"\ncommand = ["true"]\n',
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                text = manifest_text() + mutation
                with self.assertRaises(ManifestError):
                    load_suite(write_manifest(Path(temporary), text))

    def test_manifest_rejects_unknown_and_duplicate_cases(self) -> None:
        for cases in (["unknown.case"], ["hip.launch_noop", "hip.launch_noop"]):
            with self.subTest(cases=cases), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ManifestError):
                    load_suite(write_manifest(Path(temporary), manifest_text(cases=cases)))

    def test_manifest_location_does_not_affect_commands(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            suite_a = load_suite(write_manifest(Path(first)))
            suite_b = load_suite(write_manifest(Path(second)))
            options = RunOptions(
                output=Path(first) / "out",
                build_dir=Path("/tmp/build"),
                python="/usr/bin/python3",
                rocjitsu="/tmp/build/tools/rocjitsu/rocjitsu",
            )
            command_a = prepare_command(
                suite_a,
                suite_a.cases[0],
                "gfx950",
                options,
                Path(first) / "result",
                warmups=3,
                samples=21,
                phase="timing",
            )
            command_b = prepare_command(
                suite_b,
                suite_b.cases[0],
                "gfx950",
                options,
                Path(first) / "result",
                warmups=3,
                samples=21,
                phase="timing",
            )
        self.assertEqual(command_a.argv, command_b.argv)
        self.assertEqual(command_a.cwd, command_b.cwd)

    def test_provider_commands_use_modules_and_internal_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite = load_suite(write_manifest(Path(temporary)))
            options = RunOptions(
                output=Path(temporary) / "out",
                build_dir=Path(temporary) / "build",
                python="/pinned/python",
                rocjitsu="/build/rocjitsu",
            )
            commands = {
                case.provider: prepare_command(
                    suite,
                    case,
                    "gfx950",
                    options,
                    Path(temporary) / case.id,
                    warmups=3,
                    samples=21,
                    phase="timing",
                ).argv
                for case in suite.cases
            }
        self.assertIn("benchmarks.workloads.triton_workloads", commands["triton"])
        self.assertIn("rocjitsu-benchmark-hipblaslt", " ".join(commands["hipblaslt"]))
        self.assertNotIn("--expected-revision", commands["hipblaslt"])
        for command in commands.values():
            self.assertIn("--warmups", command)
            self.assertIn("21", command)
            self.assertNotIn("../", " ".join(command))


class WorkloadContractTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.suite = load_suite(write_manifest(Path(temporary.name)))
        self.case = CASE_BY_ID["hip.launch_noop"]

    def validate(self, payload: dict[str, object]) -> tuple[list[int], str | None]:
        return _workload_contract(
            payload, self.suite, self.case, "gfx950", 3, 21
        )

    def test_accepts_structural_result_without_correctness(self) -> None:
        payload = workload_payload()
        values, error = self.validate(payload)
        self.assertIsNone(error)
        self.assertEqual(values, list(range(101, 122)))
        self.assertNotIn("validation", payload)
        self.assertNotIn("correctness", payload)

    def test_rejects_bad_timing_vectors(self) -> None:
        bad_values = (
            list(range(20)),
            list(range(22)),
            [1] * 20 + [0],
            [1] * 20 + [-1],
            [1] * 20 + [True],
            [1] * 20 + [1.5],
        )
        for values in bad_values:
            with self.subTest(last=values[-1], count=len(values)):
                _timings, error = self.validate(workload_payload(values=values))
                self.assertIsNotNone(error)

    def test_rejects_sampling_target_and_parameter_drift(self) -> None:
        mutations = []
        payload = workload_payload()
        payload["warmups"] = 2
        mutations.append(payload)
        payload = workload_payload()
        payload["target"] = {"expected": "gfx950", "reported": "gfx1250"}
        mutations.append(payload)
        payload = workload_payload()
        payload["workload"] = {"blocks": 2, "threads_per_block": 64}
        mutations.append(payload)
        for payload in mutations:
            with self.subTest(payload=payload):
                _values, error = self.validate(payload)
                self.assertIsNotNone(error)

    def test_hipblaslt_provenance_must_match_the_installed_package(self) -> None:
        case = CASE_BY_ID["tensile.gemm_fp16"]
        payload = workload_payload(case.id)
        payload["provenance"] = {
            "adapter": "hipblaslt_cpp",
            "algorithm_policy": "first_heuristic",
            "hipblaslt_version": 100401,
        }
        values, error = _workload_contract(
            payload, self.suite, case, "gfx950", 3, 21
        )
        self.assertIsNone(error)
        self.assertEqual(len(values), 21)
        payload["provenance"]["algorithm_policy"] = "autotuned"  # type: ignore[index]
        _values, error = _workload_contract(
            payload, self.suite, case, "gfx950", 3, 21
        )
        self.assertIn("algorithm_policy", error or "")
        payload["provenance"]["algorithm_policy"] = "first_heuristic"  # type: ignore[index]
        payload["provenance"]["hipblaslt_version"] = 999999  # type: ignore[index]
        _values, error = _workload_contract(
            payload, self.suite, case, "gfx950", 3, 21
        )
        self.assertIn("version", error or "")


class ParserTests(unittest.TestCase):
    def test_executable_file_requires_execute_permission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "program"
            path.write_text("#!/bin/sh\n", encoding="utf-8")
            path.chmod(0o600)
            self.assertFalse(_is_executable_file(path))
            path.chmod(0o700)
            self.assertTrue(_is_executable_file(path))
        self.assertFalse(_is_executable_file(None))

    def test_profile_launcher_must_be_inside_its_build(self) -> None:
        self.assertTrue(
            _path_is_within(
                Path("/tmp/build/tools/rocjitsu/rocjitsu"), Path("/tmp/build")
            )
        )
        self.assertFalse(
            _path_is_within(
                Path("/tmp/release/tools/rocjitsu/rocjitsu"),
                Path("/tmp/relwithdebinfo"),
            )
        )

    def test_dependency_report_is_normalized_without_its_lock_path(self) -> None:
        payload = {**copy.deepcopy(DEPENDENCY_COHORT), "lock": "/machine/path"}
        cohort, error = runtime._normalize_dependency_cohort(payload)
        self.assertIsNone(error)
        self.assertEqual(cohort, DEPENDENCY_COHORT)
        self.assertNotIn("lock", cohort or {})

    def test_dependency_report_cannot_claim_success_when_versions_differ(self) -> None:
        payload = copy.deepcopy(DEPENDENCY_COHORT)
        payload["actual"]["torch"] = "different"  # type: ignore[index]
        cohort, error = runtime._normalize_dependency_cohort(payload)
        self.assertIsNone(cohort)
        self.assertIn("inconsistent", error or "")

    def test_dependency_report_must_match_the_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite = load_suite(write_manifest(Path(temporary)))
        self.assertIsNone(
            runtime._dependency_cohort_lock_error(DEPENDENCY_COHORT, suite)
        )

        bogus = copy.deepcopy(DEPENDENCY_COHORT)
        bogus["expected"] = {"bogus": "1"}
        bogus["actual"] = {"bogus": "1"}
        cohort, error = runtime._normalize_dependency_cohort(bogus)
        self.assertIsNone(error)
        self.assertIn(
            "dependency lock",
            runtime._dependency_cohort_lock_error(cohort or {}, suite) or "",
        )

        wrong_hipblaslt = copy.deepcopy(DEPENDENCY_COHORT)
        wrong_hipblaslt["hipblaslt"][  # type: ignore[index]
            "expected_package_version"
        ] = "9.9.9"
        self.assertIn(
            "hipBLASLt",
            runtime._dependency_cohort_lock_error(wrong_hipblaslt, suite) or "",
        )

    def test_hipblaslt_override_is_the_exact_target_directory(self) -> None:
        library = Path("/sdk/lib/libhipblaslt.so")
        self.assertEqual(
            runtime._hipblaslt_target_logic_path(library, {}, "gfx950"),
            Path("/sdk/lib/hipblaslt/library/gfx950"),
        )
        self.assertEqual(
            runtime._hipblaslt_target_logic_path(
                library,
                {"HIPBLASLT_TENSILE_LIBPATH": "/alternate/gfx950"},
                "gfx950",
            ),
            Path("/alternate/gfx950"),
        )

    def test_hipblaslt_logic_requires_direct_mapping_and_code_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "gfx950"
            target.mkdir()
            (target / "TensileLibrary_lazy_gfx950.dat.zlib").touch()
            (target / "kernel_gfx950.co").touch()
            self.assertIsNone(
                runtime._hipblaslt_logic_directory_error(target, "gfx950")
            )
            self.assertIsNotNone(
                runtime._hipblaslt_logic_directory_error(root, "gfx950")
            )
            (target / "kernel_gfx950.co").unlink()
            self.assertIsNotNone(
                runtime._hipblaslt_logic_directory_error(target, "gfx950")
            )

    def test_preflight_probes_hipblaslt_for_every_selected_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = load_suite(
                write_manifest(root, manifest_text(cases=["tensile.gemm_fp16"]))
            )
            options = RunOptions(
                output=root / "out",
                build_dir=root / "build",
                python="python",
                collectors=(),
            )
            verifier = mock.Mock(
                returncode=0,
                stdout=json.dumps({**DEPENDENCY_COHORT, "lock": "/machine/path"}),
                stderr="",
            )
            with mock.patch.object(
                runtime, "_resolve_program", return_value=Path("/bin/true")
            ), mock.patch.object(
                runtime, "_cmake_build_type", return_value="Release"
            ), mock.patch.object(
                runtime, "_is_executable_file", return_value=True
            ), mock.patch.object(
                runtime, "_validate_timing_config", return_value=(True, "config")
            ), mock.patch.object(
                runtime.subprocess, "run", return_value=verifier
            ), mock.patch.object(
                runtime,
                "_probe_hipblaslt",
                side_effect=((True, "gfx950"), (False, "gfx1250")),
            ) as probe:
                report = runtime.preflight(suite, options)

        self.assertFalse(report["passed"])
        self.assertEqual(report["dependency_cohort"], DEPENDENCY_COHORT)
        self.assertEqual(
            [call.args[2] for call in probe.call_args_list],
            ["gfx950", "gfx1250"],
        )
        checks = {item["name"]: item["passed"] for item in report["checks"]}
        self.assertTrue(checks["hipblaslt-package:gfx950"])
        self.assertFalse(checks["hipblaslt-package:gfx1250"])

    def test_parses_gnu_time(self) -> None:
        value = parse_time_verbose(
            """
            User time (seconds): 1.25
            System time (seconds): 0.50
            Elapsed (wall clock) time (h:mm:ss or m:ss): 0:02.00
            Maximum resident set size (kbytes): 42
            Major (requiring I/O) page faults: 1
            Minor (reclaiming a frame) page faults: 2
            """
        )
        self.assertEqual(value["elapsed_seconds"], 2.0)
        self.assertEqual(value["max_rss_kbytes"], 42)

    def test_parses_perf_stat(self) -> None:
        value = parse_perf_stat(
            """10,,cycles:u,1,100.00,
20,,instructions:u,1,100.00,
30,msec,task-clock,1,100.00,
1,,context-switches,1,100.00,
2,,cpu-migrations,1,100.00,
3,,minor-faults,1,100.00,
4,,major-faults,1,100.00,
"""
        )
        self.assertEqual(value["cycles"], 10)
        self.assertEqual(value["task_clock_milliseconds"], 30.0)

    def test_rejects_incomplete_throughput_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "throughput.log"
            path.write_text(
                '{"schema":"rocjitsu.throughput.v2","record":"summary"}\n',
                encoding="utf-8",
            )
            summary, error = _read_throughput(path)
        self.assertIsNone(summary)
        self.assertIn("dispatches", error or "")

    def test_accepts_complete_throughput_summary(self) -> None:
        records = (
            '{"schema":"rocjitsu.throughput.v2","record":"dispatch"}\n'
            '{"schema":"rocjitsu.throughput.v2","record":"summary",'
            '"dispatches":1,"wave_instructions":0,'
            '"dispatch_seconds_sum":0.0}\n'
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "throughput.log"
            path.write_text(records, encoding="utf-8")
            summary, error = _read_throughput(path)
        self.assertIsNone(error)
        self.assertEqual(summary["dispatches"], 1)  # type: ignore[index]


class ExecutionTests(unittest.TestCase):
    def test_execute_case_runs_one_process_and_summarizes_inner_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = load_suite(write_manifest(root))
            build = root / "build"
            binary = build / "benchmarks/rocjitsu-benchmark-native-gfx950"
            binary.parent.mkdir(parents=True)
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import argparse,json\n"
                "p=argparse.ArgumentParser();p.add_argument('--case');"
                "p.add_argument('--expected-target');p.add_argument('--warmups',type=int);"
                "p.add_argument('--samples',type=int);p.add_argument('--output');a=p.parse_args()\n"
                "payload={'schema':'rocjitsu.benchmark.workload.v1','case_id':a.case,"
                "'target':{'expected':a.expected_target,'reported':a.expected_target,'compiled':a.expected_target},"
                "'warmups':a.warmups,'samples':a.samples,'synchronized_dispatch_ns':list(range(100,100+a.samples)),"
                "'workload':{'blocks':1,'threads_per_block':64},'provenance':{'adapter':'fake'}}\n"
                "open(a.output,'w').write(json.dumps(payload))\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            launcher = root / "rocjitsu"
            launcher.write_text(
                "#!/bin/sh\nshift 2\n[ \"$1\" = -- ] && shift\nexec \"$@\"\n",
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")
            output = root / "output"
            options = RunOptions(
                output=output,
                build_dir=build,
                rocjitsu=str(launcher),
                configs={"gfx950": str(config)},
                collectors=(),
            )
            result = _execute_case(
                suite,
                CASE_BY_ID["hip.launch_noop"],
                "gfx950",
                options,
                output / "cases/hip.launch_noop/gfx950/timing",
                _RunFileHashes(),
                phase="timing",
                warmups=3,
                samples=21,
                collectors=(),
            )
            self.assertEqual(result["status"], "passed", result["failure"])
            self.assertEqual(result["measurement"]["statistics"]["count"], 21)
            self.assertNotIn("correctness", result)
            self.assertNotIn("validation", result["case"])
            self.assertTrue(
                (output / "cases/hip.launch_noop/gfx950/timing/case-result.json").is_file()
            )

    def test_full_orchestration_has_one_primary_result_per_matrix_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = load_suite(write_manifest(root))
            options = RunOptions(
                output=root / "out",
                build_dir=root / "build",
                rocjitsu=str(root / "build/rocjitsu"),
                collectors=(),
            )

            def fake_execute(
                suite, case, target, options, result_dir, hashes, **kwargs
            ):
                values = list(range(1, kwargs["samples"] + 1))
                return {
                    "schema": CASE_RESULT_SCHEMA,
                    "case": {"id": case.id},
                    "target": target,
                    "phase": kwargs["phase"],
                    "status": "passed",
                    "failure": None,
                    "measurement": {
                        "statistics": runtime.summarize(values),
                    },
                    "throughput_summary": (
                        {"dispatches": 1, "wave_instructions": 1}
                        if kwargs.get("throughput")
                        else None
                    ),
                }

            with mock.patch.object(
                runtime,
                "preflight",
                return_value={
                    "checks": [],
                    "dependency_cohort": DEPENDENCY_COHORT,
                },
            ), mock.patch.object(
                runtime,
                "_collect_provenance",
                return_value={},
            ), mock.patch.object(
                runtime,
                "_execute_case",
                side_effect=fake_execute,
            ) as execute:
                result = run_suite(suite, options)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(result["results"]), 24)
            self.assertEqual(len(result["throughput"]), 24)
            self.assertEqual(execute.call_count, 48)
            self.assertTrue(all(item["statistics"]["count"] == 21 for item in result["summaries"]))

    def test_budget_is_checked_before_the_throughput_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = dataclasses.replace(
                load_suite(write_manifest(root)), budget_seconds=0.05
            )
            options = RunOptions(
                output=root / "out",
                build_dir=root / "build",
                rocjitsu=str(root / "build/rocjitsu"),
                targets=("gfx950",),
                case_ids=("hip.launch_noop",),
                collectors=(),
            )

            def slow_execute(
                suite, case, target, options, result_dir, hashes, **kwargs
            ):
                time.sleep(0.1)
                return {
                    "schema": CASE_RESULT_SCHEMA,
                    "case": {"id": case.id},
                    "target": target,
                    "phase": kwargs["phase"],
                    "status": "passed",
                    "failure": None,
                    "measurement": {"statistics": runtime.summarize([1])},
                }

            with mock.patch.object(
                runtime,
                "preflight",
                return_value={
                    "checks": [],
                    "dependency_cohort": DEPENDENCY_COHORT,
                },
            ), mock.patch.object(
                runtime, "_collect_provenance", return_value={}
            ), mock.patch.object(
                runtime, "_execute_case", side_effect=slow_execute
            ) as execute:
                result = run_suite(suite, options)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(result["throughput"], [])
        self.assertIn("exceeded", result["failures"][0])


def comparison_result(
    values: list[int], *, case_id: str = "hip.launch_noop", target: str = "gfx950"
) -> dict[str, object]:
    return {
        "schema": CASE_RESULT_SCHEMA,
        "status": "passed",
        "phase": "timing",
        "case": {
            "id": case_id,
            "provider": "native",
            "parameters": {"blocks": 1, "threads_per_block": 64},
            "sources": ["src/native_workloads.hip"],
            "source_sha256": "1" * 64,
            "expected_dispatch_count": 1,
        },
        "target": target,
        "measurement": {
            "metric": "synchronized_dispatch_ns",
            "warmups": 3,
            "samples": len(values),
            "values": values,
        },
        "provenance": {
            "payload_executable": {"sha256": "2" * 64},
            "config_sha256": "3" * 64,
            "effective_environment": {
                "policy": ENVIRONMENT_PROVENANCE_POLICY,
                "values": {"PATH": "/bin"},
            },
        },
    }


def comparison_run(values: list[int]) -> dict[str, object]:
    result = comparison_result(values)
    throughput = copy.deepcopy(result)
    throughput["measurement"] = {
        "metric": "synchronized_dispatch_ns",
        "warmups": 0,
        "samples": 1,
        "values": [1],
    }
    throughput["phase"] = "throughput"
    throughput["throughput_summary"] = {
        "dispatches": 1,
        "dispatch_seconds_sum": 0.1,
        "wave_instructions": 64,
    }
    return {
        "schema": RUN_SCHEMA,
        "status": "passed",
        "mode": "run",
        "suite": {
            "name": "nightly",
            "manifest_sha256": "4" * 64,
            "catalog_sha256": "5" * 64,
            "targets": ["gfx950"],
            "cases": ["hip.launch_noop"],
        },
        "execution": {
            "measurement": {
                "metric": "synchronized_dispatch_ns",
                "warmups": 3,
                "samples": len(values),
            },
            "warmups_override": None,
            "samples_override": None,
            "collectors": [],
        },
        "provenance": {
            "runner": {"version": "1", "source_sha256": "6" * 64, "python": "3.12"},
            "catalog_sha256": "5" * 64,
            "host": {
                "hostname": "host",
                "system": "Linux",
                "release": "1",
                "machine": "x86_64",
                "cpu_model": "cpu",
                "cpu_count": 1,
            },
            "source": {},
            "dependencies": {
                "lock_sha256": "7" * 64,
                "verifier_sha256": "8" * 64,
                "cohort": copy.deepcopy(DEPENDENCY_COHORT),
            },
            "build_type": "Release",
            "explicit_environment": {},
            "configs": {"gfx950": {"path": "/config", "sha256": "3" * 64}},
            "rocjitsu": {"sha256": "9" * 64},
        },
        "results": [result],
        "throughput": [throughput],
    }


class ComparisonTests(unittest.TestCase):
    def test_compares_every_internal_timing(self) -> None:
        baseline = comparison_run([100, 110, 120])
        candidate = comparison_run([90, 100, 110])
        result = compare_runs(baseline, candidate)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(len(result["comparisons"][0]["pairs"]), 3)

    def test_rejects_sampling_and_catalog_mismatch(self) -> None:
        baseline = comparison_run([100, 110, 120])
        candidate = comparison_run([90, 100, 110])
        candidate["provenance"]["catalog_sha256"] = "9" * 64  # type: ignore[index]
        with self.assertRaisesRegex(ComparisonError, "catalog_sha256"):
            compare_runs(baseline, candidate)

        candidate = comparison_run([90, 100])
        with self.assertRaises(ComparisonError):
            compare_runs(baseline, candidate)

    def test_rejects_dependency_verifier_or_cohort_mismatch(self) -> None:
        baseline = comparison_run([100, 110, 120])
        candidate = comparison_run([90, 100, 110])
        candidate["provenance"]["dependencies"][  # type: ignore[index]
            "verifier_sha256"
        ] = "a" * 64
        with self.assertRaisesRegex(ComparisonError, "dependencies"):
            compare_runs(baseline, candidate)

        candidate = comparison_run([90, 100, 110])
        candidate["provenance"]["dependencies"]["cohort"]["actual"][  # type: ignore[index]
            "torch"
        ] = "different"
        with self.assertRaisesRegex(ComparisonError, "dependency cohort"):
            compare_runs(baseline, candidate)

        malformed = comparison_run([100, 110, 120])
        malformed["provenance"]["dependencies"]["cohort"][  # type: ignore[index]
            "expected"
        ] = {"bogus": "1"}
        malformed["provenance"]["dependencies"]["cohort"][  # type: ignore[index]
            "actual"
        ] = {"bogus": "1"}
        with self.assertRaisesRegex(ComparisonError, "dependency cohort"):
            compare_runs(malformed, comparison_run([90, 100, 110]))

    def test_rejects_result_sampling_that_contradicts_the_run(self) -> None:
        baseline = comparison_run([100, 110, 120])
        candidate = comparison_run([90, 100, 110])
        baseline["execution"]["measurement"]["samples"] = 21  # type: ignore[index]
        with self.assertRaisesRegex(ComparisonError, "contradicts execution"):
            compare_runs(baseline, candidate)

    def test_process_wall_diagnostics_do_not_affect_comparison(self) -> None:
        baseline = comparison_run([100, 110, 120])
        candidate = comparison_run([90, 100, 110])
        baseline["results"][0]["diagnostics"] = {"process_wall_time_ns": 1}  # type: ignore[index]
        candidate["results"][0]["diagnostics"] = {  # type: ignore[index]
            "process_wall_time_ns": 999
        }
        result = compare_runs(baseline, candidate)
        self.assertEqual(result["status"], "passed")

    def test_rejects_draft_v1_artifact_without_result_matrix(self) -> None:
        baseline = comparison_run([100, 110, 120])
        candidate = comparison_run([90, 100, 110])
        del baseline["results"]
        with self.assertRaisesRegex(ComparisonError, "missing required results"):
            compare_runs(baseline, candidate)

    def test_rejects_incomplete_throughput_summary(self) -> None:
        baseline = comparison_run([100, 110, 120])
        candidate = comparison_run([90, 100, 110])
        del baseline["throughput"][0]["throughput_summary"][  # type: ignore[index]
            "dispatch_seconds_sum"
        ]
        with self.assertRaisesRegex(ComparisonError, "summary is incomplete"):
            compare_runs(baseline, candidate)


if __name__ == "__main__":
    unittest.main()

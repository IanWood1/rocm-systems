"""Command-line interface for the Rocjitsu benchmark runner."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .comparison import ComparisonError, compare_files
from .manifest import ManifestError, load_suite
from .runtime import RunOptions, RunnerError, preflight, run_suite, select_suite


def _csv_values(values: Sequence[str] | None) -> tuple[str, ...]:
    result: list[str] = []
    for value in values or ():
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(result)


def _assignments(values: Sequence[str] | None, option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values or ():
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise RunnerError(f"{option} expects KEY=VALUE, got {value!r}")
        if key in result:
            raise RunnerError(f"{option} repeats key {key!r}")
        result[key] = item
    return result


def _collectors(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if value == "none" or value == "":
        return ()
    result = _csv_values([value])
    unknown = set(result) - {"time", "perf"}
    if unknown:
        raise RunnerError(f"unsupported collectors: {sorted(unknown)}")
    return result


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest", required=True, type=Path, help="suite TOML manifest"
    )
    parser.add_argument(
        "--targets",
        action="append",
        metavar="TARGET[,TARGET...]",
        help="select targets (default: every suite target)",
    )
    parser.add_argument(
        "--cases",
        action="append",
        metavar="CASE[,CASE...]",
        help="select cases (default: every suite case)",
    )
    parser.add_argument(
        "--build-dir", type=Path, help="Rocjitsu build containing benchmark workloads"
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python used for Triton workloads (default: current interpreter)",
    )
    parser.add_argument(
        "--rocjitsu", help="Rocjitsu executable (default: derived from --build-dir)"
    )
    parser.add_argument(
        "--config",
        action="append",
        metavar="TARGET=PATH",
        help="override a target config path",
    )
    parser.add_argument(
        "--env",
        action="append",
        metavar="KEY=VALUE",
        help="set a workload environment value",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rocjitsu-benchmark",
        description="Run and compare reproducible Rocjitsu benchmark suites.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight", help="validate a suite and its local dependencies"
    )
    _add_selection(preflight_parser)
    preflight_parser.add_argument("--collectors", help="time, perf, time,perf, or none")
    preflight_parser.add_argument(
        "--list", action="store_true", help="list the matrix only"
    )
    preflight_parser.add_argument(
        "--json", action="store_true", help="emit machine-readable output"
    )

    run_parser = subparsers.add_parser("run", help="execute a benchmark suite")
    _add_selection(run_parser)
    run_parser.add_argument(
        "--output", required=True, type=Path, help="new artifact directory"
    )
    run_parser.add_argument("--collectors", help="time, perf, time,perf, or none")
    run_parser.add_argument(
        "--warmups", type=int, help="override warmups per case and target"
    )
    run_parser.add_argument("--samples", type=int, help="override recorded samples")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="preflight and show commands without creating output",
    )

    compare_parser = subparsers.add_parser(
        "compare", help="strictly compare paired samples from two run artifacts"
    )
    compare_parser.add_argument("--baseline", required=True, type=Path)
    compare_parser.add_argument("--candidate", required=True, type=Path)
    compare_parser.add_argument("--output", type=Path, help="new comparison JSON path")

    profile_parser = subparsers.add_parser(
        "profile", help="collect perf record data for one workload"
    )
    profile_parser.add_argument("--manifest", required=True, type=Path)
    profile_parser.add_argument("--target", required=True)
    profile_parser.add_argument("--case", required=True)
    profile_parser.add_argument("--output", required=True, type=Path)
    profile_parser.add_argument(
        "--build-dir",
        required=True,
        type=Path,
        help="RelWithDebInfo build containing the profiled rocjitsu binary",
    )
    profile_parser.add_argument("--python", default=sys.executable)
    profile_parser.add_argument("--rocjitsu")
    profile_parser.add_argument("--config", action="append", metavar="TARGET=PATH")
    profile_parser.add_argument("--env", action="append", metavar="KEY=VALUE")
    profile_parser.add_argument(
        "--dry-run", action="store_true", help="preflight without creating output"
    )
    return parser


def _options(args: argparse.Namespace, *, profile: bool = False) -> RunOptions:
    warmups = getattr(args, "warmups", None)
    samples = getattr(args, "samples", None)
    if warmups is not None and warmups < 0:
        raise RunnerError("--warmups must be non-negative")
    if samples is not None and samples <= 0:
        raise RunnerError("--samples must be positive")
    output = getattr(args, "output", Path("benchmark-preflight-output"))
    return RunOptions(
        output=output,
        build_dir=args.build_dir,
        python=args.python,
        rocjitsu=args.rocjitsu,
        configs=_assignments(args.config, "--config"),
        environment=_assignments(args.env, "--env"),
        targets=(args.target,) if profile else _csv_values(args.targets),
        case_ids=(args.case,) if profile else _csv_values(args.cases),
        collectors=None if profile else _collectors(getattr(args, "collectors", None)),
        warmups=warmups,
        samples=samples,
        fail_fast=False if profile else getattr(args, "fail_fast", False),
    )


def _print_preflight(report: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"suite: {report['suite']}")
    print(f"targets: {', '.join(report['targets'])}")  # type: ignore[arg-type]
    print(f"cases: {', '.join(report['cases'])}")  # type: ignore[arg-type]
    for check in report["checks"]:  # type: ignore[union-attr]
        marker = "ok" if check["passed"] else "FAIL"
        print(f"[{marker}] {check['name']}: {check['detail']}")
    for command in report["commands"]:  # type: ignore[union-attr]
        print(
            f"command {command['case']}/{command['target']}/{command['phase']}: "
            + json.dumps(command["argv"])
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.subcommand == "compare":
            result = compare_files(
                args.baseline,
                args.candidate,
                output=args.output,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        suite = load_suite(args.manifest)
        if args.subcommand == "profile":
            options = _options(args, profile=True)
            if args.dry_run:
                report = preflight(suite, options, profile=True)
                _print_preflight(report, as_json=False)
                return 0 if report["passed"] else 1
            result = run_suite(suite, options, profile=True)
            print(f"profile {result['status']}: {options.output / 'run.json'}")
            return 0 if result["status"] == "passed" else 1

        options = _options(args)
        if args.subcommand == "preflight" and args.list:
            targets, selected_cases = select_suite(suite, options)
            cases = tuple(case.id for case in selected_cases)
            payload = {"suite": suite.name, "targets": targets, "cases": cases}
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"suite: {suite.name}")
                for case in cases:
                    print(f"{case}: {', '.join(targets)}")
            return 0

        if args.subcommand == "preflight" or args.dry_run:
            report = preflight(suite, options)
            _print_preflight(report, as_json=getattr(args, "json", False))
            return 0 if report["passed"] else 1

        result = run_suite(suite, options)
        print(f"run {result['status']}: {options.output / 'run.json'}")
        return 0 if result["status"] == "passed" else 1
    except (ManifestError, RunnerError, ComparisonError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

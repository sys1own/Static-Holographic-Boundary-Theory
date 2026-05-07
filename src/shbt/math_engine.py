from __future__ import annotations

"""Deterministic arithmetic helpers for cross-architecture reproducibility.

This module replaces hardware-dependent floating-point workflows with exact
fractional arithmetic and canonical result serialization. The goal is simple:
the same benchmark calculation must produce the same reduced numerator,
denominator, and SHA-256 digest on every supported runtime.

The engine uses ``fractions.Fraction`` throughout, exposing exact scalar,
vector, and matrix operations together with a benchmark verifier that checks the
canonical outputs stored under ``data/benchmarks/cross_arch/``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, TypeAlias

from shbt.paths import ProjectPaths


ExactOperand: TypeAlias = Fraction | Decimal | int | float | str
DEFAULT_CROSS_ARCH_BENCHMARK_DIR = ProjectPaths.DATA / "benchmarks" / "cross_arch"
SUPPORTED_BENCHMARK_OPERATIONS = frozenset({"add", "subtract", "multiply", "divide", "dot", "matvec", "matmul"})


def exact_number(value: ExactOperand) -> Fraction:
    """Normalize a scalar into a reduced exact rational.

    Floats are converted through their decimal string rendering so the engine
    does not inherit binary ``float64`` roundoff in downstream operations.
    """

    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid exact operands.")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(Decimal(str(value)))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Exact operand strings must be non-empty.")
        return Fraction(stripped)
    raise TypeError(f"Unsupported exact operand type: {type(value)!r}.")


def exact_vector(values: Sequence[ExactOperand]) -> tuple[Fraction, ...]:
    resolved = tuple(exact_number(value) for value in values)
    if not resolved:
        raise ValueError("Exact vectors must contain at least one entry.")
    return resolved


def exact_matrix(rows: Sequence[Sequence[ExactOperand]]) -> tuple[tuple[Fraction, ...], ...]:
    resolved_rows = tuple(exact_vector(row) for row in rows)
    if not resolved_rows:
        raise ValueError("Exact matrices must contain at least one row.")
    column_count = len(resolved_rows[0])
    if any(len(row) != column_count for row in resolved_rows):
        raise ValueError("Exact matrices must be rectangular.")
    return resolved_rows


def canonical_fraction_string(value: Fraction) -> str:
    resolved = exact_number(value)
    return f"{resolved.numerator}/{resolved.denominator}"


def canonical_result_tree(value: Any) -> Any:
    if isinstance(value, Fraction):
        return canonical_fraction_string(value)
    if isinstance(value, tuple):
        return [canonical_result_tree(item) for item in value]
    if isinstance(value, list):
        return [canonical_result_tree(item) for item in value]
    raise TypeError(f"Unsupported canonical result type: {type(value)!r}.")


def canonical_result_digest(value: Any) -> str:
    payload = json.dumps(canonical_result_tree(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CrossArchitectureCaseReport:
    name: str
    operation: str
    expected_result: Any
    observed_result: Any
    expected_digest: str | None
    observed_digest: str
    passed: bool


@dataclass(frozen=True)
class CrossArchitectureBenchmarkReport:
    benchmark_path: Path
    suite_name: str
    engine_name: str
    case_reports: tuple[CrossArchitectureCaseReport, ...]
    suite_digest: str
    expected_suite_digest: str | None

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.case_reports) and (
            self.expected_suite_digest is None or self.expected_suite_digest == self.suite_digest
        )


class MathEngine:
    """Exact rational arithmetic surface for deterministic SHBT calculations."""

    engine_name = "fractions.Fraction"

    @staticmethod
    def exact(value: ExactOperand) -> Fraction:
        return exact_number(value)

    def add(self, *operands: ExactOperand) -> Fraction:
        if not operands:
            raise ValueError("add requires at least one operand.")
        total = Fraction(0, 1)
        for operand in operands:
            total += exact_number(operand)
        return total

    def subtract(self, lhs: ExactOperand, rhs: ExactOperand) -> Fraction:
        return exact_number(lhs) - exact_number(rhs)

    def multiply(self, *operands: ExactOperand) -> Fraction:
        if not operands:
            raise ValueError("multiply requires at least one operand.")
        product = Fraction(1, 1)
        for operand in operands:
            product *= exact_number(operand)
        return product

    def divide(self, lhs: ExactOperand, rhs: ExactOperand) -> Fraction:
        denominator = exact_number(rhs)
        if denominator == 0:
            raise ZeroDivisionError("Exact division requires a non-zero denominator.")
        return exact_number(lhs) / denominator

    def dot(self, lhs: Sequence[ExactOperand], rhs: Sequence[ExactOperand]) -> Fraction:
        left = exact_vector(lhs)
        right = exact_vector(rhs)
        if len(left) != len(right):
            raise ValueError("Exact dot products require vectors of equal length.")
        return sum((left_value * right_value for left_value, right_value in zip(left, right, strict=True)), Fraction(0, 1))

    def matvec(self, matrix: Sequence[Sequence[ExactOperand]], vector: Sequence[ExactOperand]) -> tuple[Fraction, ...]:
        resolved_matrix = exact_matrix(matrix)
        resolved_vector = exact_vector(vector)
        if len(resolved_matrix[0]) != len(resolved_vector):
            raise ValueError("Exact matvec requires the matrix column count to match the vector length.")
        return tuple(self.dot(row, resolved_vector) for row in resolved_matrix)

    def matmul(
        self,
        lhs: Sequence[Sequence[ExactOperand]],
        rhs: Sequence[Sequence[ExactOperand]],
    ) -> tuple[tuple[Fraction, ...], ...]:
        left = exact_matrix(lhs)
        right = exact_matrix(rhs)
        if len(left[0]) != len(right):
            raise ValueError("Exact matmul requires the left column count to equal the right row count.")
        right_columns = tuple(tuple(row[column_index] for row in right) for column_index in range(len(right[0])))
        return tuple(tuple(self.dot(row, column) for column in right_columns) for row in left)

    def canonicalize(self, value: Any) -> Any:
        return canonical_result_tree(value)

    def digest(self, value: Any) -> str:
        return canonical_result_digest(value)

    def evaluate_benchmark_case(self, payload: Mapping[str, Any]) -> Any:
        operation = str(payload.get("operation", "")).strip().lower()
        if operation not in SUPPORTED_BENCHMARK_OPERATIONS:
            raise ValueError(f"Unsupported benchmark operation {operation!r}.")

        if operation == "add":
            operands = payload.get("operands")
            if not isinstance(operands, Sequence) or isinstance(operands, (str, bytes)):
                raise TypeError("Benchmark add cases require a sequence of operands.")
            return self.add(*operands)
        if operation == "subtract":
            return self.subtract(payload["lhs"], payload["rhs"])
        if operation == "multiply":
            operands = payload.get("operands")
            if not isinstance(operands, Sequence) or isinstance(operands, (str, bytes)):
                raise TypeError("Benchmark multiply cases require a sequence of operands.")
            return self.multiply(*operands)
        if operation == "divide":
            return self.divide(payload["lhs"], payload["rhs"])
        if operation == "dot":
            return self.dot(payload["lhs"], payload["rhs"])
        if operation == "matvec":
            return self.matvec(payload["matrix"], payload["vector"])
        return self.matmul(payload["lhs"], payload["rhs"])


def _normalize_expected_result(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_expected_result(item) for item in value]
    return canonical_fraction_string(exact_number(value))


def _suite_digest(case_reports: Sequence[CrossArchitectureCaseReport]) -> str:
    payload = [
        {"name": report.name, "digest": report.observed_digest}
        for report in case_reports
    ]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(serialized.encode("utf-8")).hexdigest()


def verify_cross_arch_benchmark(
    benchmark_path: Path | str,
    *,
    engine: MathEngine | None = None,
) -> CrossArchitectureBenchmarkReport:
    resolved_engine = MathEngine() if engine is None else engine
    resolved_path = Path(benchmark_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Cross-architecture benchmark {resolved_path} must contain a top-level object.")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise TypeError(f"Cross-architecture benchmark {resolved_path} must contain a non-empty 'cases' array.")

    case_reports: list[CrossArchitectureCaseReport] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise TypeError(f"Cross-architecture benchmark case in {resolved_path} must be an object.")
        observed = resolved_engine.evaluate_benchmark_case(raw_case)
        observed_canonical = resolved_engine.canonicalize(observed)
        observed_digest = resolved_engine.digest(observed)
        expected_canonical = _normalize_expected_result(raw_case.get("expected"))
        expected_digest = raw_case.get("digest")
        passed = observed_canonical == expected_canonical and (
            expected_digest is None or str(expected_digest) == observed_digest
        )
        case_reports.append(
            CrossArchitectureCaseReport(
                name=str(raw_case.get("name", "")),
                operation=str(raw_case.get("operation", "")),
                expected_result=expected_canonical,
                observed_result=observed_canonical,
                expected_digest=None if expected_digest is None else str(expected_digest),
                observed_digest=observed_digest,
                passed=passed,
            )
        )

    return CrossArchitectureBenchmarkReport(
        benchmark_path=resolved_path,
        suite_name=str(payload.get("suite", resolved_path.stem)),
        engine_name=str(payload.get("engine", resolved_engine.engine_name)),
        case_reports=tuple(case_reports),
        suite_digest=_suite_digest(case_reports),
        expected_suite_digest=None if payload.get("suite_digest") is None else str(payload.get("suite_digest")),
    )


def verify_cross_arch_benchmarks(
    benchmark_dir: Path | str = DEFAULT_CROSS_ARCH_BENCHMARK_DIR,
    *,
    engine: MathEngine | None = None,
) -> tuple[CrossArchitectureBenchmarkReport, ...]:
    resolved_dir = Path(benchmark_dir)
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Cross-architecture benchmark directory {resolved_dir} does not exist.")
    resolved_engine = MathEngine() if engine is None else engine
    benchmark_paths = tuple(sorted(path for path in resolved_dir.iterdir() if path.suffix == ".json"))
    if not benchmark_paths:
        raise FileNotFoundError(f"Cross-architecture benchmark directory {resolved_dir} does not contain any JSON files.")
    return tuple(verify_cross_arch_benchmark(path, engine=resolved_engine) for path in benchmark_paths)


__all__ = [
    "CrossArchitectureBenchmarkReport",
    "CrossArchitectureCaseReport",
    "DEFAULT_CROSS_ARCH_BENCHMARK_DIR",
    "MathEngine",
    "SUPPORTED_BENCHMARK_OPERATIONS",
    "canonical_fraction_string",
    "canonical_result_digest",
    "canonical_result_tree",
    "exact_matrix",
    "exact_number",
    "exact_vector",
    "verify_cross_arch_benchmark",
    "verify_cross_arch_benchmarks",
]

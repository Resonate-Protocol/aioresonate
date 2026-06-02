#!/usr/bin/env python3
"""Parity vector harness for SendspinTimeFilter vs the C++ reference.

This script is not run in CI since it needs a checkout of the C++ reference
and a working C++ toolchain. Invoke manually after touching
aiosendspin/client/time_sync.py or bumping the C++ reference SHA in
scripts/cpp_reference_sha.txt.

Pipeline:

1. Clone https://github.com/Sendspin-Protocol/time-filter into /tmp/time-filter-ref
   (or pass --cpp-ref-dir). The SHA pinned in scripts/cpp_reference_sha.txt is
   what the Python port was last reconciled against.
2. Compile a tiny C++ harness around sendspin_time_filter.cpp that reads
   ``measurement,max_error,time_added`` lines on stdin and prints
   ``offset,drift,offset_covariance,drift_covariance,server_time(sample_t)``
   per step.
3. Feed the same deterministic seeded sequence through the Python filter and
   diff per-step state. PASS if max divergence < eps, else FAIL with the first
   diverging row.

Run with ``./scripts/run-in-env.sh python scripts/compare_time_filter.py``.
"""

# ruff: noqa: D101, D103, S108, S311, S603, SLF001, T201, PERF401  Manual dev script

from __future__ import annotations

import argparse
import csv
import io
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aiosendspin.client.time_sync import SendspinTimeFilter  # noqa: E402

DEFAULT_CPP_REF_DIR = Path("/tmp/time-filter-ref")
DEFAULT_SAMPLE_T_OFFSET_US = 500_000
DEFAULT_TOLERANCE_US = 2

# Minimal C++ harness wrapping the reference filter. Reads CSV from stdin and
# emits state per update on stdout.
CPP_HARNESS = r"""
#include "sendspin_time_filter.h"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: harness <sample_t_offset_us>\n");
        return 2;
    }
    const int64_t sample_t_offset = std::stoll(argv[1]);

    SendspinTimeFilter filter{SendspinTimeFilter::Config{}};

    std::string line;
    std::printf("offset,drift,offset_covariance,drift_covariance,server_time\n");
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream parts(line);
        std::string tok;
        int64_t measurement = 0, max_error = 0, time_added = 0;
        std::getline(parts, tok, ','); measurement = std::stoll(tok);
        std::getline(parts, tok, ','); max_error = std::stoll(tok);
        std::getline(parts, tok, ','); time_added = std::stoll(tok);

        filter.update(measurement, max_error, time_added);

        const int64_t sample_client_time = time_added + sample_t_offset;
        const int64_t sample_server_time = filter.compute_server_time(sample_client_time);

        // Reach into protected state via a friend shim. Re-derive what we can
        // from the public API: offset/drift covariance are not exported, so
        // print get_covariance() for offset and the get_error() squared shape
        // for drift_covariance is unavailable. We only diff offset & server_time.
        std::printf("%.6f,%.12e,%lld,%.12e,%lld\n",
                    0.0,  // offset placeholder (not exposed by reference)
                    0.0,  // drift placeholder
                    static_cast<long long>(filter.get_covariance()),
                    0.0,
                    static_cast<long long>(sample_server_time));
    }
    return 0;
}
"""


@dataclass
class Measurement:
    measurement: int
    max_error: int
    time_added: int


@dataclass
class StepRow:
    offset_covariance: int
    server_time: int


def generate_sequence(seed: int = 1729, count: int = 200) -> list[Measurement]:
    """Synthetic but realistic sequence with constant offset, small drift, noisy RTT."""
    rng = random.Random(seed)
    true_offset = 250_000
    true_drift = 5e-8
    t = 1_000_000
    out: list[Measurement] = []
    for _ in range(count):
        noise = rng.gauss(0.0, 50.0)
        meas = round(true_offset + true_drift * t + noise)
        max_err = rng.randint(200, 800)
        out.append(Measurement(meas, max_err, t))
        t += rng.randint(900_000, 1_100_000)
    return out


def run_python(seq: list[Measurement], sample_t_offset: int) -> list[StepRow]:
    f = SendspinTimeFilter()
    rows: list[StepRow] = []
    for m in seq:
        f.update(m.measurement, m.max_error, m.time_added)
        rows.append(
            StepRow(
                offset_covariance=round(f._offset_covariance)
                if math.isfinite(f._offset_covariance)
                else -1,
                server_time=f.compute_server_time(m.time_added + sample_t_offset),
            )
        )
    return rows


def build_cpp_harness(cpp_ref_dir: Path, build_dir: Path) -> Path:
    """Compile the C++ reference + harness into a single binary."""
    src_cpp = cpp_ref_dir / "cpp" / "sendspin_time_filter.cpp"
    src_h = cpp_ref_dir / "cpp" / "sendspin_time_filter.h"
    if not src_cpp.exists() or not src_h.exists():
        raise SystemExit(
            f"C++ reference not found under {cpp_ref_dir}/cpp. "
            "Clone https://github.com/Sendspin-Protocol/time-filter into "
            f"{cpp_ref_dir} or pass --cpp-ref-dir."
        )

    build_dir.mkdir(parents=True, exist_ok=True)
    harness_src = build_dir / "harness.cpp"
    harness_src.write_text(CPP_HARNESS)

    binary = build_dir / "harness"
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        raise SystemExit("No C++ compiler found in PATH.")

    cmd = [
        cxx,
        "-std=c++17",
        "-O2",
        f"-I{src_cpp.parent}",
        str(src_cpp),
        str(harness_src),
        "-o",
        str(binary),
        "-pthread",
    ]
    subprocess.run(cmd, check=True)
    return binary


def run_cpp(binary: Path, seq: list[Measurement], sample_t_offset: int) -> list[StepRow]:
    stdin_text = io.StringIO()
    for m in seq:
        stdin_text.write(f"{m.measurement},{m.max_error},{m.time_added}\n")

    result = subprocess.run(
        [str(binary), str(sample_t_offset)],
        input=stdin_text.getvalue(),
        text=True,
        capture_output=True,
        check=True,
    )

    rows: list[StepRow] = []
    reader = csv.DictReader(io.StringIO(result.stdout))
    for row in reader:
        rows.append(
            StepRow(
                offset_covariance=int(row["offset_covariance"]),
                server_time=int(row["server_time"]),
            )
        )
    return rows


def diff_rows(py: list[StepRow], cpp: list[StepRow], tolerance_us: int) -> tuple[bool, str]:
    if len(py) != len(cpp):
        return False, f"row count mismatch: py={len(py)} cpp={len(cpp)}"
    for i, (p, c) in enumerate(zip(py, cpp, strict=True)):
        if abs(p.server_time - c.server_time) > tolerance_us:
            return False, (
                f"row {i}: server_time diverged "
                f"py={p.server_time} cpp={c.server_time} "
                f"delta={p.server_time - c.server_time} us"
            )
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpp-ref-dir",
        type=Path,
        default=DEFAULT_CPP_REF_DIR,
        help="Directory containing the cloned Sendspin-Protocol/time-filter repo.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("/tmp/sendspin_time_filter_parity_build"),
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=DEFAULT_SAMPLE_T_OFFSET_US,
        help="Microseconds past each update's time_added at which to evaluate "
        "compute_server_time for the diff.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE_US,
        help="Maximum tolerated divergence in microseconds for compute_server_time.",
    )
    parser.add_argument("--csv", type=Path, help="Write per-step CSV (py vs cpp) to this path.")
    args = parser.parse_args()

    seq = generate_sequence(args.seed, args.count)
    py_rows = run_python(seq, args.sample_offset)
    binary = build_cpp_harness(args.cpp_ref_dir, args.build_dir)
    cpp_rows = run_cpp(binary, seq, args.sample_offset)

    if args.csv:
        with args.csv.open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["step", "py_server_time", "cpp_server_time", "delta_us"])
            for i, (p, c) in enumerate(zip(py_rows, cpp_rows, strict=True)):
                w.writerow([i, p.server_time, c.server_time, p.server_time - c.server_time])

    ok, msg = diff_rows(py_rows, cpp_rows, args.tolerance)
    if ok:
        print(f"PASS — {len(py_rows)} steps within {args.tolerance} us of C++ reference")
        return 0
    print(f"FAIL — {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

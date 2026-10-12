"""Make a compact, outward-rounded inventory of accepted certificate records.

This performs consistency/provenance checks on saved records; it does not
recompute the Arb proofs.  Every displayed endpoint is checked as an exact
rational against the saved exact endpoint, so shortening a decimal cannot
silently destroy the enclosure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shlex


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent


def directed_decimal(value: Fraction, digits: int, *, upward: bool) -> str:
    with localcontext() as context:
        context.prec = digits
        context.rounding = ROUND_CEILING if upward else ROUND_FLOOR
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
    result = format(decimal, "e")
    exact_display = Fraction(result)
    assert exact_display >= value if upward else exact_display <= value
    return result


def compact_interval(record: dict, digits: int) -> dict:
    lower = Fraction(record["lower_rational"])
    upper = Fraction(record["upper_rational"])
    if not lower <= upper:
        raise ValueError("Reversed interval in a saved record.")
    return {
        "lower": directed_decimal(lower, digits, upward=False),
        "upper": directed_decimal(upper, digits, upward=True),
        "decimal_semantics": "exact outward endpoints, independently rounded from exact rational bounds",
    }


def payload_in_current_checkout(record: dict) -> Path:
    recorded = Path(record["payload_path"])
    marker = "/capacity_experiment/"
    if marker in str(recorded):
        return EXPERIMENT / str(recorded).split(marker, 1)[1]
    return recorded


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(digits: int = 17) -> dict:
    rows = []
    base_hash = sha256(HERE / "interval_core.py")
    splitter_hash = sha256(HERE / "interval_split.py")
    records = sorted(HERE.glob("*arb*.json"))
    for path in records:
        record = json.loads(path.read_text())
        n = record["n"]
        assert record["evaluator_source_sha256"] == base_hash, path
        if "exact_zero_splitter_source_sha256" in record:
            assert record["exact_zero_splitter_source_sha256"] == splitter_hash, path
        payload = payload_in_current_checkout(record)
        assert sha256(payload) == record["payload_file_sha256"], path
        assert record["all_spin_sectors_included"] is True, path
        assert [sector["q"] for sector in record["sectors"]] == list(range(n // 2 + 1)), path
        trace = record["weighted_total_trace"]
        assert Fraction(trace["lower_rational"]) <= 1 <= Fraction(trace["upper_rational"]), path
        ic = record["coherent_information_bits"]
        rate = record["achievable_rate_lower_bound_bits_per_channel_use"]
        assert Fraction(rate["lower_rational"]) <= Fraction(ic["lower_rational"]) / n, path
        assert Fraction(rate["upper_rational"]) >= Fraction(ic["upper_rational"]) / n, path
        positive = Fraction(ic["lower_rational"]) > 0
        assert positive == record["certified_positive_coherent_information"], path
        assert record["sign"] == ("positive" if positive else "negative" if Fraction(ic["upper_rational"]) < 0 else "unresolved"), path
        row = {
            "record": str(path.relative_to(EXPERIMENT)),
            "record_sha256": sha256(path),
            "payload": str(payload.relative_to(EXPERIMENT)),
            "payload_sha256": record["payload_file_sha256"],
            "n": n,
            "input_cutoff": record["input_cutoff"],
            "p_total_exact": record["p_total_exact"],
            "coherent_information_bits": compact_interval(ic, digits),
            "achievable_rate_bits_per_channel_use": compact_interval(rate, digits),
            "positive": positive,
            "precision_bits": record["precision_bits"],
            "spin_sector_count": len(record["sectors"]),
            "build_seconds": record["build_seconds"],
            "evaluation_seconds": record["evaluation_seconds"],
        }
        archive_key = "psi" if payload.name == "A45_continued.npz" else None
        if payload.suffix == ".npz" and archive_key is None:
            raise ValueError(f"Record inventory needs the explicit NPZ key for {payload}.")
        evaluator = "interval_split.py" if "exact_zero_splitter_source_sha256" in record else "interval_core.py"
        arguments = [
            "python", f"capacity_experiment/validation/{evaluator}",
            "--payload", "capacity_experiment/" + str(payload.relative_to(EXPERIMENT)),
            "--n", str(n), "--p-total", record["p_total_exact"],
            "--cutoff", str(record["input_cutoff"]),
            "--precision-bits", str(record["precision_bits"]),
            "--output", f"capacity_experiment/validation/replayed/{path.name}",
        ]
        if archive_key:
            arguments += ["--key", archive_key]
        row["replay_argv_from_experiment_parent"] = arguments
        row["replay_command_from_experiment_parent"] = shlex.join(arguments)
        if "exact_zero_splitter_source_sha256" in record:
            row["largest_joint_component_dimension"] = max(
                max(sector["joint_spectrum"]["component_dimensions"])
                for sector in record["sectors"])
            row["largest_output_component_dimension"] = max(
                max(sector["output_spectrum"]["component_dimensions"])
                for sector in record["sectors"])
        rows.append(row)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_checked": len(rows),
        "all_payload_and_source_hashes_match": True,
        "all_interval_trace_sector_and_rate_consistency_checks_passed": True,
        "summary_source_sha256": sha256(Path(__file__)),
        "interval_evaluator_source_sha256": base_hash,
        "exact_zero_splitter_source_sha256": splitter_hash,
        "evidence_status": "inventory_of_rigorous_ball_enclosures_with_FLINT_Arb_trusted; inventory_does_not_recompute_proofs",
        "display_significant_digits": digits,
        "units": {"coherent_information_bits": "bits per n-channel block", "achievable_rate_bits_per_channel_use": "numerically Ic/n, operational quantum rate in qubits per channel use"},
        "failed_unaccepted_attempts_record": "validation/failed_attempts.json",
        "records": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=17)
    parser.add_argument("--output", type=Path, default=HERE / "certificate_inventory.json")
    args = parser.parse_args()
    if args.digits < 2:
        parser.error("At least two significant digits are required.")
    result = inventory(args.digits)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

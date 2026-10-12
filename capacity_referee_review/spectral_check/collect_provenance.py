"""Record the runtime and binary identities immediately after the new runs."""

import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from capacity_referee_review.spectral_check import residual_verifier as rv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    vendor = Path(rv.flint.__file__).resolve().parents[1]
    native_dir = vendor / "python_flint.libs"
    versions = {"FLINT": str(rv.flint.__FLINT_VERSION__)}
    mpfr = ctypes.CDLL(str(next(native_dir.glob("libmpfr-*.so.*"))))
    mpfr.mpfr_get_version.restype = ctypes.c_char_p
    versions["MPFR"] = mpfr.mpfr_get_version().decode("ascii")
    gmp = ctypes.CDLL(str(next(native_dir.glob("libgmp-*.so.*"))))
    versions["GMP"] = ctypes.c_char_p.in_dll(gmp, "__gmp_version").value.decode("ascii")
    binaries = []
    for path in sorted(vendor.rglob("*")):
        if path.is_file() and (".so" in path.name or path.name in ("METADATA", "WHEEL", "RECORD")):
            binaries.append({"path_relative_to_vendor": str(path.relative_to(vendor)),
                             "size_bytes": path.stat().st_size, "sha256": digest(path)})
    result_files = sorted(Path(__file__).parent.glob("*_independent_residual.json"))
    results = []
    for path in result_files:
        data = json.loads(path.read_text())
        sources_match_now = all(digest(ROOT/name) == value for name, value in data["source_hashes"].items())
        payload_match_now = digest(data["payload_path"]) == data["payload_file_sha256"]
        if not sources_match_now or not payload_match_now:
            raise RuntimeError("Completed-result identity mismatch.")
        results.append({"path_relative_to_root": str(path.relative_to(ROOT)),
                        "sha256": digest(path), "source_hashes_match_now": sources_match_now,
                        "payload_hash_matches_now": payload_match_now})
    report = {
        "status": "recorded",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post-run runtime/binary fingerprint collection; not a pre-execution capture",
        "python": sys.version,
        "python_executable": sys.executable,
        "python_executable_sha256": digest(sys.executable),
        "platform": platform.platform(),
        "numpy_version": rv.np.__version__,
        "python_flint_version": rv.flint.__version__,
        "native_library_versions": versions,
        "flint_module_path": rv.flint.__file__,
        "binary_and_installed_metadata_hashes": binaries,
        "accepted_result_records": results,
        "collector_source_sha256": digest(__file__),
    }
    destination = Path(__file__).with_name("runtime_provenance.json")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite provenance record.")
    destination.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"native_versions": versions, "accepted_records": len(results),
                      "binaries_fingerprinted": len(binaries)}))


if __name__ == "__main__":
    main()

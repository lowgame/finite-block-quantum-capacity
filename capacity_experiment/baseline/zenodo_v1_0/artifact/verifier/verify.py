"""Pure-integer verifier for flattened rank-two endpoint certificates."""

from __future__ import annotations

from fractions import Fraction
from functools import cmp_to_key, lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
from typing import Any, Sequence


SCHEMA = "physics-proofs/flattened-rank-two-endpoint/v1"
N = 45
SQRT_BITS = 224
FROBENIUS_BITS = 192
LOG_TERMS = 96
LOG_GRID_DIGITS = 70
ACCUMULATION_DIGITS = 50
SCALE = 10**ACCUMULATION_DIGITS
EXPECTED_PAYLOAD_BYTES = 2 * (N + 1) * 16
EXPECTED_PAYLOAD_HEX = 2 * EXPECTED_PAYLOAD_BYTES
MAX_CERTIFICATE_BYTES = 64_000_000
MAX_INTEGER_BITS = 20_000
MAX_DECIMAL_DIGITS = 6_021
MAX_DYADIC_EXPONENT_ABS = 4_096
INTEGER_RE = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
HEX_RE = re.compile(r"[0-9a-f]+\Z")

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(MAX_DECIMAL_DIGITS + 100)

Gaussian = tuple[Fraction, Fraction]
ZERO: Gaussian = (Fraction(0), Fraction(0))


class VerificationError(ValueError):
    """A deterministic certificate rejection."""


def _fail(reason: str) -> None:
    raise VerificationError(reason)


def _keys(value: Any, expected: Sequence[str], reason: str) -> None:
    if not isinstance(value, dict):
        _fail(reason)
    if list(value.keys()) != list(expected):
        _fail(reason)


def _integer_text(value: Any) -> int:
    if not isinstance(value, str) or INTEGER_RE.fullmatch(value) is None:
        _fail("integer-canonical")
    digits = len(value) - (1 if value.startswith("-") else 0)
    if digits > MAX_DECIMAL_DIGITS:
        _fail("integer-too-large")
    parsed = int(value, 10)
    if parsed.bit_length() > MAX_INTEGER_BITS:
        _fail("integer-too-large")
    return parsed


def _q(value: Any) -> Fraction:
    if not isinstance(value, list) or len(value) != 2:
        _fail("rational-shape")
    numerator = _integer_text(value[0])
    denominator = _integer_text(value[1])
    if denominator <= 0:
        _fail("rational-canonical")
    if math.gcd(abs(numerator), denominator) != 1:
        _fail("rational-canonical")
    return Fraction(numerator, denominator)


def _dyad(value: Any) -> Fraction:
    if not isinstance(value, list) or len(value) != 2:
        _fail("dyadic-shape")
    mantissa = _integer_text(value[0])
    exponent = _integer_text(value[1])
    if abs(exponent) > MAX_DYADIC_EXPONENT_ABS:
        _fail("dyadic-exponent")
    if mantissa == 0:
        if exponent != 0:
            _fail("dyadic-canonical")
        return Fraction(0)
    if mantissa % 2 == 0:
        _fail("dyadic-canonical")
    if exponent >= 0:
        return Fraction(mantissa << exponent)
    return Fraction(mantissa, 1 << (-exponent))


def _gd(value: Any) -> Gaussian:
    if not isinstance(value, list) or len(value) != 4:
        _fail("gaussian-dyadic-shape")
    real = _dyad([value[0], value[1]])
    imag = _dyad([value[2], value[3]])
    return real, imag


def _qeq(left: Fraction, right: Fraction) -> bool:
    return (
        left.numerator == right.numerator
        and left.denominator == right.denominator
    )


def _qle(left: Fraction, right: Fraction) -> bool:
    return (
        left.numerator * right.denominator
        <= right.numerator * left.denominator
    )


def _qlt(left: Fraction, right: Fraction) -> bool:
    return (
        left.numerator * right.denominator
        < right.numerator * left.denominator
    )


def _qmin(left: Fraction, right: Fraction) -> Fraction:
    return left if _qle(left, right) else right


def _qmax(left: Fraction, right: Fraction) -> Fraction:
    return right if _qle(left, right) else left


def _qcmp(left: Fraction, right: Fraction) -> int:
    first = left.numerator * right.denominator
    second = right.numerator * left.denominator
    if first < second:
        return -1
    if first > second:
        return 1
    return 0


def _disc_cmp(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
) -> int:
    comparison = _qcmp(left[0], right[0])
    if comparison != 0:
        return comparison
    return _qcmp(left[1], right[1])


def _expect_equal(
    actual: Fraction,
    expected: Fraction,
    reason: str,
) -> None:
    if not _qeq(actual, expected):
        _fail(reason)


def _floor_scaled(value: Fraction) -> int:
    return value.numerator * SCALE // value.denominator


def _ceil_scaled(value: Fraction) -> int:
    return -((-value.numerator * SCALE) // value.denominator)


def _sqrt_bounds(
    value: Fraction,
    *,
    bits: int,
) -> tuple[Fraction, Fraction]:
    if value.numerator < 0:
        _fail("negative-square-root")
    if value.numerator == 0:
        return Fraction(0), Fraction(0)
    scale = 1 << bits
    target_numerator = value.numerator * scale * scale
    target_denominator = value.denominator
    low = math.isqrt(target_numerator // target_denominator)
    while (
        (low + 1) * (low + 1) * target_denominator
        <= target_numerator
    ):
        low += 1
    while low * low * target_denominator > target_numerator:
        low -= 1
    lower = Fraction(low, scale)
    if low * low * target_denominator == target_numerator:
        upper = lower
    else:
        upper = Fraction(low + 1, scale)

    if not _qle(lower * lower, value):
        _fail("internal-square-root-lower")
    if not _qle(value, upper * upper):
        _fail("internal-square-root-upper")
    width = upper - lower
    if not (
        _qeq(width, Fraction(0))
        or _qeq(width, Fraction(1, scale))
    ):
        _fail("internal-square-root-grid")
    return lower, upper


def _gaussian_add(left: Gaussian, right: Gaussian) -> Gaussian:
    return left[0] + right[0], left[1] + right[1]


def _gaussian_multiply(left: Gaussian, right: Gaussian) -> Gaussian:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _gaussian_conjugate(value: Gaussian) -> Gaussian:
    return value[0], -value[1]


def _gaussian_scale(value: Gaussian, factor: Fraction) -> Gaussian:
    return value[0] * factor, value[1] * factor


def _gaussian_equal(left: Gaussian, right: Gaussian) -> bool:
    return _qeq(left[0], right[0]) and _qeq(left[1], right[1])


def _abs_upper(value: Gaussian) -> Fraction:
    square = value[0] * value[0] + value[1] * value[1]
    return _sqrt_bounds(square, bits=SQRT_BITS)[1]


def _matrix_multiply(
    left: list[list[Gaussian]],
    right: list[list[Gaussian]],
) -> list[list[Gaussian]]:
    rows = len(left)
    middle = len(right)
    if rows == 0 or middle == 0:
        _fail("matrix-product-shape")
    columns = len(right[0])
    if any(len(row) != middle for row in left):
        _fail("matrix-product-shape")
    if any(len(row) != columns for row in right):
        _fail("matrix-product-shape")
    output = [
        [ZERO for _ in range(columns)]
        for _ in range(rows)
    ]
    for row in range(rows):
        for column in range(columns):
            value = ZERO
            for index in range(middle):
                value = _gaussian_add(
                    value,
                    _gaussian_multiply(
                        left[row][index],
                        right[index][column],
                    ),
                )
            output[row][column] = value
    return output


def _adjoint(matrix: list[list[Gaussian]]) -> list[list[Gaussian]]:
    if not matrix or not matrix[0]:
        _fail("matrix-adjoint-shape")
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        _fail("matrix-adjoint-shape")
    return [
        [
            _gaussian_conjugate(matrix[column][row])
            for column in range(len(matrix))
        ]
        for row in range(columns)
    ]


def _frobenius_upper(
    matrix: Sequence[Sequence[Gaussian]],
    *,
    bits: int,
) -> Fraction:
    square = sum(
        value[0] * value[0] + value[1] * value[1]
        for row in matrix
        for value in row
    )
    return _sqrt_bounds(square, bits=bits)[1]


def _parse_matrix(
    document: Any,
    dimension: int,
) -> list[list[Gaussian]]:
    _keys(document, ["shape", "entries"], "matrix-schema")
    if document["shape"] != [str(dimension), str(dimension)]:
        _fail("matrix-shape")
    entries = document["entries"]
    if not isinstance(entries, list) or len(entries) != dimension * dimension:
        _fail("array-size")
    matrix: list[list[Gaussian]] = []
    offset = 0
    for _row in range(dimension):
        row: list[Gaussian] = []
        for _column in range(dimension):
            row.append(_gd(entries[offset]))
            offset += 1
        matrix.append(row)
    return matrix


def _check_hermitian(
    matrix: Sequence[Sequence[Gaussian]],
    reason: str,
) -> None:
    dimension = len(matrix)
    for row in range(dimension):
        for column in range(dimension):
            expected = _gaussian_conjugate(matrix[column][row])
            if not _gaussian_equal(matrix[row][column], expected):
                _fail(reason)
        if matrix[row][row][1].numerator != 0:
            _fail(reason)


def _reject_json_number(_value: str) -> Any:
    _fail("json-number")
    return None


def _canonical_json_bytes(document: Any) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("ascii")
        + b"\n"
    )


def _validate_component_shape(component: Any) -> None:
    _keys(
        component,
        [
            "interval",
            "multiplicity",
            "clipped",
            "endpoint_entropy",
            "critical_cap_used",
            "entropy_range",
        ],
        "component-schema",
    )
    if not isinstance(component["interval"], list) or len(component["interval"]) != 2:
        _fail("component-schema")
    _q(component["interval"][0])
    _q(component["interval"][1])
    multiplicity = _integer_text(component["multiplicity"])
    if multiplicity <= 0:
        _fail("component-merge")
    if not isinstance(component["clipped"], list) or len(component["clipped"]) != 2:
        _fail("component-schema")
    _q(component["clipped"][0])
    _q(component["clipped"][1])
    endpoint = component["endpoint_entropy"]
    if not isinstance(endpoint, list) or len(endpoint) != 2:
        _fail("component-schema")
    for pair in endpoint:
        if not isinstance(pair, list) or len(pair) != 2:
            _fail("component-schema")
        _q(pair[0])
        _q(pair[1])
    if type(component["critical_cap_used"]) is not bool:
        _fail("component-schema")
    entropy_range = component["entropy_range"]
    if not isinstance(entropy_range, list) or len(entropy_range) != 2:
        _fail("component-schema")
    _q(entropy_range[0])
    _q(entropy_range[1])


def _validate_spectral_shape(
    spectral: Any,
    dimension: int,
) -> None:
    _keys(
        spectral,
        [
            "dimension",
            "center_dyadic",
            "candidate_basis",
            "bounds",
            "rows",
            "components",
            "scaled_entropy",
        ],
        "spectral-schema",
    )
    if _integer_text(spectral["dimension"]) != dimension:
        _fail("spectral-dimension")

    for matrix_name in ("center_dyadic", "candidate_basis"):
        matrix = spectral[matrix_name]
        _keys(matrix, ["shape", "entries"], "matrix-schema")
        if matrix["shape"] != [str(dimension), str(dimension)]:
            _fail("matrix-shape")
        entries = matrix["entries"]
        if not isinstance(entries, list) or len(entries) != dimension * dimension:
            _fail("array-size")
        for entry in entries:
            _gd(entry)

    bounds = spectral["bounds"]
    _keys(
        bounds,
        [
            "algebraic_frobenius",
            "center_conversion_frobenius",
            "total_matrix_frobenius",
            "gram_defect_frobenius",
            "sqrt_one_minus_lower",
            "sqrt_one_plus_upper",
            "polar_distance",
            "basis_norm_upper",
            "center_norm_upper",
            "transform_error",
        ],
        "bounds-schema",
    )
    for value in bounds.values():
        _q(value)

    rows = spectral["rows"]
    if not isinstance(rows, list) or len(rows) != dimension:
        _fail("array-size")
    for row in rows:
        _keys(row, ["offdiagonal_sum", "disc"], "row-schema")
        _q(row["offdiagonal_sum"])
        if not isinstance(row["disc"], list) or len(row["disc"]) != 2:
            _fail("row-schema")
        _q(row["disc"][0])
        _q(row["disc"][1])

    components = spectral["components"]
    if (
        not isinstance(components, list)
        or len(components) == 0
        or len(components) > dimension
    ):
        _fail("array-size")
    total_multiplicity = 0
    for component in components:
        _validate_component_shape(component)
        total_multiplicity += _integer_text(component["multiplicity"])
    if total_multiplicity != dimension:
        _fail("component-merge")

    scaled_entropy = spectral["scaled_entropy"]
    if not isinstance(scaled_entropy, list) or len(scaled_entropy) != 2:
        _fail("spectral-schema")
    _integer_text(scaled_entropy[0])
    _integer_text(scaled_entropy[1])


def _validate_document_shape(document: Any) -> None:
    _keys(
        document,
        [
            "schema",
            "n",
            "payload_hex",
            "payload_sha256",
            "p",
            "constants",
            "blocks",
            "global_trace",
            "scaled",
            "classification",
        ],
        "top-level-schema",
    )
    if document["schema"] != SCHEMA:
        _fail("schema")
    if _integer_text(document["n"]) != N:
        _fail("n")

    payload_hex = document["payload_hex"]
    if (
        not isinstance(payload_hex, str)
        or len(payload_hex) != EXPECTED_PAYLOAD_HEX
        or HEX_RE.fullmatch(payload_hex) is None
    ):
        _fail("payload-hex")
    payload_sha256 = document["payload_sha256"]
    if (
        not isinstance(payload_sha256, str)
        or len(payload_sha256) != 64
        or HEX_RE.fullmatch(payload_sha256) is None
    ):
        _fail("payload-sha256")

    _q(document["p"])
    constants = document["constants"]
    _keys(
        constants,
        [
            "sqrt_bits",
            "log_terms",
            "log_grid_digits",
            "accumulation_digits",
        ],
        "constants-schema",
    )
    if constants != {
        "sqrt_bits": "224",
        "log_terms": "96",
        "log_grid_digits": "70",
        "accumulation_digits": "50",
    }:
        _fail("constants")

    blocks = document["blocks"]
    if not isinstance(blocks, list) or len(blocks) != 23:
        _fail("block-topology")
    for index, block in enumerate(blocks):
        first_row = 23 + index
        degree = 2 * first_row - N
        output_dimension = degree + 1
        joint_dimension = 2 * output_dimension
        _keys(
            block,
            [
                "first_row",
                "degree",
                "specht_dimension",
                "trace",
                "output",
                "joint",
            ],
            "block-schema",
        )
        if (
            _integer_text(block["first_row"]) != first_row
            or _integer_text(block["degree"]) != degree
        ):
            _fail("block-topology")
        _integer_text(block["specht_dimension"])
        trace = block["trace"]
        _keys(trace, ["center", "radius"], "trace-schema")
        _q(trace["center"])
        _q(trace["radius"])
        _validate_spectral_shape(block["output"], output_dimension)
        _validate_spectral_shape(block["joint"], joint_dimension)

    global_trace = document["global_trace"]
    _keys(global_trace, ["center", "radius"], "trace-schema")
    _q(global_trace["center"])
    _q(global_trace["radius"])

    scaled = document["scaled"]
    _keys(
        scaled,
        [
            "output_lower",
            "output_upper",
            "joint_lower",
            "joint_upper",
            "coherent_lower",
            "coherent_upper",
        ],
        "scaled-schema",
    )
    for value in scaled.values():
        _integer_text(value)
    if document["classification"] not in {
        "POS",
        "NEG",
        "UNRESOLVED",
    }:
        _fail("classification")


def _load_certificate(path: Path) -> dict[str, Any]:
    try:
        information = path.lstat()
    except OSError:
        _fail("certificate-io")
    if not stat.S_ISREG(information.st_mode):
        _fail("certificate-not-regular")
    if information.st_size > MAX_CERTIFICATE_BYTES:
        _fail("certificate-too-large")
    try:
        raw = path.read_bytes()
    except OSError:
        _fail("certificate-io")
    if len(raw) != information.st_size:
        _fail("certificate-size-race")
    try:
        document = json.loads(
            raw,
            parse_int=_reject_json_number,
            parse_float=_reject_json_number,
            parse_constant=_reject_json_number,
        )
    except VerificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _fail("json-invalid")
    _validate_document_shape(document)
    if _canonical_json_bytes(document) != raw:
        _fail("json-not-canonical")
    return document


def _decode_binary64(word: int) -> Fraction:
    sign = -1 if (word >> 63) != 0 else 1
    exponent_field = (word >> 52) & 0x7FF
    fraction_field = word & ((1 << 52) - 1)
    if exponent_field == 0x7FF:
        _fail("payload-nonfinite")
    if exponent_field == 0:
        if fraction_field == 0:
            return Fraction(0)
        mantissa = fraction_field
        exponent = -1074
    else:
        mantissa = (1 << 52) | fraction_field
        exponent = exponent_field - 1023 - 52
    mantissa *= sign
    if exponent >= 0:
        return Fraction(mantissa << exponent)
    return Fraction(mantissa, 1 << (-exponent))


def _decode_payload(payload: bytes) -> list[list[Gaussian]]:
    if len(payload) != EXPECTED_PAYLOAD_BYTES:
        _fail("payload-size")
    values: list[Gaussian] = []
    for offset in range(0, len(payload), 16):
        real_word = int.from_bytes(
            payload[offset : offset + 8],
            "little",
            signed=False,
        )
        imag_word = int.from_bytes(
            payload[offset + 8 : offset + 16],
            "little",
            signed=False,
        )
        values.append(
            (_decode_binary64(real_word), _decode_binary64(imag_word))
        )
    return [
        values[: N + 1],
        values[N + 1 :],
    ]


def _exact_state(
    state: Sequence[Sequence[Gaussian]],
) -> tuple[list[list[Gaussian]], Fraction]:
    if len(state) != 2 or any(len(row) != N + 1 for row in state):
        _fail("state-shape")
    exact = [list(state[0]), list(state[1])]
    norm = sum(
        value[0] * value[0] + value[1] * value[1]
        for row in exact
        for value in row
    )
    if norm.numerator <= 0:
        _fail("candidate-zero")

    gram00 = sum(
        value[0] * value[0] + value[1] * value[1]
        for value in exact[0]
    )
    gram11 = sum(
        value[0] * value[0] + value[1] * value[1]
        for value in exact[1]
    )
    gram01 = ZERO
    for left, right in zip(exact[0], exact[1]):
        gram01 = _gaussian_add(
            gram01,
            _gaussian_multiply(
                left,
                _gaussian_conjugate(right),
            ),
        )
    determinant = (
        gram00 * gram11
        - gram01[0] * gram01[0]
        - gram01[1] * gram01[1]
    )
    if determinant.numerator <= 0:
        _fail("candidate-not-rank-two")
    return exact, norm


@lru_cache(maxsize=None)
def _irrep_rational_coefficient(
    n: int,
    first_row: int,
    final_row: int,
    final_column: int,
    k: int,
    l: int,
    p: Fraction,
) -> Fraction:
    degree = 2 * first_row - n
    determinant_power = n - first_row
    raw_row = degree - final_column
    raw_column = degree - final_row
    if k - l != raw_row - raw_column:
        return Fraction(0)

    a0 = 1 - 2 * p
    a1 = 2 * p
    d0 = 2 * p
    d1 = 1 - 2 * p
    off = 1 - 4 * p
    determinant_scalar = 2 * p * (1 - 2 * p)
    total = Fraction(0)
    lower_s = max(0, raw_row + raw_column - degree)
    upper_s = min(raw_row, raw_column)

    for summation in range(lower_s, upper_s + 1):
        b_power = raw_column - summation
        c_power = raw_row - summation
        d_power = degree - raw_row - raw_column + summation
        prefix = (
            math.comb(raw_column, summation)
            * math.comb(
                degree - raw_column,
                raw_row - summation,
            )
            * off ** (b_power + c_power)
        )
        for a_xy in range(summation + 1):
            a_term = (
                math.comb(summation, a_xy)
                * a0 ** (summation - a_xy)
                * a1**a_xy
            )
            for d_xy in range(d_power + 1):
                d_term = (
                    math.comb(d_power, d_xy)
                    * d0 ** (d_power - d_xy)
                    * d1**d_xy
                )
                determinant_xy = k - (
                    a_xy + d_xy + c_power
                )
                if not 0 <= determinant_xy <= 2 * determinant_power:
                    continue
                if (
                    l
                    != a_xy
                    + d_xy
                    + determinant_xy
                    + b_power
                ):
                    continue
                total += (
                    prefix
                    * a_term
                    * d_term
                    * math.comb(
                        2 * determinant_power,
                        determinant_xy,
                    )
                    * determinant_scalar**determinant_power
                )
    return total


@lru_cache(maxsize=None)
def _entry_kernel(
    n: int,
    first_row: int,
    final_row: int,
    final_column: int,
    p: Fraction,
) -> tuple[tuple[int, int, Fraction, Fraction], ...]:
    degree = 2 * first_row - n
    raw_row = degree - final_column
    raw_column = degree - final_row
    factorial_ratio = Fraction(
        math.factorial(raw_row)
        * math.factorial(degree - raw_row),
        math.factorial(raw_column)
        * math.factorial(degree - raw_column),
    )
    terms: list[tuple[int, int, Fraction, Fraction]] = []
    for k in range(n + 1):
        l = k - (raw_row - raw_column)
        if not 0 <= l <= n:
            continue
        rational = _irrep_rational_coefficient(
            n,
            first_row,
            final_row,
            final_column,
            k,
            l,
            p,
        )
        if rational.numerator == 0:
            continue
        square = factorial_ratio / (
            math.comb(n, k) * math.comb(n, l)
        )
        sqrt_lower, sqrt_upper = _sqrt_bounds(
            square,
            bits=SQRT_BITS,
        )
        if rational.numerator >= 0:
            lower = rational * sqrt_lower
            upper = rational * sqrt_upper
        else:
            lower = rational * sqrt_upper
            upper = rational * sqrt_lower
        terms.append((k, l, lower, upper))
    return tuple(terms)


def _multiply_gaussian_interval(
    value: Gaussian,
    lower: Fraction,
    upper: Fraction,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    real_first = value[0] * lower
    real_second = value[0] * upper
    imag_first = value[1] * lower
    imag_second = value[1] * upper
    return (
        _qmin(real_first, real_second),
        _qmax(real_first, real_second),
        _qmin(imag_first, imag_second),
        _qmax(imag_first, imag_second),
    )


def _entry_box(
    exact_state: Sequence[Sequence[Gaussian]],
    norm: Fraction,
    *,
    n: int,
    first_row: int,
    reference_row: int,
    reference_column: int,
    final_row: int,
    final_column: int,
    p: Fraction,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    re_lower = Fraction(0)
    re_upper = Fraction(0)
    im_lower = Fraction(0)
    im_upper = Fraction(0)
    for k, l, factor_lower, factor_upper in _entry_kernel(
        n,
        first_row,
        final_row,
        final_column,
        p,
    ):
        density = _gaussian_multiply(
            exact_state[reference_row][k],
            _gaussian_conjugate(
                exact_state[reference_column][l]
            ),
        )
        if _gaussian_equal(density, ZERO):
            continue
        density = _gaussian_scale(density, Fraction(1, 1) / norm)
        term = _multiply_gaussian_interval(
            density,
            factor_lower,
            factor_upper,
        )
        re_lower += term[0]
        re_upper += term[1]
        im_lower += term[2]
        im_upper += term[3]
    return re_lower, re_upper, im_lower, im_upper


def _box_center_radius(
    box: tuple[Fraction, Fraction, Fraction, Fraction],
) -> tuple[Gaussian, Fraction]:
    center = (
        (box[0] + box[1]) / 2,
        (box[2] + box[3]) / 2,
    )
    re_radius = (box[1] - box[0]) / 2
    im_radius = (box[3] - box[2]) / 2
    square = (
        re_radius * re_radius
        + im_radius * im_radius
    )
    radius = _sqrt_bounds(square, bits=SQRT_BITS)[1]
    if not _qle(square, radius * radius):
        _fail("algebraic-entry-radius")
    return center, radius


def _symmetrize(
    center: list[list[Gaussian]],
    radius: list[list[Fraction]],
) -> tuple[list[list[Gaussian]], list[list[Fraction]]]:
    dimension = len(center)
    output_center = [
        [ZERO for _ in range(dimension)]
        for _ in range(dimension)
    ]
    output_radius = [
        [Fraction(0) for _ in range(dimension)]
        for _ in range(dimension)
    ]
    for row in range(dimension):
        for column in range(row, dimension):
            combined = _gaussian_scale(
                _gaussian_add(
                    center[row][column],
                    _gaussian_conjugate(
                        center[column][row]
                    ),
                ),
                Fraction(1, 2),
            )
            combined_radius = (
                radius[row][column]
                + radius[column][row]
            ) / 2
            if row == column:
                combined_radius += abs(combined[1])
                combined = combined[0], Fraction(0)
            output_center[row][column] = combined
            output_center[column][row] = (
                _gaussian_conjugate(combined)
            )
            output_radius[row][column] = combined_radius
            output_radius[column][row] = combined_radius
    return output_center, output_radius


def _reference_block(
    exact_state: Sequence[Sequence[Gaussian]],
    norm: Fraction,
    *,
    n: int,
    first_row: int,
    p: Fraction,
) -> tuple[list[list[Gaussian]], list[list[Fraction]]]:
    degree = 2 * first_row - n
    dimension = 2 * (degree + 1)
    center = [
        [ZERO for _ in range(dimension)]
        for _ in range(dimension)
    ]
    radius = [
        [Fraction(0) for _ in range(dimension)]
        for _ in range(dimension)
    ]
    for reference_row in range(2):
        for reference_column in range(2):
            for row in range(degree + 1):
                for column in range(degree + 1):
                    box = _entry_box(
                        exact_state,
                        norm,
                        n=n,
                        first_row=first_row,
                        reference_row=reference_row,
                        reference_column=reference_column,
                        final_row=row,
                        final_column=column,
                        p=p,
                    )
                    value, error = _box_center_radius(box)
                    joint_row = (
                        reference_row * (degree + 1) + row
                    )
                    joint_column = (
                        reference_column * (degree + 1)
                        + column
                    )
                    center[joint_row][joint_column] = value
                    radius[joint_row][joint_column] = error
    return _symmetrize(center, radius)


def _partial_trace(
    joint_center: list[list[Gaussian]],
    joint_radius: list[list[Fraction]],
) -> tuple[list[list[Gaussian]], list[list[Fraction]]]:
    joint_dimension = len(joint_center)
    if joint_dimension % 2 != 0:
        _fail("joint-dimension")
    dimension = joint_dimension // 2
    center = [
        [ZERO for _ in range(dimension)]
        for _ in range(dimension)
    ]
    radius = [
        [Fraction(0) for _ in range(dimension)]
        for _ in range(dimension)
    ]
    for row in range(dimension):
        for column in range(dimension):
            for reference in range(2):
                joint_row = reference * dimension + row
                joint_column = reference * dimension + column
                center[row][column] = _gaussian_add(
                    center[row][column],
                    joint_center[joint_row][joint_column],
                )
                radius[row][column] += (
                    joint_radius[joint_row][joint_column]
                )
    return _symmetrize(center, radius)


def _atanh_log_bounds(
    value: Fraction,
) -> tuple[Fraction, Fraction]:
    if not (
        _qle(Fraction(1), value)
        and _qle(value, Fraction(2))
    ):
        _fail("log-reduced-domain")
    parameter = (value - 1) / (value + 1)
    if parameter.numerator == 0:
        return Fraction(0), Fraction(0)
    square = parameter * parameter
    grid = 10**LOG_GRID_DIGITS
    lower_scaled = 0
    upper_scaled = 0
    term = parameter
    for index in range(LOG_TERMS):
        summand = 2 * term / (2 * index + 1)
        lower_scaled += (
            summand.numerator * grid
            // summand.denominator
        )
        upper_scaled += -(
            (-summand.numerator * grid)
            // summand.denominator
        )
        term *= square
    tail = (
        2
        * term
        / ((2 * LOG_TERMS + 1) * (1 - square))
    )
    upper_scaled += -(
        (-tail.numerator * grid) // tail.denominator
    )
    return (
        Fraction(lower_scaled, grid),
        Fraction(upper_scaled, grid),
    )


@lru_cache(maxsize=None)
def _log_two_bounds() -> tuple[Fraction, Fraction]:
    return _atanh_log_bounds(Fraction(2))


def _log_bounds(
    value: Fraction,
) -> tuple[Fraction, Fraction]:
    if not (
        _qlt(Fraction(0), value)
        and _qle(value, Fraction(1))
    ):
        _fail("log-domain")
    if _qeq(value, Fraction(1)):
        return Fraction(0), Fraction(0)
    reduced = value
    powers = 0
    while _qlt(reduced, Fraction(1)):
        reduced *= 2
        powers += 1
    reduced_lower, reduced_upper = (
        _atanh_log_bounds(reduced)
    )
    two_lower, two_upper = _log_two_bounds()
    return (
        reduced_lower - powers * two_upper,
        reduced_upper - powers * two_lower,
    )


def _entropy_term_bounds(
    value: Fraction,
) -> tuple[Fraction, Fraction]:
    if value.numerator == 0:
        return Fraction(0), Fraction(0)
    logarithm_lower, logarithm_upper = _log_bounds(value)
    numerator_lower = -value * logarithm_upper
    numerator_upper = -value * logarithm_lower
    two_lower, two_upper = _log_two_bounds()
    return (
        numerator_lower / two_upper,
        numerator_upper / two_lower,
    )


def _entropy_range(
    lower: Fraction,
    upper: Fraction,
) -> tuple[
    Fraction,
    Fraction,
    Fraction,
    Fraction,
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
    bool,
]:
    clipped_lower = _qmax(Fraction(0), lower)
    clipped_upper = _qmin(Fraction(1), upper)
    if not _qle(clipped_lower, clipped_upper):
        _fail("spectrum-outside-probability-domain")
    at_lower = _entropy_term_bounds(clipped_lower)
    at_upper = _entropy_term_bounds(clipped_upper)
    minimum = _qmin(at_lower[0], at_upper[0])
    maximum = _qmax(at_lower[1], at_upper[1])
    critical = (
        _qle(
            clipped_lower,
            Fraction(367_880, 1_000_000),
        )
        and _qle(
            Fraction(367_879, 1_000_000),
            clipped_upper,
        )
    )
    if critical:
        maximum = _qmax(maximum, Fraction(1))
    return (
        clipped_lower,
        clipped_upper,
        minimum,
        maximum,
        at_lower,
        at_upper,
        critical,
    )


def _prove_e_bracket() -> None:
    lower_decimal = Fraction(367_879, 1_000_000)
    upper_decimal = Fraction(367_880, 1_000_000)
    _, lower_log_upper = _log_bounds(lower_decimal)
    upper_log_lower, _ = _log_bounds(upper_decimal)
    if not _qlt(lower_log_upper, Fraction(-1)):
        _fail("startup-e-lower-proof")
    if not _qlt(Fraction(-1), upper_log_lower):
        _fail("startup-e-upper-proof")


def _specht_dimension(n: int, first_row: int) -> int:
    return (
        math.comb(n + 1, first_row + 1)
        * (2 * first_row - n + 1)
        // (n + 1)
    )


def _verify_square_upper(
    recorded: Fraction,
    square: Fraction,
    deterministic: Fraction,
    reason: str,
) -> None:
    if recorded.numerator < 0:
        _fail(reason)
    if not _qle(square, recorded * recorded):
        _fail(reason)
    _expect_equal(recorded, deterministic, reason)


def _verify_spectral(
    center: list[list[Gaussian]],
    radius: list[list[Fraction]],
    record: dict[str, Any],
    *,
    dimension: int,
    specht_dimension: int,
) -> tuple[int, int]:
    binary_center = _parse_matrix(
        record["center_dyadic"],
        dimension,
    )
    candidate_basis = _parse_matrix(
        record["candidate_basis"],
        dimension,
    )
    _check_hermitian(binary_center, "center-hermiticity")

    bounds = record["bounds"]
    algebraic = _q(bounds["algebraic_frobenius"])
    algebraic_square = sum(
        value * value
        for row in radius
        for value in row
    )
    deterministic_algebraic = _sqrt_bounds(
        algebraic_square,
        bits=SQRT_BITS,
    )[1]
    _verify_square_upper(
        algebraic,
        algebraic_square,
        deterministic_algebraic,
        "algebraic-frobenius",
    )

    conversion_errors = [
        [
            (
                center[row][column][0]
                - binary_center[row][column][0],
                center[row][column][1]
                - binary_center[row][column][1],
            )
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]
    conversion_square = sum(
        value[0] * value[0] + value[1] * value[1]
        for row in conversion_errors
        for value in row
    )
    conversion = _q(
        bounds["center_conversion_frobenius"]
    )
    deterministic_conversion = _sqrt_bounds(
        conversion_square,
        bits=FROBENIUS_BITS,
    )[1]
    _verify_square_upper(
        conversion,
        conversion_square,
        deterministic_conversion,
        "center-conversion-frobenius",
    )

    total = _q(bounds["total_matrix_frobenius"])
    _expect_equal(
        total,
        algebraic + conversion,
        "total-matrix-frobenius",
    )

    gram = _matrix_multiply(
        _adjoint(candidate_basis),
        candidate_basis,
    )
    for index in range(dimension):
        gram[index][index] = _gaussian_add(
            gram[index][index],
            (Fraction(-1), Fraction(0)),
        )
    gram_square = sum(
        value[0] * value[0] + value[1] * value[1]
        for row in gram
        for value in row
    )
    gram_defect = _q(bounds["gram_defect_frobenius"])
    deterministic_gram = _sqrt_bounds(
        gram_square,
        bits=FROBENIUS_BITS,
    )[1]
    _verify_square_upper(
        gram_defect,
        gram_square,
        deterministic_gram,
        "gram-defect-frobenius",
    )
    if not _qlt(gram_defect, Fraction(1)):
        _fail("gram-defect-frobenius")

    one_minus = 1 - gram_defect
    one_plus = 1 + gram_defect
    sqrt_minus_lower = _q(
        bounds["sqrt_one_minus_lower"]
    )
    sqrt_plus_upper = _q(
        bounds["sqrt_one_plus_upper"]
    )
    expected_minus = _sqrt_bounds(
        one_minus,
        bits=SQRT_BITS,
    )[0]
    expected_plus = _sqrt_bounds(
        one_plus,
        bits=SQRT_BITS,
    )[1]
    if (
        sqrt_minus_lower.numerator < 0
        or not _qle(
            sqrt_minus_lower * sqrt_minus_lower,
            one_minus,
        )
    ):
        _fail("polar-square-root")
    if (
        sqrt_plus_upper.numerator < 0
        or not _qle(
            one_plus,
            sqrt_plus_upper * sqrt_plus_upper,
        )
    ):
        _fail("polar-square-root")
    _expect_equal(
        sqrt_minus_lower,
        expected_minus,
        "polar-square-root",
    )
    _expect_equal(
        sqrt_plus_upper,
        expected_plus,
        "polar-square-root",
    )

    polar_distance = _q(bounds["polar_distance"])
    expected_polar = _qmax(
        1 - sqrt_minus_lower,
        sqrt_plus_upper - 1,
    )
    if (
        not _qle(1 - sqrt_minus_lower, polar_distance)
        or not _qle(
            sqrt_plus_upper - 1,
            polar_distance,
        )
    ):
        _fail("polar-distance")
    _expect_equal(
        polar_distance,
        expected_polar,
        "polar-distance",
    )

    basis_norm = _q(bounds["basis_norm_upper"])
    _expect_equal(
        basis_norm,
        sqrt_plus_upper,
        "basis-norm-upper",
    )

    center_square = sum(
        value[0] * value[0] + value[1] * value[1]
        for row in binary_center
        for value in row
    )
    center_norm = _q(bounds["center_norm_upper"])
    deterministic_center_norm = _sqrt_bounds(
        center_square,
        bits=FROBENIUS_BITS,
    )[1]
    _verify_square_upper(
        center_norm,
        center_square,
        deterministic_center_norm,
        "center-norm-upper",
    )

    transform_error = _q(bounds["transform_error"])
    expected_transform_error = (
        total
        + polar_distance
        * center_norm
        * (1 + basis_norm)
    )
    if not _qle(expected_transform_error, transform_error):
        _fail("transform-error")
    _expect_equal(
        transform_error,
        expected_transform_error,
        "transform-error",
    )

    transformed = _matrix_multiply(
        _matrix_multiply(
            _adjoint(candidate_basis),
            binary_center,
        ),
        candidate_basis,
    )
    _check_hermitian(transformed, "transform-hermiticity")

    discs: list[tuple[Fraction, Fraction]] = []
    for row in range(dimension):
        diagonal = transformed[row][row]
        if diagonal[1].numerator != 0:
            _fail("transform-hermiticity")
        calculated_offdiagonal = sum(
            _abs_upper(transformed[row][column])
            for column in range(dimension)
            if column != row
        )
        row_record = record["rows"][row]
        recorded_offdiagonal = _q(
            row_record["offdiagonal_sum"]
        )
        _expect_equal(
            recorded_offdiagonal,
            calculated_offdiagonal,
            "offdiagonal-sum",
        )

        radius_value = (
            recorded_offdiagonal
            + dimension * transform_error
        )
        expected_lower = diagonal[0] - radius_value
        expected_upper = diagonal[0] + radius_value
        recorded_lower = _q(row_record["disc"][0])
        recorded_upper = _q(row_record["disc"][1])
        if (
            not _qle(recorded_lower, expected_lower)
            or not _qle(expected_upper, recorded_upper)
        ):
            _fail("gershgorin-containment")
        if (
            not _qeq(recorded_lower, expected_lower)
            or not _qeq(recorded_upper, expected_upper)
        ):
            _fail("gershgorin-containment")
        discs.append((recorded_lower, recorded_upper))

    discs.sort(key=cmp_to_key(_disc_cmp))
    merged: list[tuple[Fraction, Fraction, int]] = []
    for lower, upper in discs:
        if not merged or _qlt(merged[-1][1], lower):
            merged.append((lower, upper, 1))
        else:
            old_lower, old_upper, count = merged[-1]
            merged[-1] = (
                old_lower,
                _qmax(old_upper, upper),
                count + 1,
            )

    component_records = record["components"]
    if len(component_records) != len(merged):
        _fail("component-merge")
    entropy_lower = Fraction(0)
    entropy_upper = Fraction(0)
    total_count = 0

    for component_record, expected in zip(
        component_records,
        merged,
    ):
        expected_lower, expected_upper, expected_count = expected
        recorded_lower = _q(
            component_record["interval"][0]
        )
        recorded_upper = _q(
            component_record["interval"][1]
        )
        recorded_count = _integer_text(
            component_record["multiplicity"]
        )
        if (
            not _qeq(recorded_lower, expected_lower)
            or not _qeq(recorded_upper, expected_upper)
            or recorded_count != expected_count
        ):
            _fail("component-merge")
        total_count += recorded_count

        (
            clipped_lower,
            clipped_upper,
            term_lower,
            term_upper,
            endpoint_lower,
            endpoint_upper,
            critical,
        ) = _entropy_range(
            expected_lower,
            expected_upper,
        )
        if (
            not _qeq(
                _q(component_record["clipped"][0]),
                clipped_lower,
            )
            or not _qeq(
                _q(component_record["clipped"][1]),
                clipped_upper,
            )
        ):
            _fail("entropy-replay")

        endpoint_record = component_record[
            "endpoint_entropy"
        ]
        expected_endpoints = (
            endpoint_lower,
            endpoint_upper,
        )
        for endpoint_index in range(2):
            for bound_index in range(2):
                if not _qeq(
                    _q(
                        endpoint_record[endpoint_index][
                            bound_index
                        ]
                    ),
                    expected_endpoints[endpoint_index][
                        bound_index
                    ],
                ):
                    _fail("entropy-replay")

        if component_record["critical_cap_used"] is not critical:
            _fail("entropy-replay")
        if (
            not _qeq(
                _q(component_record["entropy_range"][0]),
                term_lower,
            )
            or not _qeq(
                _q(component_record["entropy_range"][1]),
                term_upper,
            )
        ):
            _fail("entropy-replay")

        entropy_lower += recorded_count * term_lower
        entropy_upper += recorded_count * term_upper

    if total_count != dimension:
        _fail("component-merge")

    scaled_lower = _floor_scaled(
        specht_dimension * entropy_lower
    )
    scaled_upper = _ceil_scaled(
        specht_dimension * entropy_upper
    )
    if (
        _integer_text(record["scaled_entropy"][0])
        != scaled_lower
        or _integer_text(record["scaled_entropy"][1])
        != scaled_upper
    ):
        _fail("scaled-entropy")
    return scaled_lower, scaled_upper


def _parse_expected_p(value: str) -> Fraction:
    parts = value.split("/")
    if len(parts) != 2:
        _fail("expected-p-format")
    return _q([parts[0], parts[1]])


def verify(
    certificate_path: Path,
    expected_payload_sha256: str,
    expected_p_text: str,
) -> tuple[str, int, int]:
    _prove_e_bracket()
    document = _load_certificate(certificate_path)

    if (
        len(expected_payload_sha256) != 64
        or HEX_RE.fullmatch(expected_payload_sha256) is None
    ):
        _fail("expected-payload-sha256")
    expected_p = _parse_expected_p(expected_p_text)
    certificate_p = _q(document["p"])
    if not _qeq(certificate_p, expected_p):
        _fail("endpoint-binding")
    if document["payload_sha256"] != expected_payload_sha256:
        _fail("payload-external-digest")

    try:
        payload = bytes.fromhex(document["payload_hex"])
    except ValueError:
        _fail("payload-hex")
    actual_payload_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_payload_sha256 != document["payload_sha256"]:
        _fail("payload-self-digest")

    decoded_state = _decode_payload(payload)
    exact_state, norm = _exact_state(decoded_state)

    output_lower = 0
    output_upper = 0
    joint_lower = 0
    joint_upper = 0
    global_trace_center = Fraction(0)
    global_trace_radius = Fraction(0)

    for index, block in enumerate(document["blocks"]):
        first_row = 23 + index
        degree = 2 * first_row - N
        if (
            _integer_text(block["first_row"]) != first_row
            or _integer_text(block["degree"]) != degree
        ):
            _fail("block-topology")

        expected_specht = _specht_dimension(N, first_row)
        recorded_specht = _integer_text(
            block["specht_dimension"]
        )
        if recorded_specht != expected_specht:
            _fail("specht-dimension")

        joint_center, joint_radius = _reference_block(
            exact_state,
            norm,
            n=N,
            first_row=first_row,
            p=certificate_p,
        )
        output_center, output_radius = _partial_trace(
            joint_center,
            joint_radius,
        )

        block_trace_center = sum(
            output_center[diagonal][diagonal][0]
            for diagonal in range(degree + 1)
        )
        block_trace_radius = sum(
            output_radius[diagonal][diagonal]
            for diagonal in range(degree + 1)
        )
        recorded_trace_center = _q(
            block["trace"]["center"]
        )
        recorded_trace_radius = _q(
            block["trace"]["radius"]
        )
        _expect_equal(
            recorded_trace_center,
            block_trace_center,
            "trace-center",
        )
        _expect_equal(
            recorded_trace_radius,
            block_trace_radius,
            "trace-radius",
        )

        block_output_lower, block_output_upper = (
            _verify_spectral(
                output_center,
                output_radius,
                block["output"],
                dimension=degree + 1,
                specht_dimension=expected_specht,
            )
        )
        block_joint_lower, block_joint_upper = (
            _verify_spectral(
                joint_center,
                joint_radius,
                block["joint"],
                dimension=2 * (degree + 1),
                specht_dimension=expected_specht,
            )
        )

        output_lower += block_output_lower
        output_upper += block_output_upper
        joint_lower += block_joint_lower
        joint_upper += block_joint_upper
        global_trace_center += (
            expected_specht * block_trace_center
        )
        global_trace_radius += (
            expected_specht * block_trace_radius
        )

    recorded_global_center = _q(
        document["global_trace"]["center"]
    )
    recorded_global_radius = _q(
        document["global_trace"]["radius"]
    )
    _expect_equal(
        recorded_global_center,
        global_trace_center,
        "global-trace",
    )
    _expect_equal(
        recorded_global_radius,
        global_trace_radius,
        "global-trace",
    )
    if not (
        _qle(
            global_trace_center - global_trace_radius,
            Fraction(1),
        )
        and _qle(
            Fraction(1),
            global_trace_center + global_trace_radius,
        )
    ):
        _fail("global-trace")

    coherent_lower = output_lower - joint_upper
    coherent_upper = output_upper - joint_lower
    expected_scaled = {
        "output_lower": output_lower,
        "output_upper": output_upper,
        "joint_lower": joint_lower,
        "joint_upper": joint_upper,
        "coherent_lower": coherent_lower,
        "coherent_upper": coherent_upper,
    }
    for key, expected in expected_scaled.items():
        if _integer_text(document["scaled"][key]) != expected:
            _fail("scaled-values")

    classification = (
        "POS"
        if coherent_lower > 0
        else "NEG"
        if coherent_upper < 0
        else "UNRESOLVED"
    )
    if document["classification"] != classification:
        _fail("classification")
    return classification, coherent_lower, coherent_upper


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 3:
        print("REJECT usage")
        return 1
    try:
        classification, lower, upper = verify(
            Path(arguments[0]),
            arguments[1],
            arguments[2],
        )
    except VerificationError as exc:
        print(f"REJECT {exc}")
        return 1
    except OSError:
        print("REJECT io-error")
        return 1
    except Exception:
        print("REJECT internal-error")
        return 1

    print(
        f"ACCEPT {classification} coherent "
        f"in [{lower},{upper}]x1e-50"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import statistics
import time
from functools import partial
from pathlib import Path

import src.edwards25519curve as curve
import src.field as field
import src.scalar_multiply as scalar_multiply

REPEATS = 21
ITERATIONS = 5

# k <= L/2: the modification does not change the scalar.
UNCHANGED_SCALARS = [
    ("k = 1", 1),
    ("k = 2^32", 2**32),
    ("k = 2^128", 2**128),
    ("k = L // 10", field.SUBGROUP_ORDER // 10),
    ("k = L // 4", field.SUBGROUP_ORDER // 4),
    ("k = L // 2", field.SUBGROUP_ORDER // 2),
]

# k > L/2: the working scalar is L-k.
COMPLEMENT_SCALARS = [
    ("k = L // 2 + 1", field.SUBGROUP_ORDER // 2 + 1),
    ("k = 3L // 4", 3 * field.SUBGROUP_ORDER // 4),
    ("k = 9L // 10", 9 * field.SUBGROUP_ORDER // 10),
    ("k = 99L // 100", 99 * field.SUBGROUP_ORDER // 100),
    (
        "k = L - (2^128 + 1)",
        field.SUBGROUP_ORDER - (2**128 + 1),
    ),
    (
        "k = L - (2^128 + 2^64 + 1)",
        field.SUBGROUP_ORDER - (2**128 + 2**64 + 1),
    ),
]

# The working scalar is a power of two with one set bit.
POWER_COMPLEMENT_SCALARS = [
    ("k = L - 2^191", field.SUBGROUP_ORDER - 2**191),
    ("k = L - 2^127", field.SUBGROUP_ORDER - 2**127),
    ("k = L - 2^63", field.SUBGROUP_ORDER - 2**63),
    ("k = L - 2^31", field.SUBGROUP_ORDER - 2**31),
    ("k = L - 2^15", field.SUBGROUP_ORDER - 2**15),
    ("k = L - 1", field.SUBGROUP_ORDER - 1),
]


def measure_operation(
    scalar: int,
    method: scalar_multiply.ScalarMultiplicationMethod,
    modified: bool,
) -> float:
    point = curve.BASE_EXTENDED_POINT
    start_time = time.perf_counter()

    if modified:
        scalar_multiply.scalar_multiply_modified(
            scalar,
            point,
            method,
        )
    else:
        method(scalar, point)

    return time.perf_counter() - start_time


def measure_pair(
    scalar: int,
    method: scalar_multiply.ScalarMultiplicationMethod,
) -> tuple[float, float]:
    base_samples = []
    modified_samples = []
    point = curve.BASE_EXTENDED_POINT

    method(scalar, point)
    scalar_multiply.scalar_multiply_modified(scalar, point, method)

    for repeat_index in range(REPEATS):
        for iteration_index in range(ITERATIONS):
            base_first = (
                repeat_index + iteration_index
            ) % 2 == 0

            if base_first:
                base_samples.append(
                    measure_operation(scalar, method, False)
                )
                modified_samples.append(
                    measure_operation(scalar, method, True)
                )
            else:
                modified_samples.append(
                    measure_operation(scalar, method, True)
                )
                base_samples.append(
                    measure_operation(scalar, method, False)
                )

    return (
        statistics.median(base_samples),
        statistics.median(modified_samples),
    )


def write_results(
    method_name: str,
    method: scalar_multiply.ScalarMultiplicationMethod,
    output_path: Path,
) -> None:
    lines = [
        method_name,
        "",
        f"{'Scalar':<34} {'Base, s':>14} "
        f"{'Modified, s':>14} {'Gain, %':>10}",
        "-" * 75,
    ]
    scalar_groups = (
        UNCHANGED_SCALARS,
        COMPLEMENT_SCALARS,
        POWER_COMPLEMENT_SCALARS,
    )

    for scalar_group in scalar_groups:
        for scalar_name, scalar in scalar_group:
            base_point = method(scalar, curve.BASE_EXTENDED_POINT)
            modified_point = scalar_multiply.scalar_multiply_modified(
                scalar,
                curve.BASE_EXTENDED_POINT,
                method,
            )

            if curve.to_affine_point(
                base_point
            ) != curve.to_affine_point(modified_point):
                raise RuntimeError("Different multiplication results")

            base_time, modified_time = measure_pair(scalar, method)
            gain = (base_time - modified_time) / base_time * 100

            lines.append(
                f"{scalar_name:<34} "
                f"{base_time:>14.9f} "
                f"{modified_time:>14.9f} "
                f"{gain:>10.2f}"
            )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    methods = {
        "double_and_add": scalar_multiply.scalar_multiply_double_and_add,
        "montgomery": scalar_multiply.scalar_multiply_montgomery,
        "window_w4": partial(
            scalar_multiply.scalar_multiply_window,
            window_size=4,
        ),
        "window_w6": partial(
            scalar_multiply.scalar_multiply_window,
            window_size=6,
        ),
        "window_w9": partial(
            scalar_multiply.scalar_multiply_window,
            window_size=9,
        ),
    }
    results_directory = Path(__file__).parent / "results"
    results_directory.mkdir(exist_ok=True)

    for method_name, method in methods.items():
        output_path = results_directory / f"{method_name}.txt"
        write_results(method_name, method, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import src.field as field
import src.schnorr as schnorr
import src.scalar_multiply as scalar_multiply

REPEATS = 21
ITERATIONS = 5

PRIVATE_KEYS = [
    field.SUBGROUP_ORDER // 4,
    9 * field.SUBGROUP_ORDER // 10,
    field.SUBGROUP_ORDER - 2**15,
]
MESSAGE_SIZES = [4096, 16384, 262144, 1048576]


def measure_operation(operation: Callable[[], object]) -> float:
    start_time = time.perf_counter()
    operation()

    return time.perf_counter() - start_time


def measure_pair(
    base_operation: Callable[[], object],
    modified_operation: Callable[[], object],
) -> tuple[float, float]:
    base_samples = []
    modified_samples = []

    base_operation()
    modified_operation()

    for repeat_index in range(REPEATS):
        for iteration_index in range(ITERATIONS):
            base_first = (
                repeat_index + iteration_index
            ) % 2 == 0

            if base_first:
                base_samples.append(
                    measure_operation(base_operation)
                )
                modified_samples.append(
                    measure_operation(modified_operation)
                )
            else:
                modified_samples.append(
                    measure_operation(modified_operation)
                )
                base_samples.append(
                    measure_operation(base_operation)
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
    table_header = (
        f"{'Key':<8} {'Message, B':>12} {'Base, s':>14} "
        f"{'Modified, s':>14} {'Gain, %':>10}"
    )
    lines = [
        method_name,
        "",
        "sign",
        "",
        table_header,
        "-" * 72,
    ]
    signing_keys = [
        schnorr.SigningKey.from_private_key(private_key, method)
        for private_key in PRIVATE_KEYS
    ]
    signatures = {}

    for key_index, signing_key in enumerate(signing_keys, start=1):
        for message_size in MESSAGE_SIZES:
            message = bytes(message_size)
            base_operation = partial(
                schnorr.sign,
                message,
                signing_key,
                method,
                use_modification=False,
            )
            modified_operation = partial(
                schnorr.sign,
                message,
                signing_key,
                method,
                use_modification=True,
            )

            base_signature = base_operation()
            modified_signature = modified_operation()

            if base_signature != modified_signature:
                raise RuntimeError("Different signatures")

            signatures[(key_index, message_size)] = base_signature
            base_time, modified_time = measure_pair(
                base_operation,
                modified_operation,
            )
            gain = (base_time - modified_time) / base_time * 100

            lines.append(
                f"key_{key_index:<3} {message_size:>12} "
                f"{base_time:>14.9f} {modified_time:>14.9f} "
                f"{gain:>10.2f}"
            )

    lines.extend([
        "",
        "verify",
        "",
        table_header,
        "-" * 72,
    ])

    for key_index, signing_key in enumerate(signing_keys, start=1):
        for message_size in MESSAGE_SIZES:
            message = bytes(message_size)
            signature = signatures[(key_index, message_size)]
            base_operation = partial(
                schnorr.verify,
                message,
                signing_key.public_key,
                signature,
                method,
                use_modification=False,
            )
            modified_operation = partial(
                schnorr.verify,
                message,
                signing_key.public_key,
                signature,
                method,
                use_modification=True,
            )

            if not base_operation() or not modified_operation():
                raise RuntimeError("Signature verification failed")

            base_time, modified_time = measure_pair(
                base_operation,
                modified_operation,
            )
            gain = (base_time - modified_time) / base_time * 100

            lines.append(
                f"key_{key_index:<3} {message_size:>12} "
                f"{base_time:>14.9f} {modified_time:>14.9f} "
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
    results_directory = Path(__file__).parent / "signature_results"
    results_directory.mkdir(exist_ok=True)

    for method_name, method in methods.items():
        output_path = results_directory / f"{method_name}.txt"
        write_results(method_name, method, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

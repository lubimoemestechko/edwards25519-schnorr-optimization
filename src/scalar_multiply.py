from collections.abc import Callable

import src.edwards25519curve as curve
import src.field as field

ScalarMultiplicationMethod = Callable[
    [int, curve.ExtendedPoint],
    curve.ExtendedPoint,
]


def scalar_multiply_double_and_add(
    scalar: int,
    point: curve.ExtendedPoint,
) -> curve.ExtendedPoint:
    if scalar < 0:
        raise ValueError("Scalar must not be negative")

    remaining_scalar = scalar
    result_point = curve.IDENTITY_EXTENDED_POINT
    current_point = point

    while remaining_scalar > 0:
        if remaining_scalar & 1:
            result_point = curve.add_extended_points(
                result_point,
                current_point,
            )

        remaining_scalar >>= 1

        if remaining_scalar > 0:
            current_point = curve.double_extended_point(current_point)

    return result_point


def scalar_multiply_montgomery(
    scalar: int,
    point: curve.ExtendedPoint,
) -> curve.ExtendedPoint:
    if scalar < 0:
        raise ValueError("Scalar must not be negative")

    first_accumulator = curve.IDENTITY_EXTENDED_POINT
    second_accumulator = point

    for bit_index in range(scalar.bit_length() - 1, -1, -1):
        current_bit = (scalar >> bit_index) & 1

        if current_bit == 0:
            second_accumulator = curve.add_extended_points(
                first_accumulator,
                second_accumulator,
            )
            first_accumulator = curve.double_extended_point(
                first_accumulator,
            )
        else:
            first_accumulator = curve.add_extended_points(
                first_accumulator,
                second_accumulator,
            )
            second_accumulator = curve.double_extended_point(
                second_accumulator,
            )

    return first_accumulator


def build_window_table(
    point: curve.ExtendedPoint,
    window_size: int,
) -> list[curve.ExtendedPoint]:
    if window_size <= 0:
        raise ValueError("Window size must be positive")

    point_table = [curve.IDENTITY_EXTENDED_POINT, point]

    for table_index in range(2, 2**window_size):
        point_table.append(
            curve.add_extended_points(point_table[-1], point)
        )

    return point_table


def scalar_multiply_window_with_table(
    scalar: int,
    point_table: list[curve.ExtendedPoint],
    window_size: int,
) -> curve.ExtendedPoint:
    if scalar < 0:
        raise ValueError("Scalar must not be negative")

    if window_size <= 0:
        raise ValueError("Window size must be positive")

    if len(point_table) != 2**window_size:
        raise ValueError("Point table does not match window size")

    if scalar == 0:
        return curve.IDENTITY_EXTENDED_POINT

    window_mask = (1 << window_size) - 1
    remaining_scalar = scalar
    window_values = []

    while remaining_scalar > 0:
        window_values.append(remaining_scalar & window_mask)
        remaining_scalar >>= window_size

    result_point = point_table[window_values[-1]]

    for current_window in reversed(window_values[:-1]):
        for _ in range(window_size):
            result_point = curve.double_extended_point(result_point)

        if current_window != 0:
            result_point = curve.add_extended_points(
                result_point,
                point_table[current_window],
            )

    return result_point


def scalar_multiply_window(
    scalar: int,
    point: curve.ExtendedPoint,
    window_size: int = 4,
) -> curve.ExtendedPoint:
    point_table = build_window_table(point, window_size)

    return scalar_multiply_window_with_table(
        scalar,
        point_table,
        window_size,
    )


class WindowMultiplier:
    window_size: int
    cache_tables: bool
    point_tables: dict[
        curve.ExtendedPoint,
        list[curve.ExtendedPoint],
    ]

    def __init__(
        self,
        window_size: int = 4,
        cache_tables: bool = True,
    ) -> None:
        if window_size <= 0:
            raise ValueError("Window size must be positive")

        self.window_size = window_size
        self.cache_tables = cache_tables
        self.point_tables = {}

    def __call__(
        self,
        scalar: int,
        point: curve.ExtendedPoint,
    ) -> curve.ExtendedPoint:
        if not self.cache_tables:
            return scalar_multiply_window(
                scalar,
                point,
                self.window_size,
            )

        if point not in self.point_tables:
            self.point_tables[point] = build_window_table(
                point,
                self.window_size,
            )

        return scalar_multiply_window_with_table(
            scalar,
            self.point_tables[point],
            self.window_size,
        )


def scalar_multiply_modified(
    scalar: int,
    point: curve.ExtendedPoint,
    multiplication_method: ScalarMultiplicationMethod,
) -> curve.ExtendedPoint:
    if scalar < 0:
        raise ValueError("Scalar must not be negative")

    normalized_scalar = scalar % field.SUBGROUP_ORDER
    use_complement = normalized_scalar > field.SUBGROUP_ORDER // 2

    if use_complement:
        reduced_scalar = field.SUBGROUP_ORDER - normalized_scalar
    else:
        reduced_scalar = normalized_scalar

    result_point = multiplication_method(
        reduced_scalar,
        point,
    )

    if use_complement:
        result_point = curve.negative_extended_point(result_point)

    return result_point

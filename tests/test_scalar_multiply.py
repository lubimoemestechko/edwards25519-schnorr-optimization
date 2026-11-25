import pytest

import src.edwards25519curve as curve
import src.field as field
import src.scalar_multiply as scalar_multiply


def multiply_affine_reference(scalar: int) -> curve.AffinePoint:
    result_point = curve.IDENTITY_POINT

    for _ in range(scalar):
        result_point = curve.add_affine_points(
            result_point,
            curve.BASE_POINT,
        )

    return result_point


@pytest.mark.parametrize("scalar", [0, 1, 2, 3, 5, 17])
def test_methods_match_affine_reference(
    scalar: int,
    multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
) -> None:
    actual_point = multiplication_method(
        scalar,
        curve.BASE_EXTENDED_POINT,
    )

    assert curve.to_affine_point(
        actual_point
    ) == multiply_affine_reference(scalar)


@pytest.mark.parametrize(
    "scalar",
    [2**32 + 1, field.SUBGROUP_ORDER // 2, field.SUBGROUP_ORDER - 1,
     field.SUBGROUP_ORDER],
)
def test_methods_match_double_and_add(
    scalar: int,
    multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
) -> None:
    expected_point = scalar_multiply.scalar_multiply_double_and_add(
        scalar,
        curve.BASE_EXTENDED_POINT,
    )
    actual_point = multiplication_method(
        scalar,
        curve.BASE_EXTENDED_POINT,
    )

    assert curve.to_affine_point(actual_point) == curve.to_affine_point(
        expected_point
    )


@pytest.mark.parametrize(
    "scalar",
    [
        0,
        1,
        field.SUBGROUP_ORDER // 2,
        field.SUBGROUP_ORDER // 2 + 1,
        field.SUBGROUP_ORDER - 27,
        field.SUBGROUP_ORDER - 1,
        field.SUBGROUP_ORDER,
        field.SUBGROUP_ORDER + 1,
    ],
)
def test_modified_multiplication_matches_direct(
    scalar: int,
    multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
) -> None:
    expected_point = multiplication_method(
        scalar,
        curve.BASE_EXTENDED_POINT,
    )
    actual_point = scalar_multiply.scalar_multiply_modified(
        scalar,
        curve.BASE_EXTENDED_POINT,
        multiplication_method,
    )

    assert curve.to_affine_point(actual_point) == curve.to_affine_point(
        expected_point
    )


@pytest.mark.parametrize(
    "scalar",
    [
        field.SUBGROUP_ORDER // 2 + 1,
        field.SUBGROUP_ORDER - 27,
        field.SUBGROUP_ORDER - 1,
    ],
)
def test_modified_multiplication_uses_inverse(
    scalar: int,
    multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
) -> None:
    reduced_scalar = field.SUBGROUP_ORDER - scalar
    reduced_point = multiplication_method(
        reduced_scalar,
        curve.BASE_EXTENDED_POINT,
    )
    expected_point = curve.negative_extended_point(reduced_point)
    actual_point = scalar_multiply.scalar_multiply_modified(
        scalar,
        curve.BASE_EXTENDED_POINT,
        multiplication_method,
    )

    assert curve.to_affine_point(actual_point) == curve.to_affine_point(
        expected_point
    )


def test_negative_scalars_are_rejected(
    multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
) -> None:
    with pytest.raises(ValueError):
        multiplication_method(-1, curve.BASE_EXTENDED_POINT)

    with pytest.raises(ValueError):
        scalar_multiply.scalar_multiply_modified(
            -1,
            curve.BASE_EXTENDED_POINT,
            multiplication_method,
        )


def test_window_rejects_invalid_window_size() -> None:
    with pytest.raises(ValueError):
        scalar_multiply.scalar_multiply_window(
            1,
            curve.BASE_EXTENDED_POINT,
            window_size=0,
        )

    with pytest.raises(ValueError):
        scalar_multiply.WindowMultiplier(window_size=0)


def test_window_with_table_rejects_wrong_table_size() -> None:
    with pytest.raises(ValueError):
        scalar_multiply.scalar_multiply_window_with_table(
            1,
            [curve.IDENTITY_EXTENDED_POINT, curve.BASE_EXTENDED_POINT],
            window_size=4,
        )


def test_window_multiplier_reuses_cached_table() -> None:
    multiplier = scalar_multiply.WindowMultiplier(
        window_size=4,
        cache_tables=True,
    )

    first_point = multiplier(5, curve.BASE_EXTENDED_POINT)
    cached_table = multiplier.point_tables[curve.BASE_EXTENDED_POINT]
    second_point = multiplier(17, curve.BASE_EXTENDED_POINT)

    assert len(multiplier.point_tables) == 1
    assert multiplier.point_tables[curve.BASE_EXTENDED_POINT] is cached_table
    assert curve.to_affine_point(first_point) == multiply_affine_reference(5)
    assert curve.to_affine_point(second_point) == multiply_affine_reference(17)


def test_window_multiplier_can_disable_cache() -> None:
    multiplier = scalar_multiply.WindowMultiplier(
        window_size=4,
        cache_tables=False,
    )

    multiplier(5, curve.BASE_EXTENDED_POINT)

    assert multiplier.point_tables == {}

import pytest

import src.edwards25519curve as curve


@pytest.mark.parametrize(
    "point",
    [curve.IDENTITY_POINT, curve.BASE_POINT],
)
def test_known_points_are_on_curve(point: curve.AffinePoint) -> None:
    assert curve.is_on_curve(point)


def test_identity_is_neutral_for_addition() -> None:
    assert curve.add_affine_points(
        curve.IDENTITY_POINT,
        curve.BASE_POINT,
    ) == curve.BASE_POINT
    assert curve.add_affine_points(
        curve.BASE_POINT,
        curve.IDENTITY_POINT,
    ) == curve.BASE_POINT


def test_point_plus_its_negative_is_identity() -> None:
    negative_base = curve.negative_point(curve.BASE_POINT)

    assert curve.add_affine_points(
        curve.BASE_POINT,
        negative_base,
    ) == curve.IDENTITY_POINT


def test_affine_and_extended_doubling_match() -> None:
    expected = curve.double_affine_point(curve.BASE_POINT)
    actual = curve.to_affine_point(
        curve.double_extended_point(curve.BASE_EXTENDED_POINT)
    )

    assert curve.is_on_curve(expected)
    assert actual == expected


def test_affine_and_extended_addition_match() -> None:
    second_point = curve.double_affine_point(curve.BASE_POINT)
    expected = curve.add_affine_points(
        curve.BASE_POINT,
        second_point,
    )
    actual = curve.to_affine_point(
        curve.add_extended_points(
            curve.BASE_EXTENDED_POINT,
            curve.to_extended_point(second_point),
        )
    )

    assert actual == expected


@pytest.mark.parametrize(
    "point",
    [
        curve.BASE_POINT,
        curve.double_affine_point(curve.BASE_POINT),
        curve.negative_point(curve.BASE_POINT),
        curve.IDENTITY_POINT,
    ],
)
def test_extended_conversion_round_trip(
    point: curve.AffinePoint,
) -> None:
    extended_point = curve.to_extended_point(point)

    assert curve.to_affine_point(extended_point) == point


def test_extended_negative_matches_affine_negative() -> None:
    negative_point = curve.negative_extended_point(
        curve.BASE_EXTENDED_POINT
    )

    assert curve.to_affine_point(
        negative_point
    ) == curve.negative_point(curve.BASE_POINT)

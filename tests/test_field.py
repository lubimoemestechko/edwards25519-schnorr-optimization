import pytest

import src.field as field


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (1, 1),
        (field.FIELD_MODULUS, 0),
        (field.FIELD_MODULUS + 1, 1),
        (-1, field.FIELD_MODULUS - 1),
        (-field.FIELD_MODULUS - 1, field.FIELD_MODULUS - 1),
    ],
)
def test_normalize_field_element(value: int, expected: int) -> None:
    assert field.normalize_field_element(value) == expected


@pytest.mark.parametrize("value", [1, 2, 42, field.FIELD_MODULUS - 1])
def test_invert_field_element(value: int) -> None:
    inverse = field.invert_field_element(value)

    assert 0 < inverse < field.FIELD_MODULUS
    assert value * inverse % field.FIELD_MODULUS == 1


def test_zero_has_no_inverse() -> None:
    with pytest.raises(ZeroDivisionError):
        field.invert_field_element(0)

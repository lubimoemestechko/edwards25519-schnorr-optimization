import pytest

import src.edwards25519curve as curve
import src.encoding as encoding
import src.field as field

BASE_POINT_ENCODING = bytes.fromhex("58" + "66" * 31)
IDENTITY_ENCODING = b"\x01" + b"\x00" * 31


def test_base_point_encoding_matches_rfc8032() -> None:
    assert encoding.encode_point(
        curve.BASE_EXTENDED_POINT
    ) == BASE_POINT_ENCODING


def test_identity_encoding() -> None:
    assert encoding.encode_point(
        curve.IDENTITY_EXTENDED_POINT
    ) == IDENTITY_ENCODING


@pytest.mark.parametrize(
    "point",
    [
        curve.BASE_EXTENDED_POINT,
        curve.double_extended_point(curve.BASE_EXTENDED_POINT),
        curve.negative_extended_point(curve.BASE_EXTENDED_POINT),
        curve.IDENTITY_EXTENDED_POINT,
    ],
)
def test_point_encoding_round_trip(
    point: curve.ExtendedPoint,
) -> None:
    decoded_point = encoding.decode_point(encoding.encode_point(point))

    assert curve.to_affine_point(
        decoded_point
    ) == curve.to_affine_point(point)


@pytest.mark.parametrize("scalar", [0, 1, 42, field.SUBGROUP_ORDER - 1])
def test_scalar_encoding_round_trip(scalar: int) -> None:
    assert encoding.decode_scalar(
        encoding.encode_scalar(scalar)
    ) == scalar


@pytest.mark.parametrize("encoded_point", [bytes(31), bytes(33)])
def test_point_encoding_rejects_invalid_length(
    encoded_point: bytes,
) -> None:
    with pytest.raises(ValueError):
        encoding.decode_point(encoded_point)


@pytest.mark.parametrize("encoded_scalar", [bytes(31), bytes(33)])
def test_scalar_encoding_rejects_invalid_length(
    encoded_scalar: bytes,
) -> None:
    with pytest.raises(ValueError):
        encoding.decode_scalar(encoded_scalar)


def test_point_decoding_rejects_noncanonical_y() -> None:
    with pytest.raises(ValueError):
        encoding.decode_point(
            field.FIELD_MODULUS.to_bytes(32, byteorder="little")
        )


def test_point_decoding_rejects_one_sign_bit_for_zero_x() -> None:
    invalid_encoding = bytearray(IDENTITY_ENCODING)
    invalid_encoding[31] |= 0x80

    with pytest.raises(ValueError):
        encoding.decode_point(bytes(invalid_encoding))


@pytest.mark.parametrize("scalar", [-1, field.SUBGROUP_ORDER])
def test_scalar_encoding_rejects_out_of_range_values(
    scalar: int,
) -> None:
    with pytest.raises(ValueError):
        encoding.encode_scalar(scalar)


def test_scalar_decoding_rejects_noncanonical_value() -> None:
    with pytest.raises(ValueError):
        encoding.decode_scalar(
            field.SUBGROUP_ORDER.to_bytes(32, byteorder="little")
        )

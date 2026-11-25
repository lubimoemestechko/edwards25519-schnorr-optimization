import src.edwards25519curve as curve
import src.field as field


SQRT_MINUS_ONE = pow(
    2,
    (field.FIELD_MODULUS - 1) // 4,
    field.FIELD_MODULUS,
)


def encode_point(point: curve.ExtendedPoint) -> bytes:
    affine_point = curve.to_affine_point(point)

    x_coordinate = affine_point.x % field.FIELD_MODULUS
    y_coordinate = affine_point.y % field.FIELD_MODULUS

    encoded_point = bytearray(
        y_coordinate.to_bytes(32, byteorder="little")
    )

    sign_bit = x_coordinate & 1
    encoded_point[31] |= sign_bit << 7

    return bytes(encoded_point)


def decode_point(encoded_point: bytes) -> curve.ExtendedPoint:
    if len(encoded_point) != 32:
        raise ValueError("Encoded point must be 32 bytes")

    encoded_bytes = bytearray(encoded_point)

    sign_bit = (encoded_bytes[31] >> 7) & 1
    encoded_bytes[31] &= 0x7F

    y_coordinate = int.from_bytes(
        encoded_bytes,
        byteorder="little",
    )

    if y_coordinate >= field.FIELD_MODULUS:
        raise ValueError("Y-coordinate is not canonical")

    y_squared = (
        y_coordinate * y_coordinate
    ) % field.FIELD_MODULUS

    numerator = (y_squared - 1) % field.FIELD_MODULUS
    denominator = (
        field.CURVE_PARAMETER_D * y_squared + 1
    ) % field.FIELD_MODULUS

    x_squared = (
        numerator * field.invert_field_element(denominator)
    ) % field.FIELD_MODULUS

    x_coordinate = pow(
        x_squared,
        (field.FIELD_MODULUS + 3) // 8,
        field.FIELD_MODULUS,
    )

    if (
        x_coordinate * x_coordinate
    ) % field.FIELD_MODULUS != x_squared:
        x_coordinate = (
            x_coordinate * SQRT_MINUS_ONE
        ) % field.FIELD_MODULUS

    if (
        x_coordinate * x_coordinate
    ) % field.FIELD_MODULUS != x_squared:
        raise ValueError("Encoded point is not on the curve")

    if x_coordinate == 0 and sign_bit == 1:
        raise ValueError("Invalid sign bit for zero x-coordinate")

    if (x_coordinate & 1) != sign_bit:
        x_coordinate = field.FIELD_MODULUS - x_coordinate

    point = curve.AffinePoint(
        x_coordinate,
        y_coordinate,
    )

    if not curve.is_on_curve(point):
        raise ValueError("Point is not on Edwards25519")

    return curve.to_extended_point(point)


def encode_scalar(scalar: int) -> bytes:
    if not 0 <= scalar < field.SUBGROUP_ORDER:
        raise ValueError("Scalar is outside the valid range")

    return scalar.to_bytes(32, byteorder="little")


def decode_scalar(encoded_scalar: bytes) -> int:
    if len(encoded_scalar) != 32:
        raise ValueError("Encoded scalar must be 32 bytes")

    scalar = int.from_bytes(
        encoded_scalar,
        byteorder="little",
    )

    if scalar >= field.SUBGROUP_ORDER:
        raise ValueError("Scalar is not canonical")

    return scalar
from dataclasses import dataclass
import src.field as field


@dataclass(frozen=True)
class AffinePoint:
    x: int
    y: int


IDENTITY_POINT = AffinePoint(0, 1)
BASE_POINT = AffinePoint(field.BASE_POINT_X, field.BASE_POINT_Y)


def is_on_curve(point: AffinePoint) -> bool:
    x_squared = point.x**2
    y_squared = point.y**2
    left_side = (-x_squared + y_squared) % field.FIELD_MODULUS
    right_side = (
        1 + field.CURVE_PARAMETER_D * x_squared * y_squared
    ) % field.FIELD_MODULUS
    return left_side == right_side


def negative_point(point: AffinePoint) -> AffinePoint:
    return AffinePoint(-point.x % field.FIELD_MODULUS, point.y)


def add_affine_points(first_point: AffinePoint, second_point: AffinePoint) -> AffinePoint:
    x_product = first_point.x * second_point.x
    y_product = first_point.y * second_point.y
    shared_product = (
        field.CURVE_PARAMETER_D * x_product * y_product
    ) % field.FIELD_MODULUS

    x_numerator = (
        first_point.x * second_point.y
        + first_point.y * second_point.x
    ) % field.FIELD_MODULUS
    x_denominator = (1 + shared_product) % field.FIELD_MODULUS

    y_numerator = (
        y_product - field.CURVE_PARAMETER_A * x_product
    ) % field.FIELD_MODULUS
    y_denominator = (1 - shared_product) % field.FIELD_MODULUS

    result_x = (
        x_numerator * field.invert_field_element(x_denominator)
    ) % field.FIELD_MODULUS
    result_y = (
        y_numerator * field.invert_field_element(y_denominator)
    ) % field.FIELD_MODULUS

    return AffinePoint(result_x, result_y)


def double_affine_point(point: AffinePoint) -> AffinePoint:
    return add_affine_points(point, point)


@dataclass(frozen=True)
class ExtendedPoint:
    x: int
    y: int
    z: int
    t: int


IDENTITY_EXTENDED_POINT = ExtendedPoint(0, 1, 1, 0)


def to_extended_point(point: AffinePoint) -> ExtendedPoint:
    return ExtendedPoint(
        point.x,
        point.y,
        1,
        point.x * point.y % field.FIELD_MODULUS,
    )


def to_affine_point(point: ExtendedPoint) -> AffinePoint:
    z_inverse = field.invert_field_element(point.z)
    result_x = point.x * z_inverse % field.FIELD_MODULUS
    result_y = point.y * z_inverse % field.FIELD_MODULUS
    return AffinePoint(result_x, result_y)


BASE_EXTENDED_POINT = to_extended_point(BASE_POINT)


def negative_extended_point(point: ExtendedPoint) -> ExtendedPoint:
    return ExtendedPoint(
        -point.x % field.FIELD_MODULUS,
        point.y,
        point.z,
        -point.t % field.FIELD_MODULUS,
    )


def add_extended_points(first_point: ExtendedPoint, second_point: ExtendedPoint) -> ExtendedPoint:
    difference_product = (
        (first_point.y - first_point.x)
        * (second_point.y - second_point.x)
    ) % field.FIELD_MODULUS
    sum_product = (
        (first_point.y + first_point.x)
        * (second_point.y + second_point.x)
    ) % field.FIELD_MODULUS
    scaled_t_product = (
        2
        * field.CURVE_PARAMETER_D
        * first_point.t
        * second_point.t
    ) % field.FIELD_MODULUS
    doubled_z_product = (
        2 * first_point.z * second_point.z
    ) % field.FIELD_MODULUS

    x_factor = (sum_product - difference_product) % field.FIELD_MODULUS
    z_factor = (
        doubled_z_product - scaled_t_product
    ) % field.FIELD_MODULUS
    y_factor = (
        doubled_z_product + scaled_t_product
    ) % field.FIELD_MODULUS
    t_factor = (sum_product + difference_product) % field.FIELD_MODULUS

    return ExtendedPoint(
        x_factor * z_factor % field.FIELD_MODULUS,
        y_factor * t_factor % field.FIELD_MODULUS,
        z_factor * y_factor % field.FIELD_MODULUS,
        x_factor * t_factor % field.FIELD_MODULUS,
    )


def double_extended_point(point: ExtendedPoint) -> ExtendedPoint:
    x_squared = point.x**2 % field.FIELD_MODULUS
    y_squared = point.y**2 % field.FIELD_MODULUS
    doubled_z_squared = 2 * point.z**2 % field.FIELD_MODULUS
    negated_x_squared = -x_squared % field.FIELD_MODULUS

    cross_term = (
        (point.x + point.y) ** 2 - x_squared - y_squared
    ) % field.FIELD_MODULUS
    y_sum = (negated_x_squared + y_squared) % field.FIELD_MODULUS
    z_factor = (y_sum - doubled_z_squared) % field.FIELD_MODULUS
    y_difference = (
        negated_x_squared - y_squared
    ) % field.FIELD_MODULUS

    return ExtendedPoint(
        cross_term * z_factor % field.FIELD_MODULUS,
        y_sum * y_difference % field.FIELD_MODULUS,
        z_factor * y_sum % field.FIELD_MODULUS,
        cross_term * y_difference % field.FIELD_MODULUS,
    )

FIELD_MODULUS = 2**255 - 19
SUBGROUP_ORDER = 2**252 + 27742317777372353535851937790883648493

BASE_POINT_X = (
    15112221349535400772501151409588531511454012693041857206046113283949847762202
)

BASE_POINT_Y = (
    46316835694926478169428394003475163141307993866256225615783033603165251855960
)

CURVE_PARAMETER_A = -1
CURVE_PARAMETER_D = (
    -121665 * pow(121666, -1, FIELD_MODULUS)
) % FIELD_MODULUS


def normalize_field_element(field_element: int) -> int:
    return field_element % FIELD_MODULUS


def invert_field_element(field_element: int) -> int:
    normalized_field_element = normalize_field_element(field_element)

    if normalized_field_element == 0:
        raise ZeroDivisionError(
            "Zero has no multiplicative inverse in this field"
        )

    return pow(normalized_field_element, -1, FIELD_MODULUS)

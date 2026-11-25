from functools import partial

import pytest

import src.field as field
import src.schnorr as schnorr
import src.scalar_multiply as scalar_multiply


@pytest.fixture(
    params=[
        scalar_multiply.scalar_multiply_double_and_add,
        scalar_multiply.scalar_multiply_montgomery,
        partial(
            scalar_multiply.scalar_multiply_window,
            window_size=4,
        ),
        partial(
            scalar_multiply.scalar_multiply_window,
            window_size=6,
        ),
        partial(
            scalar_multiply.scalar_multiply_window,
            window_size=9,
        ),
    ],
    ids=[
        "double_and_add",
        "montgomery",
        "window_w4",
        "window_w6",
        "window_w9",
    ],
)
def multiplication_method(
    request: pytest.FixtureRequest,
) -> scalar_multiply.ScalarMultiplicationMethod:
    return request.param


@pytest.fixture(scope="module")
def signing_key() -> schnorr.SigningKey:
    return schnorr.SigningKey.from_private_key(
        field.SUBGROUP_ORDER // 4,
        scalar_multiply.scalar_multiply_double_and_add,
    )

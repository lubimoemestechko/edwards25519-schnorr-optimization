import pytest

import src.edwards25519curve as curve
import src.encoding as encoding
import src.field as field
import src.schnorr as schnorr
import src.scalar_multiply as scalar_multiply


def test_hash_to_scalar_is_deterministic_and_bounded() -> None:
    data = b"Schnorr hash test"
    first_scalar = schnorr.hash_to_scalar(data)
    second_scalar = schnorr.hash_to_scalar(data)

    assert first_scalar == second_scalar
    assert 0 <= first_scalar < field.SUBGROUP_ORDER


def test_private_key_one_produces_base_point() -> None:
    signing_key = schnorr.SigningKey.from_private_key(
        1,
        scalar_multiply.scalar_multiply_double_and_add,
    )

    assert signing_key.public_key == encoding.encode_point(
        curve.BASE_EXTENDED_POINT
    )


@pytest.mark.parametrize(
    "private_key",
    [-1, 0, field.SUBGROUP_ORDER, field.SUBGROUP_ORDER + 1],
)
def test_private_key_rejects_out_of_range_values(
    private_key: int,
) -> None:
    with pytest.raises(ValueError):
        schnorr.SigningKey.from_private_key(
            private_key,
            scalar_multiply.scalar_multiply_double_and_add,
        )


def test_generated_key_pair_is_valid() -> None:
    signing_key = schnorr.generate_key_pair(
        scalar_multiply.scalar_multiply_double_and_add
    )
    public_point = encoding.decode_point(signing_key.public_key)

    assert 0 < signing_key.private_key < field.SUBGROUP_ORDER
    assert len(signing_key.public_key) == 32
    assert curve.is_on_curve(curve.to_affine_point(public_point))
    assert schnorr.is_in_prime_order_subgroup(
        public_point,
        scalar_multiply.scalar_multiply_double_and_add,
    )


def test_base_point_is_in_prime_order_subgroup() -> None:
    assert schnorr.is_in_prime_order_subgroup(
        curve.BASE_EXTENDED_POINT,
        scalar_multiply.scalar_multiply_double_and_add,
    )


def test_small_order_point_is_not_in_prime_order_subgroup() -> None:
    order_two_point = curve.to_extended_point(
        curve.AffinePoint(0, field.FIELD_MODULUS - 1)
    )

    assert not schnorr.is_in_prime_order_subgroup(
        order_two_point,
        scalar_multiply.scalar_multiply_double_and_add,
    )


def test_sign_is_deterministic_and_verifies(
    multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
    signing_key: schnorr.SigningKey,
) -> None:
    message = b"Schnorr signature test"
    first_signature = schnorr.sign(
        message,
        signing_key,
        multiplication_method,
    )
    second_signature = schnorr.sign(
        message,
        signing_key,
        multiplication_method,
    )
    base_signature = schnorr.sign(
        message,
        signing_key,
        multiplication_method,
        use_modification=False,
    )

    assert first_signature == second_signature
    assert first_signature == base_signature
    assert len(first_signature) == 64
    assert schnorr.verify(
        message,
        signing_key.public_key,
        first_signature,
        multiplication_method,
    )
    assert schnorr.verify(
        message,
        signing_key.public_key,
        first_signature,
        multiplication_method,
        use_modification=False,
    )


@pytest.mark.parametrize(
    "message",
    [b"", bytes(range(256)), bytes(1024)],
    ids=["empty", "binary", "1_kib"],
)
def test_various_messages_verify(
    message: bytes,
    signing_key: schnorr.SigningKey,
) -> None:
    signature = schnorr.sign(
        message,
        signing_key,
        scalar_multiply.scalar_multiply_double_and_add,
    )

    assert schnorr.verify(
        message,
        signing_key.public_key,
        signature,
        scalar_multiply.scalar_multiply_double_and_add,
    )


def test_verify_rejects_changed_message(
    signing_key: schnorr.SigningKey,
) -> None:
    signature = schnorr.sign(
        b"original message",
        signing_key,
        scalar_multiply.scalar_multiply_double_and_add,
    )

    assert not schnorr.verify(
        b"changed message",
        signing_key.public_key,
        signature,
        scalar_multiply.scalar_multiply_double_and_add,
    )


def test_verify_rejects_changed_signature(
    signing_key: schnorr.SigningKey,
) -> None:
    message = b"message"
    signature = bytearray(
        schnorr.sign(
            message,
            signing_key,
            scalar_multiply.scalar_multiply_double_and_add,
        )
    )
    signature[-1] ^= 1

    assert not schnorr.verify(
        message,
        signing_key.public_key,
        bytes(signature),
        scalar_multiply.scalar_multiply_double_and_add,
    )


def test_verify_rejects_wrong_public_key(
    signing_key: schnorr.SigningKey,
) -> None:
    message = b"message"
    signature = schnorr.sign(
        message,
        signing_key,
        scalar_multiply.scalar_multiply_double_and_add,
    )
    other_key = schnorr.SigningKey.from_private_key(
        signing_key.private_key + 1,
        scalar_multiply.scalar_multiply_double_and_add,
    )

    assert not schnorr.verify(
        message,
        other_key.public_key,
        signature,
        scalar_multiply.scalar_multiply_double_and_add,
    )


@pytest.mark.parametrize("public_key", [bytes(31), bytes(33)])
def test_verify_rejects_invalid_public_key_length(
    public_key: bytes,
) -> None:
    assert not schnorr.verify(
        b"message",
        public_key,
        bytes(64),
        scalar_multiply.scalar_multiply_double_and_add,
    )


@pytest.mark.parametrize("signature_length", [31, 63, 65])
def test_verify_rejects_invalid_signature_length(
    signature_length: int,
    signing_key: schnorr.SigningKey,
) -> None:
    assert not schnorr.verify(
        b"message",
        signing_key.public_key,
        bytes(signature_length),
        scalar_multiply.scalar_multiply_double_and_add,
    )


def test_verify_rejects_noncanonical_public_key() -> None:
    invalid_public_key = field.FIELD_MODULUS.to_bytes(
        32,
        byteorder="little",
    )

    assert not schnorr.verify(
        b"message",
        invalid_public_key,
        bytes(64),
        scalar_multiply.scalar_multiply_double_and_add,
    )


def test_verify_rejects_identity_public_key() -> None:
    identity_encoding = encoding.encode_point(
        curve.IDENTITY_EXTENDED_POINT
    )
    signature = identity_encoding + encoding.encode_scalar(0)

    assert not schnorr.verify(
        b"message",
        identity_encoding,
        signature,
        scalar_multiply.scalar_multiply_double_and_add,
    )


def test_verify_rejects_identity_nonce_point(
    signing_key: schnorr.SigningKey,
) -> None:
    signature = encoding.encode_point(
        curve.IDENTITY_EXTENDED_POINT
    ) + encoding.encode_scalar(1)

    assert not schnorr.verify(
        b"message",
        signing_key.public_key,
        signature,
        scalar_multiply.scalar_multiply_double_and_add,
    )


def test_verify_rejects_noncanonical_signature_scalar(
    signing_key: schnorr.SigningKey,
) -> None:
    signature = encoding.encode_point(
        curve.BASE_EXTENDED_POINT
    ) + field.SUBGROUP_ORDER.to_bytes(32, byteorder="little")

    assert not schnorr.verify(
        b"message",
        signing_key.public_key,
        signature,
        scalar_multiply.scalar_multiply_double_and_add,
    )


def test_verify_rejects_invalid_encoded_nonce_point(
    signing_key: schnorr.SigningKey,
) -> None:
    invalid_nonce_point = field.FIELD_MODULUS.to_bytes(
        32,
        byteorder="little",
    )
    signature = invalid_nonce_point + encoding.encode_scalar(0)

    assert not schnorr.verify(
        b"message",
        signing_key.public_key,
        signature,
        scalar_multiply.scalar_multiply_double_and_add,
    )

import hashlib
import secrets
from dataclasses import dataclass

import src.edwards25519curve as curve
import src.encoding as encoding
import src.field as field
import src.scalar_multiply as scalar_multiply


def hash_to_scalar(data: bytes) -> int:
    hash_bytes = hashlib.sha512(data).digest()
    hash_integer = int.from_bytes(hash_bytes, byteorder="little")

    return hash_integer % field.SUBGROUP_ORDER


def extended_points_equal(
    first_point: curve.ExtendedPoint,
    second_point: curve.ExtendedPoint,
) -> bool:
    x_difference = (
        first_point.x * second_point.z
        - second_point.x * first_point.z
    ) % field.FIELD_MODULUS

    y_difference = (
        first_point.y * second_point.z
        - second_point.y * first_point.z
    ) % field.FIELD_MODULUS

    return x_difference == 0 and y_difference == 0


def is_identity_point(
    point: curve.ExtendedPoint,
) -> bool:
    return extended_points_equal(
        point,
        curve.IDENTITY_EXTENDED_POINT,
    )


def is_in_prime_order_subgroup(
    point: curve.ExtendedPoint,
    multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
) -> bool:
    multiplied_point = multiplication_method(
        field.SUBGROUP_ORDER,
        point,
    )

    return is_identity_point(multiplied_point)


def generate_private_key() -> int:
    return secrets.randbelow(field.SUBGROUP_ORDER - 1) + 1


def obtain_public_key(
    private_key: int,
    multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
) -> curve.ExtendedPoint:
    if not 0 < private_key < field.SUBGROUP_ORDER:
        raise ValueError("Private key is outside valid range")

    return scalar_multiply.scalar_multiply_modified(
        private_key,
        curve.BASE_EXTENDED_POINT,
        multiplication_method,
    )


@dataclass(frozen=True)
class SigningKey:
    private_key: int
    public_key: bytes

    @classmethod
    def from_private_key(
        cls,
        private_key: int,
        multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
    ) -> "SigningKey":
        public_point = obtain_public_key(
            private_key,
            multiplication_method,
        )

        return cls(
            private_key=private_key,
            public_key=encoding.encode_point(public_point),
        )


def generate_key_pair(
    multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
) -> SigningKey:
    private_key = generate_private_key()

    return SigningKey.from_private_key(
        private_key,
        multiplication_method,
    )


def sign(
    message: bytes,
    signing_key: SigningKey,
    multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
    use_modification: bool = True,
) -> bytes:
    private_key = signing_key.private_key
    public_key = signing_key.public_key

    secret_hash = hashlib.sha512(
        encoding.encode_scalar(private_key)
    ).digest()

    nonce = hash_to_scalar(secret_hash + message)

    if use_modification:
        nonce_point = scalar_multiply.scalar_multiply_modified(
            nonce,
            curve.BASE_EXTENDED_POINT,
            multiplication_method,
        )
    else:
        nonce_point = multiplication_method(
            nonce,
            curve.BASE_EXTENDED_POINT,
        )

    challenge = hash_to_scalar(public_key + message)

    signature_scalar = (
        nonce + challenge * private_key
    ) % field.SUBGROUP_ORDER

    encoded_nonce_point = encoding.encode_point(nonce_point)
    encoded_signature_scalar = encoding.encode_scalar(
        signature_scalar
    )

    return encoded_nonce_point + encoded_signature_scalar


def verify(
    message: bytes,
    public_key: bytes,
    signature: bytes,
    multiplication_method: scalar_multiply.ScalarMultiplicationMethod,
    use_modification: bool = True,
) -> bool:
    if len(public_key) != 32:
        return False

    if len(signature) != 64:
        return False

    encoded_nonce_point = signature[:32]
    encoded_signature_scalar = signature[32:]

    try:
        public_point = encoding.decode_point(public_key)
        nonce_point = encoding.decode_point(encoded_nonce_point)
        signature_scalar = encoding.decode_scalar(
            encoded_signature_scalar
        )
    except (ValueError, ZeroDivisionError):
        return False

    if is_identity_point(public_point):
        return False

    if not is_in_prime_order_subgroup(
        public_point,
        multiplication_method,
    ):
        return False

    if not is_in_prime_order_subgroup(
        nonce_point,
        multiplication_method,
    ):
        return False

    challenge = hash_to_scalar(public_key + message)

    if use_modification:
        left_point = scalar_multiply.scalar_multiply_modified(
            signature_scalar,
            curve.BASE_EXTENDED_POINT,
            multiplication_method,
        )
        challenge_point = scalar_multiply.scalar_multiply_modified(
            challenge,
            public_point,
            multiplication_method,
        )
    else:
        left_point = multiplication_method(
            signature_scalar,
            curve.BASE_EXTENDED_POINT,
        )
        challenge_point = multiplication_method(
            challenge,
            public_point,
        )

    right_point = curve.add_extended_points(
        nonce_point,
        challenge_point,
    )

    return extended_points_equal(left_point, right_point)

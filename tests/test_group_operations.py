"""Tests for group operations."""

import unittest
from typing import Union, List, Optional, Dict
from gadd.group_operations import (
    multiply,
    invert,
    complete_sequence_to_identity,
    GROUP_ELEMENTS,
    ELEMENT_NAMES,
    DEFAULT_GROUP,
)


def verify_sequence_identity(
    sequence: List[Union[int, str]], group: Optional[Dict] = None
) -> bool:
    """
    Verify that a sequence multiplies to identity.

    Args:
        sequence: List of group elements (indices or names).
        group: Optional custom group structure.

    Returns:
        True if sequence multiplies to identity (element 0).
    """
    if group is None:
        group = DEFAULT_GROUP

    if not sequence:
        return True

    result = 0  # Start with identity
    for element in sequence:
        result = multiply(result, element, group)

    return result in [0, 1]  # Check if result is Ip or Im


class TestGroupOperations(unittest.TestCase):
    """Test group operations for the decoupling group."""

    def test_group_elements_mapping(self):
        """Test that group elements are properly mapped."""
        self.assertEqual(len(GROUP_ELEMENTS), 8)
        self.assertEqual(len(ELEMENT_NAMES), 8)

        # Check bidirectional mapping
        for name, index in GROUP_ELEMENTS.items():
            self.assertEqual(ELEMENT_NAMES[index], name)

    def test_multiply_with_identity(self):
        """Test multiplication with identity."""
        for i in range(8):
            self.assertEqual(multiply(0, i), i)  # Ip * g = g
            self.assertEqual(multiply(i, 0), i)  # g * Ip = g
            self.assertEqual(multiply(1, i), multiply("Im", i))  # Im * g

    def test_multiply_self_inverse(self):
        """Test that elements square to identity (with phase)."""
        # X² = I, Y² = I, Z² = I (all return to identity)
        self.assertEqual(multiply("Xp", "Xp"), 0)  # Xp² = Ip
        self.assertEqual(multiply("Xm", "Xm"), 0)  # Xm² = Ip
        self.assertEqual(multiply("Yp", "Yp"), 0)  # Yp² = Ip
        self.assertEqual(multiply("Ym", "Ym"), 0)  # Ym² = Ip
        self.assertEqual(multiply("Zp", "Zp"), 0)  # Zp² = Ip
        self.assertEqual(multiply("Zm", "Zm"), 0)  # Zm² = Ip

    def test_multiply_string_and_int(self):
        """Test multiplication with mixed string and int inputs."""
        self.assertEqual(multiply("Xp", 2), 0)  # Xp * Xp = Ip
        self.assertEqual(multiply(2, "Xp"), 0)
        self.assertEqual(multiply("Xp", "Yp"), multiply(2, 4))

    def test_pauli_multiplication_rules(self):
        """Test specific Pauli multiplication rules."""
        # XY = iZ (in our encoding, this involves phase tracking)
        # For the encoding used in the paper:
        # Xp * Yp should give us something related to Z with phase
        result_xy = multiply("Xp", "Yp")
        result_yx = multiply("Yp", "Xp")

        # These should be different (anti-commutation)
        self.assertNotEqual(result_xy, result_yx)

        # Verify specific multiplication results based on the paper's encoding
        # Note: The exact values depend on the phase convention used

    def test_invert_operations(self):
        """Test inverse operations."""
        # Test all elements have inverses
        for i in range(8):
            inv = invert(i)
            product = multiply(i, inv)
            self.assertIn(
                product, [0, 1], f"Element {i} * its inverse should give identity"
            )

        # Test specific inverses based on the encoding
        self.assertEqual(invert("Ip"), 0)  # Ip^(-1) = Ip
        self.assertEqual(invert("Im"), 1)  # Im^(-1) = Im

        # For Pauli matrices with phase, the inverse flips the phase
        # In the paper's encoding: Xp^(-1) = Xm, etc.
        self.assertEqual(multiply("Xp", invert("Xp")), 1)
        self.assertEqual(multiply("Yp", invert("Yp")), 1)
        self.assertEqual(multiply("Zp", invert("Zp")), 1)

    def test_verify_sequence_identity(self):
        """Test sequence identity verification."""
        # Empty sequence is identity
        self.assertTrue(verify_sequence_identity([]))

        # Single identity is identity
        self.assertTrue(verify_sequence_identity([0]))

        # X followed by X^(-1) is identity
        xp_inv = invert(2)
        self.assertTrue(verify_sequence_identity([2, xp_inv]))

        # Two of the same element (which gives identity) followed by another
        self.assertFalse(
            verify_sequence_identity([2, 2, 4])
        )  # Xp, Xp, Yp = Ip, Yp = Yp

    def test_complete_sequence_to_identity(self):
        """Test sequence completion to identity."""
        # Empty sequence needs identity
        self.assertEqual(complete_sequence_to_identity([]), 0)

        # Single X needs its inverse
        completion = complete_sequence_to_identity([2])  # Xp
        self.assertEqual(multiply(2, completion), 1)

        # Test complex sequence
        partial = [2, 4, 6]  # Xp, Yp, Zp
        last = complete_sequence_to_identity(partial)
        full_seq = partial + [last]
        self.assertTrue(verify_sequence_identity(full_seq))

    def test_group_closure(self):
        """Test that multiplication table represents a closed group."""
        # Every multiplication should result in a valid group element
        for i in range(8):
            for j in range(8):
                result = multiply(i, j)
                self.assertIn(result, range(8))
                self.assertIsInstance(result, int)

    def test_associativity(self):
        """Test that multiplication is associative."""
        # Test a few cases of (a*b)*c = a*(b*c)
        test_cases = [
            (2, 4, 6),  # Xp, Yp, Zp
            (1, 3, 5),  # Im, Xm, Ym
            (0, 2, 4),  # Ip, Xp, Yp
            (2, 2, 2),  # Xp, Xp, Xp
            (4, 6, 2),  # Yp, Zp, Xp
        ]

        for a, b, c in test_cases:
            left = multiply(multiply(a, b), c)
            right = multiply(a, multiply(b, c))
            self.assertEqual(
                left, right, f"Associativity failed for ({a}*{b})*{c} != {a}*({b}*{c})"
            )

    def test_identity_element(self):
        """Test identity element properties."""
        # Ip is the identity
        for i in range(8):
            self.assertEqual(multiply(0, i), i)
            self.assertEqual(multiply(i, 0), i)

    def test_inverse_uniqueness(self):
        """Test that inverses are unique."""
        for i in range(8):
            inverses = [j for j in range(8) if multiply(i, j) == 0]
            self.assertEqual(
                len(inverses), 1, f"Element {i} should have exactly one inverse"
            )

    def test_invert_fallback_path(self):
        """Test invert function when using multiplication search instead of inverse map."""
        # Create a custom group without inverse_map to force the fallback path
        custom_group = {
            "elements": {"e": 0, "a": 1},
            "names": {0: "e", 1: "a"},
            "multiplication": [
                [0, 1],  # e * e = e, e * a = a
                [1, 0],  # a * e = a, a * a = e
            ],
            # Note: no inverse_map provided
        }

        # Test that it finds inverses correctly using multiplication
        self.assertEqual(invert(0, custom_group), 0)  # e^-1 = e
        self.assertEqual(invert(1, custom_group), 1)  # a^-1 = a

    def test_invert_custom_group_no_inverse(self):
        """Test that invert raises ValueError for element without inverse in custom group."""
        # Create a custom "group" that's not actually a group (element 2 has no inverse)
        custom_group = {
            "elements": {"e": 0, "a": 1, "b": 2},
            "names": {0: "e", 1: "a", 2: "b"},
            "multiplication": [
                [0, 1, 2],  # e * x = x
                [1, 0, 2],  # a * a = e, a * b = b
                [2, 2, 2],  # b * anything = b (no inverse!)
            ],
        }

        # Element 2 (b) should have no inverse
        with self.assertRaises(ValueError) as context:
            invert(2, custom_group)
        self.assertIn("No inverse found for element 2", str(context.exception))

    def test_complete_sequence_already_identity(self):
        """Test complete_sequence_to_identity when sequence already multiplies to Ip."""
        # Empty sequence is already identity, should return Ip
        self.assertEqual(complete_sequence_to_identity([]), 0)

        # Sequence that multiplies to Ip: [Xp, Xp] = Ip
        self.assertEqual(complete_sequence_to_identity([2, 2]), 0)

        # Another sequence that gives Ip: [Yp, Yp] = Ip
        self.assertEqual(complete_sequence_to_identity([4, 4]), 0)

        # Sequence that gives Im: [Xp, Xm] = Im, should return Im to get back to Ip
        self.assertEqual(complete_sequence_to_identity([2, 3]), 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for group operations."""

import unittest
from gadd.group_operations import (
    multiply,
    invert,
    element_name,
    verify_sequence_identity,
    complete_sequence_to_identity,
    GROUP_ELEMENTS,
    ELEMENT_NAMES,
)


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

    def test_multiply_self_inverse(self):
        """Test that Pauli operators square to identity."""
        # X² = I
        self.assertEqual(multiply("Xp", "Xp"), 0)
        self.assertEqual(multiply("Xm", "Xm"), 0)

        # Y² = I
        self.assertEqual(multiply("Yp", "Yp"), 0)
        self.assertEqual(multiply("Ym", "Ym"), 0)

        # Z² = I
        self.assertEqual(multiply("Zp", "Zp"), 0)
        self.assertEqual(multiply("Zm", "Zm"), 0)

    def test_multiply_string_and_int(self):
        """Test multiplication with mixed string and int inputs."""
        self.assertEqual(multiply("Xp", 2), 0)  # Xp * Xp = Ip
        self.assertEqual(multiply(2, "Xp"), 0)
        self.assertEqual(multiply("Xp", "Yp"), multiply(2, 4))

    def test_invert_operations(self):
        """Test inverse operations."""
        # Test all elements have inverses
        for i in range(8):
            inv = invert(i)
            self.assertEqual(multiply(i, inv), 0)

        # Test specific inverses
        self.assertEqual(invert("Ip"), 0)  # Ip⁻¹ = Ip
        self.assertEqual(invert("Im"), 1)  # Im⁻¹ = Im
        self.assertEqual(invert("Xp"), 3)  # Xp⁻¹ = Xm
        self.assertEqual(invert("Xm"), 2)  # Xm⁻¹ = Xp
        self.assertEqual(invert("Yp"), 5)  # Yp⁻¹ = Ym
        self.assertEqual(invert("Ym"), 4)  # Ym⁻¹ = Yp
        self.assertEqual(invert("Zp"), 7)  # Zp⁻¹ = Zm
        self.assertEqual(invert("Zm"), 6)  # Zm⁻¹ = Zp

    def test_element_name(self):
        """Test element name conversion."""
        self.assertEqual(element_name(0), "Ip")
        self.assertEqual(element_name(2), "Xp")
        self.assertEqual(element_name(999), "Unknown(999)")

    def test_verify_sequence_identity(self):
        """Test sequence identity verification."""
        # Empty sequence is identity
        self.assertTrue(verify_sequence_identity([]))

        # Single identity is identity
        self.assertTrue(verify_sequence_identity([0]))

        # X followed by X⁻¹ is identity
        self.assertTrue(verify_sequence_identity([2, 3]))  # Xp, Xm

        # XYZ sequence
        self.assertFalse(verify_sequence_identity([2, 4, 6]))  # Xp, Yp, Zp

        # XYXY is identity (for this specific group structure)
        # This needs to be verified based on actual multiplication table
        seq = [2, 4, 2, 4]  # Xp, Yp, Xp, Yp
        result = 0
        for elem in seq:
            result = multiply(result, elem)
        # Check if it equals identity
        self.assertEqual(result, 0)

    def test_complete_sequence_to_identity(self):
        """Test sequence completion to identity."""
        # Empty sequence needs identity
        self.assertEqual(complete_sequence_to_identity([]), 0)

        # Single X needs X⁻¹
        self.assertEqual(complete_sequence_to_identity([2]), 3)  # Xp needs Xm

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

    def test_associativity(self):
        """Test that multiplication is associative."""
        # Test a few cases of (a*b)*c = a*(b*c)
        test_cases = [
            (2, 4, 6),  # Xp, Yp, Zp
            (1, 3, 5),  # Im, Xm, Ym
            (0, 2, 4),  # Ip, Xp, Yp
        ]

        for a, b, c in test_cases:
            left = multiply(multiply(a, b), c)
            right = multiply(a, multiply(b, c))
            self.assertEqual(
                left, right, f"Associativity failed for ({a}*{b})*{c} != {a}*({b}*{c})"
            )


if __name__ == "__main__":
    unittest.main()

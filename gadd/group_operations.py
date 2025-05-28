"""
Group operations for the decoupling group G used in GADD.

The group G = {Ip, Im, Xp, Xm, Yp, Ym, Zp, Zm} represents
the single-qubit Pauli operations and their inverses.
"""

from typing import Union, List

# Group elements mapping
GROUP_ELEMENTS = {
    "Ip": 0,  # Identity (+phase)
    "Im": 1,  # Identity (-phase)
    "Xp": 2,  # X rotation (+π)
    "Xm": 3,  # X rotation (-π)
    "Yp": 4,  # Y rotation (+π)
    "Ym": 5,  # Y rotation (-π)
    "Zp": 6,  # Z rotation (+π)
    "Zm": 7,  # Z rotation (-π)
}

# Inverse mapping
ELEMENT_NAMES = {v: k for k, v in GROUP_ELEMENTS.items()}

# Multiplication table for the group
# This is based on Pauli matrix multiplication rules:
# I² = I, X² = I, Y² = I, Z² = I
# XY = iZ, YZ = iX, ZX = iY
# YX = -iZ, ZY = -iX, XZ = -iY
MULTIPLICATION_TABLE = [
    # Ip  Im  Xp  Xm  Yp  Ym  Zp  Zm
    [0, 1, 2, 3, 4, 5, 6, 7],  # Ip
    [1, 0, 3, 2, 5, 4, 7, 6],  # Im
    [2, 3, 0, 1, 7, 6, 5, 4],  # Xp
    [3, 2, 1, 0, 6, 7, 4, 5],  # Xm
    [4, 5, 6, 7, 0, 1, 3, 2],  # Yp
    [5, 4, 7, 6, 1, 0, 2, 3],  # Ym
    [6, 7, 5, 4, 2, 3, 0, 1],  # Zp
    [7, 6, 4, 5, 3, 2, 1, 0],  # Zm
]


def multiply(p1: Union[int, str], p2: Union[int, str]) -> int:
    """
    Multiply two group elements.

    Group elements: 0,1: I, 2: X, 3: Xb, 4: Y, 5: Yb, 6: Z, 7: Zb
    where Xb = -X, Yb = -Y, Zb = -Z (with phase)

    Args:
        p1: First element (index or name).
        p2: Second element (index or name).

    Returns:
        Index of the resulting element.
    """
    # Convert string names to indices if needed
    if isinstance(p1, str):
        p1 = GROUP_ELEMENTS[p1]
    if isinstance(p2, str):
        p2 = GROUP_ELEMENTS[p2]

    # Convert input to a better representation for multiplying: get n -> (a,b) where n = i^b*sigma_a
    n_to_ps = {
        0: (0, 0),
        1: (0, 2),
        2: (1, 0),
        3: (1, 2),
        4: (2, 0),
        5: (2, 2),
        6: (3, 0),
        7: (3, 2),
    }

    p1_rep = n_to_ps[p1]
    p2_rep = n_to_ps[p2]

    # Multiplication rules
    sign = p1_rep[1] + p2_rep[1]

    # Identity element
    if p1_rep[0] == 0:
        pauli = p2_rep[0]
    elif p2_rep[0] == 0:
        pauli = p1_rep[0]
    elif p1_rep[0] == p2_rep[0]:
        pauli = 0
    else:
        # Non identity elements
        pauli = -(p1_rep[0] + p2_rep[0]) % 3 if -(p1_rep[0] + p2_rep[0]) % 3 > 0 else 3
        if (p2_rep[0] - p1_rep[0]) % 3 == 2:
            # anti multiplication of pauli matrices
            sign += 2

    ps_to_n = {}
    for key in n_to_ps:
        ps_to_n[n_to_ps[key]] = key

    return ps_to_n[(pauli, sign % 4)]


def invert(a: Union[int, str]) -> int:
    """
    Find the inverse of a group element.

    Args:
        a: Element to invert (index or name)

    Returns:
        Index of the inverse element
    """
    if isinstance(a, str):
        a = GROUP_ELEMENTS[a]

    # Find element b such that a * b = 0 (identity Ip)
    for b in range(8):
        if multiply(a, b) == 0:
            return b

    # This should never happen with a valid group
    raise ValueError(f"No inverse found for element {a}")


def element_name(index: int) -> str:
    """Convert element index to name."""
    return ELEMENT_NAMES.get(index, f"Unknown({index})")


def verify_sequence_identity(sequence: List[Union[int, str]]) -> bool:
    """
    Verify that a sequence multiplies to identity.

    Args:
        sequence: List of group elements (indices or names)

    Returns:
        True if sequence multiplies to identity (Ip)
    """
    if not sequence:
        return True

    result = 0  # Start with identity
    for element in sequence:
        result = multiply(result, element)

    return result == 0  # Check if result is Ip


def complete_sequence_to_identity(partial_sequence: List[Union[int, str]]) -> int:
    """
    Find the element needed to complete a sequence to multiply to identity.

    Args:
        partial_sequence: Incomplete sequence of group elements

    Returns:
        Index of element needed to complete sequence to identity
    """
    if not partial_sequence:
        return 0  # Return identity

    # Multiply all elements in sequence
    result = 0  # Start with identity
    for element in partial_sequence:
        result = multiply(result, element)

    # Return inverse of result
    return invert(result)

"""
Group operations for the decoupling group.
"""

from typing import Union, List, Optional, Dict

# The default group group G = {Ip, Im, Xp, Xm, Yp, Ym, Zp, Zm} of
# the single-qubit Pauli operations and their inverses as defined
# in the paper.
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

# Default group structure
DEFAULT_GROUP = {
    "elements": GROUP_ELEMENTS,
    "names": ELEMENT_NAMES,
    "multiplication": MULTIPLICATION_TABLE,
    "inverse_map": {
        0: 0,  # Ip^-1 = Ip
        1: 1,  # Im^-1 = Im
        2: 3,  # Xp^-1 = Xm
        3: 2,  # Xm^-1 = Xp
        4: 5,  # Yp^-1 = Ym
        5: 4,  # Ym^-1 = Yp
        6: 7,  # Zp^-1 = Zm
        7: 6,  # Zm^-1 = Zp
    },
}


def multiply(
    p1: Union[int, str], p2: Union[int, str], group: Optional[Dict] = None
) -> int:
    """
    Multiply two group elements.

    Args:
        p1: First element (index or name).
        p2: Second element (index or name).
        group: Optional custom group structure. Defaults to the
            G = {Ip, Im, Xp, Xm, Yp, Ym, Zp, Zm} used in the paper.

    Returns:
        Index of the resulting element.
    """
    # Use provided group or default
    if group is None:
        group = DEFAULT_GROUP

    # Convert string names to indices
    elements = group["elements"]
    if isinstance(p1, str):
        p1 = elements.get(p1, p1)
    if isinstance(p2, str):
        p2 = elements.get(p2, p2)

    # Use multiplication table
    return group["multiplication"][p1][p2]


def invert(a: Union[int, str], group: Optional[Dict] = None) -> int:
    """
    Find the inverse of a group element.

    Args:
        a: Element to invert (index or name).
        group: Optional custom group structure. Defaults to the
            G = {Ip, Im, Xp, Xm, Yp, Ym, Zp, Zm} used in the paper.

    Returns:
        Index of the inverse element.
    """
    # Use provided group or default
    if group is None:
        group = DEFAULT_GROUP

    # Convert string to index
    elements = group["elements"]
    if isinstance(a, str):
        a = elements.get(a, a)

    # Check if group has inverse map
    if "inverse_map" in group:
        return group["inverse_map"][a]

    # Otherwise search for inverse
    group_size = len(group["elements"])
    for b in range(group_size):
        if multiply(a, b, group) == 0:
            return b

    raise ValueError(f"No inverse found for element {a}")


def complete_sequence_to_identity(
    partial_sequence: List[Union[int, str]], group: Optional[Dict] = None
) -> int:
    """
    Find the element needed to complete a sequence to multiply to identity.

    Args:
        partial_sequence: Incomplete sequence of group elements.
        group: Optional custom group structure.

    Returns:
        Index of element needed to complete sequence to identity.
    """
    if group is None:
        group = DEFAULT_GROUP

    if not partial_sequence:
        return 0  # Return identity

    # Multiply all elements in sequence
    result = 0  # Start with identity
    for element in partial_sequence:
        result = multiply(result, element, group)

    # We need to find an element that when multiplied with result gives Ip (0)
    # This is not necessarily the inverse if result is already a form of identity
    if result == 0:  # Already Ip
        return 0
    if result == 1:  # Result is Im
        return 1
    # For non-identity results, find the inverse
    return invert(result, group)

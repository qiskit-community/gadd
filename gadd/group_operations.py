"""
Decoupling groups and group operations.
"""

from typing import Union, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class DecouplingGroup:
    """Mathematical structure defining a decoupling group for DD sequences.

    This class represents the algebraic structure used in the GADD paper for
    constructing dynamical decoupling sequences. It encodes the group elements,
    their names, multiplication table, and inverse relationships according to
    the Pauli group algebra extended with phase information.

    The default group used in the paper is G = {Ip, Im, Xp, Xm, Yp, Ym, Zp, Zm}
    where the subscripts p/m indicate positive/negative phase rotations on the
    Bloch sphere. This encoding allows for systematic exploration of DD sequences
    that maintain the group constraint of multiplying to the identity element.

    Attributes:
        elements: Mapping from element names to integer indices.
        names: Reverse mapping from indices to element names.
        multiplication: 2D list encoding the group multiplication table where
            multiplication[i][j] gives the result of multiplying element i with element j.
        inverse_map: Optional mapping from elements to their group inverses.

    Example:
        >>> group = DEFAULT_GROUP
        >>> multiply("Xp", "Yp", group)  # Returns index for result
        >>> group.element_name(2)  # Returns "Xp"
    """

    elements: Dict[str, int]
    names: Dict[int, str]
    multiplication: List[List[int]]
    inverse_map: Optional[Dict[int, int]]

    @property
    def size(self) -> int:
        """Size of the group."""
        return len(self.elements)

    def element_name(self, index: int) -> str:
        """Get element name from index."""
        return self.names.get(index, f"g{index}")

    def element_index(self, name: str) -> int:
        """Get element index from name."""
        if name in self.elements:
            return self.elements[name]
        raise ValueError(f"Unknown element name: {name}")


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
DEFAULT_GROUP = DecouplingGroup(
    elements=GROUP_ELEMENTS,
    names=ELEMENT_NAMES,
    multiplication=MULTIPLICATION_TABLE,
    inverse_map={
        0: 0,  # Ip^-1 = Ip
        1: 1,  # Im^-1 = Im
        2: 3,  # Xp^-1 = Xm
        3: 2,  # Xm^-1 = Xp
        4: 5,  # Yp^-1 = Ym
        5: 4,  # Ym^-1 = Yp
        6: 7,  # Zp^-1 = Zm
        7: 6,  # Zm^-1 = Zp
    },
)


def multiply(
    p1: Union[int, str], p2: Union[int, str], group: Optional[DecouplingGroup] = None
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
    if isinstance(p1, str):
        p1 = group.element_index(p1)
    if isinstance(p2, str):
        p2 = group.element_index(p2)

    # Use multiplication table
    return group.multiplication[p1][p2]


def invert(a: Union[int, str], group: Optional[DecouplingGroup] = None) -> int:
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
    if isinstance(a, str):
        a = group.element_index(a)

    # Use inverse map
    if a in group.inverse_map:
        return group.inverse_map[a]

    # Otherwise search for inverse (shouldn't happen for valid groups)
    for b in range(group.size):
        if multiply(a, b, group) == 0:
            return b

    raise ValueError(f"No inverse found for element {a}")


def complete_sequence_to_identity(
    partial_sequence: List[Union[int, str]], group: Optional[DecouplingGroup] = None
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

    # Return inverse of result
    return invert(result, group)

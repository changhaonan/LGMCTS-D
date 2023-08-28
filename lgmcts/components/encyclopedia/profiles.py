from enum import Enum


class ProfilePedia(Enum):
    """
    An encyclopedia of profiles in VIMA world.

    Profile entries could be further added.
    """

    UNDETERMINED = -1
    SQUARE_LIKE = 0
    CIRCLE_LIKE = 1
    RECTANGLE_LIKE = 2
    TRIANGLE_LIKE = 3
    STAR_LIKE = 4
    CYLINDER_LIKE = 5

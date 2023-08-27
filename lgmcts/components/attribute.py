"""Define a series of different attribute judger"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from lgmcts.components.encyclopedia import ObjEntry, TextureEntry, SizeRange


## Comparision definition

class CompareRel:
    """Compare relationship for attributes"""
    words: list[str] = []


class EqualRel(CompareRel):
    """Equal relationship"""
    words = ["is equal to", "is same as", "is identical to", "is alike", "is equivalent to"]

    def __eq__(self, other):
        return isinstance(other, EqualRel)


class DifferentRel(CompareRel):
    """Different relationship"""
    words = ["is different with", "is not the same as", "is not equal to", "is not identical to", "is not alike", "is not equivalent to", "is different from"]

    def __eq__(self, other):
        """smaller, bigger is different, and verse versa"""
        return not isinstance(other, EqualRel)


class SmallerRel(CompareRel):
    """Smaller relationship"""
    words = ["is smaller than", "is less than", "is shorter than", "is narrower than", "is smaller in size than"]

    def __eq__(self, other):
        return isinstance(other, SmallerRel) or isinstance(other, DifferentRel)


class BiggerRel(CompareRel):
    """Bigger relationship"""
    words = ["is bigger than", "is larger than", "is greater than", "is taller than", "is wider than", "is bigger in size than"]

    def __eq__(self, other):
        return isinstance(other, BiggerRel) or isinstance(other, DifferentRel)


@dataclass
class ObjectBag:
    """Bag of attributes for objects"""
    obj_id: int
    obj_name: str
    obj_size: list[float]
    color_name: str
    color_value: list[float]


def get_object_bag(obj_id, obj_entry: ObjEntry, texture_entry: TextureEntry) -> ObjectBag:
    """Get object bag"""
    obj_size = [obj_entry.size_range.high[0] - obj_entry.size_range.low[0],
                obj_entry.size_range.high[1] - obj_entry.size_range.low[1], 
                obj_entry.size_range.high[2] - obj_entry.size_range.low[2]]
    return ObjectBag(obj_id, obj_entry.name, obj_size, texture_entry.name, texture_entry.color_value)


def compare_color(obj_1: ObjectBag, obj_2: ObjectBag) -> CompareRel:
    """Compare color"""
    if obj_1.color_name == obj_2.color_name:
        return EqualRel()
    else:
        return DifferentRel()


def compare_size(obj_1: ObjectBag, obj_2: ObjectBag) -> CompareRel:
    """Compare 3d size"""
    volume_1 = obj_1.obj_size[0] * obj_1.obj_size[1] * obj_1.obj_size[2]
    volume_2 = obj_2.obj_size[0] * obj_2.obj_size[1] * obj_2.obj_size[2]
    if volume_1 == volume_2:
        return EqualRel()
    elif volume_1 < volume_2:
        return SmallerRel()
    else:
        return BiggerRel()


def compare_name(obj_1: ObjectBag, obj_2: ObjectBag) -> CompareRel:
    """Compare name"""
    if obj_1.obj_name == obj_2.obj_name:
        return EqualRel()
    else:
        return DifferentRel()


COMPARE_DICT = {
    "color": compare_color,
    # "size": compare_size,
    # "name": compare_name
}
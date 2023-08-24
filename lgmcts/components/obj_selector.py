"""Object selector"""
from __future__ import annotations
from lgmcts.components.encyclopedia import ObjEntry, TextureEntry, SizeRange
from lgmcts.components.attribute import COMPARE_DICT, CompareRel, EqualRel, DifferentRel, SmallerRel, BiggerRel
from lgmcts.components.attribute import ObjectBag, get_object_bag


class ObjectSelector:
    """Obj selector"""
    def __init__(self,  rng):
        self.rng = rng
        self.obj_bag_list = []

    def reset(self):
        """Reset"""
        self.obj_bag_list = []   
    
    def set_objs(self, obj_id_list, obj_list: list[ObjEntry], texture_list: list[TextureEntry]):
        """Set objects"""
        assert len(obj_list) == len(texture_list)
        for obj_id, obj_entry, texture_entry in zip(obj_id_list, obj_list, texture_list):
            self.obj_bag_list.append(get_object_bag(obj_id, obj_entry, texture_entry))

    def select_obj(self, anchor_obj_bag: ObjectBag, attribute: str, compare_rel: CompareRel):
        """Select object based on attribute
        """
        assert attribute in COMPARE_DICT, f"Attribute {attribute} not supported"
        selected_list = []
        for obj_bag in self.obj_bag_list:
            if compare_rel == COMPARE_DICT[attribute](obj_bag, anchor_obj_bag):
                selected_list.append(obj_bag.obj_id)
        return selected_list

    def gen_anchor_obj_prompt(self):
        """Based on the obj we have, generate a valid anchor obj prompt"""
        anchor_obj_bag = self.rng.choice(self.obj_bag_list)
        anchor_obj = anchor_obj_bag.obj_name
        attribute = self.rng.choice(COMPARE_DICT.keys())
        compare_rel = self.rng.choice([EqualRel(), DifferentRel(), SmallerRel(), BiggerRel()])
        compare_rel_str = self.rng.choice(compare_rel.words)
        prompt_str = f"Objects {compare_rel_str} {attribute} {anchor_obj}"
        return anchor_obj_bag, attribute, compare_rel, prompt_str

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
        self.obj_list = []
        self.texture_list = []
        self.size_list = []  #FIXME: to implement it

    def reset(self):
        """Reset"""
        self.obj_bag_list = []
        self.obj_list = []
        self.texture_list = []
        self.size_list = []
    
    def set_objs(self, obj_list: list[ObjEntry], texture_list: list[TextureEntry]):
        """Set objects"""
        #TODO: add size and other pre-selection
        self.obj_list = obj_list
        self.texture_list = texture_list
        assert len(obj_list) == len(texture_list), "Object and texture list should have the same length"
        for id, (obj_entry, texture_entry) in enumerate(zip(obj_list, texture_list)):
            self.obj_bag_list.append(get_object_bag(id, obj_entry.value, texture_entry.value))

    def select_obj(self, anchor_obj_bag: ObjectBag, attribute: str, compare_rel: CompareRel):
        """Select object based on attribute
        """
        assert attribute in COMPARE_DICT, f"Attribute {attribute} not supported"
        in_obj, in_color, in_size, out_obj, out_color, out_size = [], [], [], [], [], []
        # check self-include
        self_include = False
        if compare_rel == EqualRel():
            in_obj.append(self.obj_list[anchor_obj_bag.obj_id])
            in_color.append(self.texture_list[anchor_obj_bag.obj_id])
            # selected_size.append(self.size_list[anchor_obj_bag.size_id])
            self_include = True
        for obj_bag in self.obj_bag_list:
            if compare_rel == COMPARE_DICT[attribute](obj_bag, anchor_obj_bag):
                if anchor_obj_bag.obj_id == obj_bag.obj_id:
                    continue
                else:
                    in_obj.append(self.obj_list[obj_bag.obj_id])
                    in_color.append(self.texture_list[obj_bag.obj_id])
                # selected_size.append(self.size_list[obj_bag.size_id])
            else:
                if obj_bag.obj_id == anchor_obj_bag.obj_id:
                    continue  # jump anchor for outlist
                out_obj.append(self.obj_list[obj_bag.obj_id])
                out_color.append(self.texture_list[obj_bag.obj_id])
                # out_size.append(self.size_list[obj_bag.size_id])
        return self_include, in_obj, in_color, in_size, out_obj, out_color, out_size

    def gen_anchor_obj_prompt(self):
        """Based on the obj we have, generate a valid anchor obj prompt"""
        ## random select anchor
        max_try = 3
        for i in range(max_try):
            anchor_obj_bag = self.rng.choice(self.obj_bag_list)
            anchor_obj = anchor_obj_bag.obj_name
            attribute = self.rng.choice(list(COMPARE_DICT.keys()))
            if attribute == "color":
                compare_rel = self.rng.choice([EqualRel(), DifferentRel()]) if i < max_try - 1 else EqualRel()
            elif attribute == "size":
                compare_rel = self.rng.choice([EqualRel(), DifferentRel(), SmallerRel(), BiggerRel()]) if i < max_try - 1 else EqualRel()
            else:
                raise ValueError("Attribute not supported")
            compare_rel_str = self.rng.choice(compare_rel.words)
            prompt_str = f"objects whose {attribute} {compare_rel_str} {anchor_obj}"

            ## select objects
            self_include, in_obj, in_color, in_size, out_obj, out_color, out_size = self.select_obj(anchor_obj_bag, attribute, compare_rel)
            if len(in_obj) >= 3:  # at least 3 objects to formulate a pattern
                if not self_include:
                    anchor_obj = self.obj_list[anchor_obj_bag.obj_id]
                    anchor_color = self.texture_list[anchor_obj_bag.obj_id]
                    anchor_size = None
                else:
                    anchor_obj = None
                    anchor_color = None
                    anchor_size = None
                return {
                    "prompt_str": prompt_str,
                    "anchor_obj": anchor_obj,
                    "anchor_color": anchor_color,
                    "anchor_size": anchor_size,
                    "in_obj": in_obj,
                    "in_color": in_color,
                    "in_size": in_size,
                    "out_obj": out_obj,
                    "out_color": out_color,
                    "out_size": out_size,
                }
        raise ValueError("Cannot generate a valid prompt")

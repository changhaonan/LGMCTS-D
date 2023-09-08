"""Object selector"""
from __future__ import annotations
import warnings
from lgmcts.components.encyclopedia import ObjEntry, TextureEntry, SizeRange
from lgmcts.components.attribute import COMPARE_DICT, CompareRel, EqualRel, DifferentRel, SmallerRel, BiggerRel
from lgmcts.components.attribute import ObjectBag, get_object_bag
import os 
import numpy as np
import pickle 

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
        ##FIXME: Rewrite the logic for anchor selection
        max_try = 10
        for i in range(max_try):
            anchor_obj_bag = self.rng.choice(self.obj_bag_list)
            anchor_obj = anchor_obj_bag.obj_name
            attribute = self.rng.choice(list(COMPARE_DICT.keys()))
            if attribute == "color":
                # compare_rel = self.rng.choice([EqualRel(), DifferentRel()]) if i < max_try - 1 else EqualRel()
                compare_rel = EqualRel() #To accomodate queries for StructFormer
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
        # warnings.warn("Cannot generate a valid prompt")
        assert False, "Cannot generate a valid prompt"
        return {"anchor_obj": None, "in_obj": [], "in_color": [], "in_size": [], "out_obj": [], "out_color": [], "out_size": []}
    
    def parse_llm_result(self, dataset_path: str, llm_result: str, check_point_list: list[str], num_objs: int):
        """Parse the result from LLM"""
            # generate prompt
        prompt_folder = os.path.join(os.path.dirname(dataset_path), "prompt")
        if not os.path.exists(prompt_folder):
            os.makedirs(prompt_folder)

        goals = []
        for ind, res in enumerate(llm_result):
            env_state = None
            with open(os.path.join(dataset_path, check_point_list[ind]), "rb") as f:
                env_state = pickle.load(f)
            obj_list = [None]* num_objs
            texture_list = [None]* num_objs
            for entry in env_state["obj_id_reverse_mapping"]:
                obj_list[entry] = env_state["obj_id_reverse_mapping"][entry]["obj_name"]
                texture_list[entry] = env_state["obj_id_reverse_mapping"][entry]["texture_name"]
            goal = []
            for entry in res:
                goal_entry = dict()
                if entry["pattern"] != "spatial":
                    goal_entry["type"] = f"pattern:{entry['pattern']}"
                    goal_entry["obj_ids"] = []
                    anchor_ind = obj_list.index(entry["anchor"])
                    goal_entry["anchor_id"] = anchor_ind
                    if entry["anchor_relation"] == "same":
                        for obj, color in zip(obj_list, texture_list):
                            if color == texture_list[anchor_ind]:
                                goal_entry["obj_ids"].append(obj_list.index(obj))
                                
                    else:
                        for obj, color in zip(obj_list, texture_list):
                            if color != texture_list[anchor_ind]:
                                goal_entry["obj_ids"].append(obj_list.index(obj))
                    goal.append(goal_entry)
                else:
                    goal_entry["type"] = f"pattern:{entry['pattern']}"
                    goal_entry["obj_ids"] = []
                    # for obj_name, obj_color in zip(entry["objects"], entry["colors"]):
                    #     for obj, color, k in zip(obj_list, texture_list, range(len(obj_list))):
                    #         if obj == obj_name and color == obj_color:
                    #             goal_entry["obj_ids"].append(k)
                    for obj_name in entry["objects"]:
                        if obj_name in obj_list:
                            goal_entry["obj_ids"].append(obj_list.index(obj_name))
                    goal_entry["obj_ids"] = goal_entry["obj_ids"][::-1]
                    goal_entry["spatial_label"] = np.array([0, 0, 0, 0], dtype=np.int32)
                    if "left" in entry["spatial_label"]:
                        goal_entry["spatial_label"][0] = 1
                    if "right" in entry["spatial_label"]:
                        goal_entry["spatial_label"][1] = 1
                    if "front" in entry["spatial_label"]:
                        goal_entry["spatial_label"][2] = 1
                    if "behind" in entry["spatial_label"]:
                        goal_entry["spatial_label"][3] = 1
                    goal_entry["spatial_str"] = entry["spatial_str"]
                    goal.append(goal_entry)
            goals.append(goal)           
        with open(f"{dataset_path}/goal.pkl", "wb") as fp:
            pickle.dump(goals, fp)


        # prompt_bg = "Assume you are a language-based motion planner. You will parse user's requirement into goal configuration and constraints. Follow the examples we provide. You should strictly adhere to our format. \n"
        # obj_id_list = list(range(len(self.obj_list)))
        # obj_name_list = []
        # obj_color_list = []
        # for obj_id in obj_id_list:
        #     # obj_name_list.append(entry["obj_name"] for entry in env.obj_id_reverse_mapping[obj_id])
        #     obj_name_list.append(self.obj_list[obj_id].name.lower().replace("shapenet_", ""))
        #     obj_color_list.append(self.color_list[obj_id].name.lower().replace("_", " "))
        # prompt_bg += f"Object_id of the objects in the scene are: {obj_id_list} for {obj_name_list}\n"
        # prompt_bg += f"And correspondingly colors of the objects in the scene are:  {obj_color_list}\n"
        # with open(f"{prompt_folder}/prompt_bg.txt", "w") as f:
        #     f.write(prompt_bg)

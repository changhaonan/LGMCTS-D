"""Parse the semantic goal into id based"""


class SemanticPattern:

    @classmethod
    def parse_goal(cls, name_ids):
        """Parse the goal of dinner into id based"""
        raise NotImplementedError


class DinnerPattern(SemanticPattern):
    @classmethod
    def parse_goal(cls, name_ids):
        """Parse the goal of dinner into id based"""
        # Step 1: get a name id dict first
        name_id_dict = {}
        for name_id in name_ids:
            name = name_id[0].split("_")[0].lower()
            if name not in name_id_dict:
                name_id_dict[name] = []
            name_id_dict[name].append(name_id[1])
        # Step 2: parse the goal
        goals = []

        # plates should be stacked
        if "plate" in name_id_dict:
            goals.append({"type": "pattern:tower", "obj_ids": name_id_dict["plate"]})

        # forks should be on the left of plates
        if "fork" in name_id_dict and "plate" in name_id_dict:
            goals.append({"type": "pattern:spatial", "obj_ids": [name_id_dict["fork"][0], name_id_dict["plate"][0]], "spatial_label": [1, 0, 0, 0]})
        # knives should be on the right of plates
        if "knife" in name_id_dict and "plate" in name_id_dict:
            goals.append({"type": "pattern:spatial", "obj_ids": [name_id_dict["knife"][0], name_id_dict["plate"][0]], "spatial_label": [0, 1, 0, 0]})
        # spoons should be on the right of plates
        if "spoon" in name_id_dict and "plate" in name_id_dict:
            goals.append({"type": "pattern:spatial", "obj_ids": [name_id_dict["spoon"][0], name_id_dict["plate"][0]], "spatial_label": [0, 1, 0, 0]})
        # cups should be on the right of plates
        if "cup" in name_id_dict and "plate" in name_id_dict:
            goals.append({"type": "pattern:spatial", "obj_ids": [name_id_dict["cup"][0], name_id_dict["plate"][0]], "spatial_label": [0, 1, 0, 0]})
        # bowls should be on the stack on plates
        if "bowl" in name_id_dict and "plate" in name_id_dict:
            goals.append({"type": "pattern:tower", "obj_ids": [name_id_dict["plate"][0]] + name_id_dict["bowl"]})
        return goals


SEMANTIC_PATTERN_DICT = {
    "dinner": DinnerPattern,
}

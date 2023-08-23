"""Template to generate prompt
We majorly care three different prompts and their combination:
0. Anchor: (Define a anchor objects)  # Object-selection
1. Region: Put/Place/Leave {objs} at/in/on {region}
2. Pattern: Put/Place/Leave {objs} at/in/on {pattern}
3. Pair: Put/Place/Leave {obj1} {relation} {obj2}
"""
from __future__ import annotations
import random


class PromptGenerator:
    """Prompt generator"""
    def __init__(self, rng):
        self.rng = rng
        self.place_action_list = ["Put", "Place", "Leave"]
        self.pattern_list = ["Circle", "Square"]
        self.obj_list = ["Letter A", "Letter B", "Letter C"]
        self.prep_list = ["at", "in", "on"]
        self.correlative_list = ["; then ", "; and ", "; while ", "; so ", "; "]
        self.region_prompt = ""
        self.pattern_prompt = ""
        self.pair_prompt = ""

    def reset(self):
        """Reset the generator"""
        self.region_prompt = ""
        self.pattern_prompt = ""
        self.pair_prompt = ""

    def gen_anchor_prompt(self):
        pass

    def gen_pattern_prompt(self, pattern_objs: list[str], pattern: str):
        """Generate pattern prompt"""
        self.rng.shuffle(pattern_objs)
        obj_str = ", ".join(pattern_objs)
        self.pattern_prompt = f"{self.rng.choice(self.place_action_list)} {obj_str} {self.rng.choice(self.prep_list)} a {pattern} pattern"

    def gen_region_prompt(self, region_objs: list[str], region: str):
        """Generate region prompt"""
        self.rng.shuffle(region_objs)
        obj_str = ", ".join(region_objs)
        self.region_prompt = f"{self.rng.choice(self.place_action_list)} {obj_str} {self.rng.choice(self.prep_list)} {region}"

    def gen_pair_prompt(self, obj1: str, obj2: str, relation: str):
        """Generate pair prompt"""
        self.pair_prompt = f"{self.rng.choice(self.place_action_list)} {obj1} {relation} {obj2}"

    @property
    def prompt(self):
        """Randomly combine three prompt"""
        prompt_candidate = [self.region_prompt, self.pattern_prompt, self.pair_prompt]
        prompt_candidate = list(filter(lambda x: x != "", prompt_candidate))
        num_prompt = min(self.rng.integers(1, 3), len(prompt_candidate))
        prompt_candidate = self.rng.choice(prompt_candidate, num_prompt, replace=False)
        correlative = self.rng.choice(self.correlative_list, num_prompt - 1)
        prompt_str = ""
        for i in range(num_prompt):
            prompt_str += prompt_candidate[i]
            if i < num_prompt - 1:
                prompt_str += correlative[i]
        return prompt_str


if __name__ == "__main__":
    rng = random.Random(1)
    prompt_generator = PromptGenerator(rng)
    prompt_generator.gen_pattern_prompt(["Letter A", "Letter B", "Letter C"], "Circle")
    prompt_generator.gen_region_prompt(["Letter A", "Letter B"], "Blue part")
    prompt_generator.gen_pair_prompt("Letter A", "Letter B", "next to")
    print(prompt_generator.prompt)
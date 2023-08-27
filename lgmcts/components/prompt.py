"""Template to generate prompt
We majorly care three different prompts and their combination:
1. Region: Put/Place/Leave {obj_str} at/in/on {region}
2. Pattern: Put/Place/Leave {obj_str} at/in/on {pattern}
3. Pair: Put/Place/Leave {obj1} {relation} {obj2}

PromptGenerator has different styles:
1. strdiff: prompt in StructDiff style
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

    ## Prompt generation
    def gen_pattern_prompt(self, obj_str: str, pattern: str):
        """Generate pattern prompt"""
        prompt_type = self.rng.integers(0, 2)
        if prompt_type == 0:
            self.pattern_prompt = f"{self.rng.choice(self.place_action_list)} {obj_str} {self.rng.choice(self.prep_list)} a {pattern} pattern"
        elif prompt_type == 1:
            self.pattern_prompt = f"{self.rng.choice(self.place_action_list)} a {pattern} pattern {self.rng.choice(self.prep_list)} using {obj_str}"
        else:
            self.pattern_prompt = f"Select {obj_str}, and {self.rng.choice(self.place_action_list)} them {self.rng.choice(self.prep_list)} a {pattern} pattern"
    
    def gen_region_prompt(self, obj_str: str, region: str):
        """Generate region prompt"""
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
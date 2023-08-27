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
import cv2
import numpy as np
import os
import time


class PromptGenerator:
    """Prompt generator"""
    def __init__(self, rng, render_img_height: int = 128, font_scale: float = 0.8, font_weight: int = 1):
        self.rng = rng
        self.place_action_list = ["Put", "Place", "Leave"]
        self.pattern_list = ["Circle", "Square"]
        self.obj_list = ["Letter A", "Letter B", "Letter C"]
        self.prep_list = ["at", "in", "on"]
        self.correlative_list = ["; then ", "; and ", "; while ", "; so ", "; "]
        self.region_prompt = ""
        self.pattern_prompt = ""
        self.pair_prompt = ""
        self._prompt_str = ""
        # vis-related
        self._render_img_height = render_img_height
        self._font_scale = font_scale
        self._font_weight = font_weight
        self._font = cv2.FONT_HERSHEY_COMPLEX
        self._display = Cv2Display(window_name="LGMCTS Task Prompt")

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
            self.pattern_prompt = f"{self.rng.choice(self.place_action_list)} a {pattern} pattern using {obj_str}"
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
        return self._prompt_str

    @prompt.setter
    def prompt(self, prompt_str):
        """Set prompt"""
        self._prompt_str = prompt_str

    def gen_prompt(self):
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
        self._prompt_str = prompt_str

    ## Prompt visualization method
    def get_prompt_img_from_text(
        self,
        text: str,
        left_margin: int = 0,
    ):
        lang_textsize = cv2.getTextSize(
            text, self._font, self._font_scale, self._font_weight
        )[0]

        text_width, text_height = lang_textsize[0], lang_textsize[1]
        lang_textX = left_margin
        lang_textY = (self._render_img_height + lang_textsize[1]) // 2

        image_size = self._render_img_height, text_width + left_margin, 3
        image = np.zeros(image_size, dtype=np.uint8)
        image.fill(255)
        text_img = cv2.putText(
            image,
            text,
            org=(lang_textX, lang_textY),
            fontScale=self._font_scale,
            fontFace=self._font,
            color=(0, 0, 0),
            thickness=self._font_weight,
            lineType=cv2.LINE_AA,
        )
        return text_img

    def render(self):
        prompt_img = self.get_prompt_img_from_text(self._prompt_str)
        self._display(prompt_img)

    def close(self):
        self._display.close()


class Cv2Display:
    def __init__(
        self,
        window_name="display",
        image_size=None,
        channel_order="auto",
        bgr2rgb=True,
        step_sleep=0,
        enabled=True,
    ):
        """
        Use cv2.imshow() to pop a window, requires virtual desktop GUI

        Args:
            channel_order: auto, hwc, or chw
            image_size: None to use the original image size, otherwise resize
            step_sleep: sleep for a few seconds
        """
        self._window_name = window_name
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            assert image_size is None or len(image_size) == 2
        self._image_size = image_size
        assert channel_order in ["auto", "chw", "hwc"]
        self._channel_order = channel_order
        self._bgr2rgb = bgr2rgb
        self._step_sleep = step_sleep
        self._enabled = enabled

    def _resize(self, img):
        if self._image_size is None:
            return img
        H, W = img.shape[:2]
        Ht, Wt = self._image_size  # target
        return cv2.resize(
            img,
            self._image_size,
            interpolation=cv2.INTER_AREA if Ht < H else cv2.INTER_LINEAR,
        )

    def _reorder(self, img):
        if self._channel_order == "chw":
            return np.transpose(img, (1, 2, 0))
        elif self._channel_order == "hwc":
            return img
        else:
            if img.shape[0] in [1, 3]:  # chw
                return np.transpose(img, (1, 2, 0))
            else:
                return img

    def __call__(self, img):
        if not self._enabled:
            return
        import torch

        # prevent segfault in IsaacGym
        display_var = os.environ.get("DISPLAY", None)
        if not display_var:
            os.environ["DISPLAY"] = ":0.0"

        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()

        img = self._resize(self._reorder(img))
        if self._bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        time.sleep(self._step_sleep)
        cv2.imshow(self._window_name, img)
        cv2.waitKey(1)

        if display_var is not None:
            os.environ["DISPLAY"] = display_var

    def close(self):
        if not self._enabled:
            return
        cv2.destroyWindow(self._window_name)


if __name__ == "__main__":
    rng = random.Random(1)
    prompt_generator = PromptGenerator(rng)
    prompt_generator.gen_pattern_prompt(["Letter A", "Letter B", "Letter C"], "Circle")
    prompt_generator.gen_region_prompt(["Letter A", "Letter B"], "Blue part")
    prompt_generator.gen_pair_prompt("Letter A", "Letter B", "next to")
    print(prompt_generator.prompt)
import os
from functools import partial

import numpy as np

__all__ = ["container_replace_fn", "kit_obj_fn", "google_scanned_obj_fn", "shapenet_obj_fn"]

Z_SHRINK_FACTOR = 1.1
XY_SHRINK_FACTOR = 1


def default_replace_fn(*args, **kwargs):
    size = kwargs["size"]
    scaling = kwargs["scaling"]
    if isinstance(scaling, float):
        size = [s * scaling for s in size]
    else:
        size = [s1 * s2 for s1, s2 in zip(scaling, size)]
    return {"DIM": size}


def container_replace_fn(*args, **kwargs):
    size = kwargs["size"]
    size = [
        value / XY_SHRINK_FACTOR if i < 2 else value / Z_SHRINK_FACTOR
        for i, value in enumerate(size)
    ]
    scaling = kwargs["scaling"]
    if isinstance(scaling, float):
        size = [s * scaling for s in size]
    else:
        size = [s1 * s2 for s1, s2 in zip(scaling, size)]

    return {"DIM": size, "HALF": np.float32(size) / 2}


def _kit_obj_common(*args, **kwargs):
    fname = kwargs["fname"]
    assets_root = kwargs["assets_root"]
    fname = os.path.join(assets_root, "kitting", fname)
    scale = get_scale_from_map(fname[:-4].split("/")[-1], _kit_obj_scale_map)
    scaling = kwargs["scaling"]
    if isinstance(scaling, float):
        scale = [s * scaling for s in scale]
    else:
        scale = [s1 * s2 for s1, s2 in zip(scaling, scale)]
    return {
        "FNAME": (fname,),
        "SCALE": [
            scale[0] / XY_SHRINK_FACTOR,
            scale[1] / XY_SHRINK_FACTOR,
            scale[2] / Z_SHRINK_FACTOR,
        ],
    }


def kit_obj_fn(fname):
    return partial(_kit_obj_common, fname=fname)


def _google_scanned_obj_common(*args, **kwargs):
    fname = kwargs["fname"]
    assets_root = kwargs["assets_root"]
    fname = os.path.join(assets_root, "google", "meshes_fixed", fname)
    scale = get_scale_from_map(fname[:-4].split("/")[-1], _google_scanned_obj_scale_map)
    scaling = kwargs["scaling"]
    if isinstance(scaling, float):
        scale = [s * scaling for s in scale]
    else:
        scale = [s1 * s2 for s1, s2 in zip(scaling, scale)]
    return {
        "FNAME": (fname,),
        "SCALE": [
            scale[0] / XY_SHRINK_FACTOR,
            scale[1] / XY_SHRINK_FACTOR,
            scale[2] / Z_SHRINK_FACTOR,
        ],
        "COLOR": (0.2, 0.2, 0.2),
    }


def google_scanned_obj_fn(fname):
    return partial(_google_scanned_obj_common, fname=fname)


def _shapenet_obj_common(*args, **kwargs):
    fname = kwargs["fname"]
    assets_root = kwargs["assets_root"]
    fname = os.path.join(assets_root, "shapenet", "meshes", fname)
    scale = get_scale_from_map(fname[:-4].split("/")[-1], _shapenet_obj_scale_map)
    scaling = kwargs["scaling"]
    if isinstance(scaling, float):
        scale = [s * scaling for s in scale]
    else:
        scale = [s1 * s2 for s1, s2 in zip(scaling, scale)]
    return {
        "FNAME": (fname,),
        "SCALE": [
            scale[0] / XY_SHRINK_FACTOR,
            scale[1] / XY_SHRINK_FACTOR,
            scale[2] / Z_SHRINK_FACTOR,
        ],
        # "COLOR": (0.5, 0.5, 0.5),
    }


def shapenet_obj_fn(fname):
    return partial(_shapenet_obj_common, fname=fname)


def get_scale_from_map(key, map):
    scale = map.get(key)
    if isinstance(scale, float):
        scale = [scale, scale, scale]
    return scale


_kit_obj_scale_map = {
    "capital_letter_a": 1.0,
    "capital_letter_e": 1.0,
    "capital_letter_g": 1.0,
    "capital_letter_m": 1.0,
    "capital_letter_r": 1.0,
    "capital_letter_t": 1.0,
    "capital_letter_v": 1.0,
    "cross": 1.0,
    "diamond": 1.0,
    "triangle": 1.0,
    "flower": 1.0,
    "heart": 1.0,
    "hexagon": 1.0,
    "pentagon": 1.0,
    "right_angle": 1.0,
    "ring": 1.0,
    "round": 1.0,
    "square": 1.0,
    "star": 1.0,
}


_google_scanned_obj_scale_map = {
    "frypan": 0.275 * 3,
}

_shapenet_obj_scale_map = {
    "mug": 1.0,
    "basket": 1.0,
    "basket1": 1.0,
    "basket3": 1.0,
    "beerbottle": 1.0,
    "beerbottle1": 1.0,
    "bottle": 1.0,
    "bottle2": 1.0,
    "shapenet_bowl": 1.0,
    "cellphone": 1.0,
    "cellphone2": 1.0,
    "knife": 1.0,
    "mug1": 1.0,
    "mug2": 1.0,
    "pillbottle": 1.0,
    "pillbottle2": 1.0,
    "sodacan2": 1.0,
    "winebottle": 1.0,
    "winebottle1": 1.0,
    "winebottle2": 1.0
}

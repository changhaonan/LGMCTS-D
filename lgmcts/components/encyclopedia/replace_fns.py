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
    "capital_letter_a": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "capital_letter_e": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "capital_letter_g": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "capital_letter_m": [0.003 * 1.7, 0.003 * 1.7, 0.001 * 1.7],
    "capital_letter_r": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "capital_letter_t": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "capital_letter_v": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "cross": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "diamond": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "triangle": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "flower": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "heart": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "hexagon": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "pentagon": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "right_angle": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "ring": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "round": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "square": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "star": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
}


_google_scanned_obj_scale_map = {
    "frypan": 0.275 * 3,
}

# _shapenet_obj_scale_map = {
#     "mug": 0.175,
#     "basket": 0.175,
#     "basket1": 0.175,
#     "basket3": 0.175,
#     "beerbottle": 0.175,
#     "beerbottle1": 0.175,
#     "bottle": 0.175,
#     "bottle2": 0.175,
#     "bowl": 0.175,
#     "cellphone": 0.175,
#     "cellphone2": 0.175,
#     "knife": 0.175,
#     "mug1": 0.175,
#     "mug2": 0.175,
#     "pillbottle": 0.175,
#     "pillbottle2": 0.175,
#     "sodacan2": 0.175,
#     "winebottle": 0.175,
#     "winebottle1": 0.175,
#     "winebottle2": 0.175
# }

_shapenet_obj_scale_map = {
    "mug": 1.0,
    "basket": 1.0,
    "basket1": 2.0,
    "basket3": 0.2,
    "beerbottle": 0.3,
    "beerbottle1": 0.3,
    "bottle": 0.3,
    "bottle2": 0.3,
    "bowl": 0.7,
    "cellphone": 1.0,
    "cellphone2": 0.5,
    "knife": 0.4,
    "mug1": 0.3,
    "mug2": 0.3,
    "pillbottle": 0.3,
    "pillbottle2": 0.3,
    "sodacan2": 0.3,
    "winebottle": 0.3,
    "winebottle1": 0.3,
    "winebottle2": 0.3
}

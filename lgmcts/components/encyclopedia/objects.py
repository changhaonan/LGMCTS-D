import math
from enum import Enum
from typing import List
import json
import importlib_resources
from .definitions import ObjEntry, SizeRange
from .profiles import ProfilePedia
from .replace_fns import *

ASSET_ROOT = str(importlib_resources.files("lgmcts.assets"))  #FIXME: this is a hack


class ObjPedia(Enum):
    """
    An encyclopedia of objects in VIMA world.
    """

    BOWL = ObjEntry(
        name="bowl",
        assets="bowl/bowl.urdf",
        size_range=SizeRange(
            low=(0.17, 0.17, 0),
            high=(0.17, 0.17, 0),
        ),
        symmetry=0,
        profile=ProfilePedia.CIRCLE_LIKE,
    )

    BLOCK = ObjEntry(
        name="block",
        alias=["cube"],
        assets="stacking/block.urdf",
        size_range=SizeRange(
            low=(0.07, 0.07, 0.07),
            high=(0.07, 0.07, 0.07),
        ),
        from_template=True,
        symmetry=1 / 4 * math.pi,
        profile=ProfilePedia.SQUARE_LIKE,
    )

    SHORTER_BLOCK = ObjEntry(
        name="shorter block",
        alias=["cube"],
        assets="stacking/block.urdf",
        size_range=SizeRange(
            low=(0.07, 0.07, 0.03),
            high=(0.07, 0.07, 0.03),
        ),
        from_template=True,
        symmetry=1 / 4 * math.pi,
        profile=ProfilePedia.SQUARE_LIKE,
    )

    PALLET = ObjEntry(
        name="pallet",
        assets="pallet/pallet.urdf",
        size_range=SizeRange(
            low=(0.3 * 0.7, 0.25 * 0.7, 0.25 * 0.7 - 0.14),
            high=(0.3 * 0.7, 0.25 * 0.7, 0.25 * 0.7 - 0.14),
        ),
        profile=ProfilePedia.SQUARE_LIKE,
    )

    FRAME = ObjEntry(
        name="frame",
        novel_name=["zone"],
        assets="zone/zone.urdf",
        size_range=SizeRange(
            low=(0.15 * 1.5, 0.15 * 1.5, 0),
            high=(0.15 * 1.5, 0.15 * 1.5, 0),
        ),
        profile=ProfilePedia.SQUARE_LIKE,
    )

    CONTAINER = ObjEntry(
        name="container",
        assets="container/container-template.urdf",
        size_range=SizeRange(
            low=(0.15, 0.15, 0.05),
            high=(0.17, 0.17, 0.05),
        ),
        from_template=True,
        replace_fn=container_replace_fn,
        profile=ProfilePedia.SQUARE_LIKE,
    )

    THREE_SIDED_RECTANGLE = ObjEntry(
        name="three-sided rectangle",
        assets="square/square-template.urdf",
        size_range=SizeRange(
            low=(0.2, 0.2, 0.0),
            high=(0.2, 0.2, 0.0),
        ),
        from_template=True,
        replace_fn=container_replace_fn,
    )

    SMALL_BLOCK = ObjEntry(
        name="small block",
        assets="block/small.urdf",
        size_range=SizeRange(
            low=(0.03, 0.03, 0.03),
            high=(0.03, 0.03, 0.03),
        ),
        from_template=True,  # this will activate the replace dict, i.e. (SIZE here)
    )

    LINE = ObjEntry(
        name="line",
        assets="line/line-template.urdf",
        size_range=SizeRange(
            low=(0.25, 0.04, 0.001),
            high=(0.25, 0.04, 0.001),
        ),
        from_template=True,  # this will activate the replace dict, i.e. (SIZE here)
    )

    SQUARE = ObjEntry(
        name="square",
        assets="square/square-template-allsides.urdf",
        size_range=SizeRange(
            low=(0.2, 0.04, 0.001),
            high=(0.2, 0.04, 0.001),
        ),
        from_template=True,
        replace_fn=container_replace_fn,
        profile=ProfilePedia.SQUARE_LIKE,
    )

    CAPITAL_LETTER_A = ObjEntry(
        name="letter A",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_a.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_E = ObjEntry(
        name="letter E",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_e.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_G = ObjEntry(
        name="letter G",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_g.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_M = ObjEntry(
        name="letter M",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.7, 0.08 * 1.7, 0.02 * 1.7),
            high=(0.08 * 1.7, 0.08 * 1.7, 0.02 * 1.7),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_m.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_R = ObjEntry(
        name="letter R",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_r.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_T = ObjEntry(
        name="letter T",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_t.obj"),
        symmetry=2 * math.pi,
    )

    CAPITAL_LETTER_V = ObjEntry(
        name="letter V",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("capital_letter_v.obj"),
        symmetry=2 * math.pi,
    )

    CROSS = ObjEntry(
        name="cross",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("cross.obj"),
        symmetry=math.pi,
    )

    DIAMOND = ObjEntry(
        name="diamond",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("diamond.obj"),
        symmetry=math.pi,
    )

    TRIANGLE = ObjEntry(
        name="triangle",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("triangle.obj"),
        symmetry=2 / 3 * math.pi,
    )

    FLOWER = ObjEntry(
        name="flower",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("flower.obj"),
        symmetry=math.pi / 2,
    )

    HEART = ObjEntry(
        name="heart",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("heart.obj"),
        symmetry=2 * math.pi,
    )

    HEXAGON = ObjEntry(
        name="hexagon",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("hexagon.obj"),
        symmetry=2 / 6 * math.pi,
    )

    PENTAGON = ObjEntry(
        name="pentagon",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("pentagon.obj"),
        symmetry=2 / 5 * math.pi,
    )

    L_BLOCK = ObjEntry(
        name="L-shaped block",
        alias=["L-shape"],
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("right_angle.obj"),
        symmetry=2 * math.pi,
    )

    RING = ObjEntry(
        name="ring",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("ring.obj"),
        symmetry=0,
        profile=ProfilePedia.CIRCLE_LIKE,
    )

    ROUND = ObjEntry(
        name="round",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("round.obj"),
        profile=ProfilePedia.CIRCLE_LIKE,
    )

    STAR = ObjEntry(
        name="star",
        assets="kitting/object-template.urdf",
        size_range=SizeRange(
            low=(0.08 * 1.6, 0.08 * 1.6, 0.02 * 1.6),
            high=(0.08 * 1.8, 0.08 * 1.8, 0.02 * 1.8),
        ),
        from_template=True,
        replace_fn=kit_obj_fn("star.obj"),
        symmetry=2 * math.pi / 5,
    )

    # Google scanned objects
    PAN = ObjEntry(
        name="pan",
        assets="google/object-template.urdf",
        # size_range=SizeRange(low=(0.275, 0.275, 0.05), high=(0.275, 0.275, 0.05),),
        size_range=SizeRange(
            low=(0.16, 0.24, 0.03),
            high=(0.16, 0.24, 0.03),
        ),
        from_template=True,
        replace_fn=google_scanned_obj_fn("frypan.obj"),
        symmetry=0,
        profile=ProfilePedia.CIRCLE_LIKE,
    )

    HANOI_STAND = ObjEntry(
        name="stand",
        assets="hanoi/stand.urdf",
        size_range=SizeRange(
            low=(0.18, 0.54, 0.01),
            high=(0.18, 0.54, 0.01),
        ),
    )
    HANOI_DISK = ObjEntry(
        name="disk",
        assets="hanoi/disk-mod.urdf",
        size_range=SizeRange(low=(0.18, 0.18, 0.035), high=(0.18, 0.18, 0.035)),
        profile=ProfilePedia.CIRCLE_LIKE,
    )
     
    # ShapeNet objects
    SHAPENET_BASKET = ObjEntry(
        name="basket",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/basket.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/basket.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("basket.obj"),
        # symmetry=0,
    )
    
    SHAPENET_BASKET1 = ObjEntry(
        name="basket1",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/basket1.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/basket1.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("basket1.obj"),
        # symmetry=0,
    )
    
    SHAPENET_BASKET3 = ObjEntry(
        name="basket3",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/basket3.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/basket3.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("basket3.obj"),
        # symmetry=0,
    )
    
    SHAPENET_BEERBOTTLE = ObjEntry(
        name="beerbottle",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("beerbottle.obj"),
        # symmetry=0,
    )
    
    SHAPENET_BEERBOTTLE1 = ObjEntry(
        name="beerbottle1",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle1.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle1.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("beerbottle1.obj"),
        # symmetry=0,
    )
    
    SHAPENET_BOTTLE = ObjEntry(
        name="bottle",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bottle.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bottle.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("bottle.obj"),
        # symmetry=0,
    )

    SHAPENET_BOTTLE2 = ObjEntry(
        name="bottle2",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bottle2.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bottle2.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("bottle2.obj"),
        # symmetry=0,
    )

    SHAPENET_BOWL = ObjEntry(
        name="bowl",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("bowl.obj"),
        # symmetry=0,
    )

    SHAPENET_CELLPHONE = ObjEntry(
        name="cellphone",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/cellphone.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/cellphone.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("cellphone.obj"),
        # symmetry=0,
    )

    SHAPENET_CELLPHONE2 = ObjEntry(
        name="cellphone2",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/cellphone2.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/cellphone2.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("cellphone2.obj"),
        # symmetry=0,
    )


    SHAPENET_KNIFE = ObjEntry(
        name="knife",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/knife.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/knife.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("knife.obj"),
        # symmetry=0,
    )

    SHAPENET_MUG = ObjEntry(
        name="mug",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("mug.obj"),
        # symmetry=0,
    )

    SHAPENET_MUG1 = ObjEntry(
        name="mug1",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug1.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug1.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("mug1.obj"),
        # symmetry=0,
    )


    SHAPENET_MUG2 = ObjEntry(
        name="mug2",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug2.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug2.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("mug2.obj"),
        # symmetry=0,
    )

    # For pillbottle
    SHAPENET_PILLBOTTLE = ObjEntry(
        name="pillbottle",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/pillbottle.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/pillbottle.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("pillbottle.obj"),
    )

    # For pillbottle2
    SHAPENET_PILLBOTTLE2 = ObjEntry(
        name="pillbottle2",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/pillbottle2.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/pillbottle2.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("pillbottle2.obj"),
    )

    # For sodacan2
    SHAPENET_SODACAN2 = ObjEntry(
        name="sodacan2",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/sodacan2.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/sodacan2.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("sodacan2.obj"),
    )

    # For winebottle
    SHAPENET_WINEBOTTLE = ObjEntry(
        name="winebottle",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/winebottle.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/winebottle.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("winebottle.obj"),
    )

    # For winebottle1
    SHAPENET_WINEBOTTLE1 = ObjEntry(
        name="winebottle1",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/winebottle1.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/winebottle1.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("winebottle1.obj"),
    )

    # For winebottle2
    SHAPENET_WINEBOTTLE2 = ObjEntry(
        name="winebottle2",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/winebottle2.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/winebottle2.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("winebottle2.obj"),
    )

    # For mug3
    SHAPENET_MUG3 = ObjEntry(
        name="mug3",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug3.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug3.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("mug3.obj"),
    )

    # For mug4
    SHAPENET_MUG4 = ObjEntry(
        name="mug4",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug4.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug4.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("mug4.obj"),
    )

    # For mug5
    SHAPENET_MUG5 = ObjEntry(
        name="mug5",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug5.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/mug5.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("mug5.obj"),
    )

    # For cellphone4
    SHAPENET_CELLPHONE4 = ObjEntry(
        name="cellphone4",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/cellphone4.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/cellphone4.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("cellphone4.obj"),
    )

    # For bowl1
    SHAPENET_BOWL1 = ObjEntry(
        name="bowl1",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl1.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl1.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("bowl1.obj"),
    )

    # For beerbottle2
    SHAPENET_BEERBOTTLE2 = ObjEntry(
        name="beerbottle2",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle2.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle2.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("beerbottle2.obj"),
    )

    # For bowl3
    SHAPENET_BOWL3 = ObjEntry(
        name="bowl3",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl3.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl3.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("bowl3.obj"),
    )

    # For bowl4
    SHAPENET_BOWL4 = ObjEntry(
        name="bowl4",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl4.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl4.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("bowl4.obj"),
    )

    # For bowl5
    SHAPENET_BOWL5 = ObjEntry(
        name="bowl5",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl5.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl5.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("bowl5.obj"),
    )

    # For bowl6
    SHAPENET_BOWL6 = ObjEntry(
        name="bowl6",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl6.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bowl6.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("bowl6.obj"),
    )

    # For bottle3
    SHAPENET_BOTTLE3 = ObjEntry(
        name="bottle3",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bottle3.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/bottle3.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("bottle3.obj"),
    )
    # For beerbottle3
    SHAPENET_BEERBOTTLE3 = ObjEntry(
        name="beerbottle3",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle3.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle3.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("beerbottle3.obj"),
    )

    # For beerbottle4
    SHAPENET_BEERBOTTLE4 = ObjEntry(
        name="beerbottle4",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle4.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle4.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("beerbottle4.obj"),
    )

    # For beerbottle5
    SHAPENET_BEERBOTTLE5 = ObjEntry(
        name="beerbottle5",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle5.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle5.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("beerbottle5.obj"),
    )

    # For beerbottle6
    SHAPENET_BEERBOTTLE6 = ObjEntry(
        name="beerbottle6",
        assets="shapenet/object-template.urdf",
        size_range=SizeRange(low=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle6.json"))["min"]), high=tuple(json.load(open(f"{ASSET_ROOT}/shapenet/meshes/beerbottle6.json"))["max"])),
        profile=ProfilePedia.CIRCLE_LIKE,
        from_template=True,
        replace_fn=shapenet_obj_fn("beerbottle6.obj"),
    )

    @classmethod
    def all_entries(cls) -> List[ObjEntry]:
        return [e for e in cls]

    @classmethod
    def lookup_object_by_name(cls, name: str) -> ObjEntry:
        """
        Given an object name, return corresponding object entry
        """
        for e in cls:
            if name == e.value.name:
                return e
        raise ValueError(f"Cannot find provided object {name}")

    @classmethod
    def all_entries_no_rotational_symmetry(cls) -> List[ObjEntry]:
        return [
            e
            for e in cls
            if e.value.symmetry is not None
            and math.isclose(e.value.symmetry, 2 * math.pi, rel_tol=1e-6, abs_tol=1e-8)
        ]

    ##TODO: Add more objects @Knowdi
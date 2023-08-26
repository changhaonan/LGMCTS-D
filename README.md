# Data generate bed for LGMCTS

## How to run

1. Generate data (LGMCTS)

```
python lgmcts/scripts/data_generation/gen_lgmcts.py
```

2. Run offline test

```
python lgmcts/scripts/eval/eval_lgmcts.py
```

## Generate data for StructDiffusion

```
python lgmcts/scripts/data_generation/gen_strdiff.py
```

## Output format

obs:
    - rgb
        - top
        - front
    - depth
    - point cloud
    - pose

Point cloud & pose are all padded with zero. Their shape is of (max_num_obj * max_pcd_size, 3) and (max_num_obj, 7). 

## BUG

- Currently, the sequential sampling is not working. Need to be fixed.

## TODO:

- Fix the environment setup.
- Test MCTS
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

## Batch-level generation 

```
python lgmcts/scripts/data_generation/gen_lgmcts.py --num_episodes=100
```

## Eval

Use without gt_pose.

```
python lgmcts/scripts/eval/eval_lgmcts.py --method=mcts --n_epoches=100 --mask_mode=raw_mask
```

Use with gt_pose
```
python lgmcts/scripts/eval/eval_lgmcts.py --method=mcts --n_epoches=100 --mask_mode=raw_mask --use_gt_pose
```
## BUG

- Currently, the sequential sampling is not working. Need to be fixed.
- Is the sampling repeatable??

- Why do we need flip_xy in eval of mcts, but not in eval of mcts + GT?

## TODO:

- [ ] Theoretically, the action part should have bug, but what kind of bug is still unknown.
- [ ] Include the spatial pattern into pipeline.
- [ ] Change the seed structure.
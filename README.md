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

## Evaluating LLMs as Few Shot Planners (LFSP)  - Progprompt and Code as Policies

### Dataset preparation
Download the [Structformer dataset](https://drive.google.com/drive/folders/19k2ZTlgC0itD-BLl22J0AkN8Ej8UMWnX?usp=sharing) for each of the four patterns (line, circle, dinner, and tower). Extract and place them inside the folder `output/lfsp/eval_single_pattern/`
### Generating the files for few-shot prompting
For Line Pattern
```
python lgmcts/scripts/eval/lfsp_prompt.py --pattern=line
```
Similarly generate the files for the other 3 patterns by replacing the command line argument 'pattern' with 'circle', 'dinner' and 'tower' respectively. 

### Generating LFSP's TAMP solution on the Structformer dataset
For Line Pattern
```
python lgmcts/scripts/eval/lfsp_res.py --pattern=line
```
Similarly generate the files for the other 3 patterns by replacing the command line argument 'pattern' with 'circle', 'dinner' and 'tower' respectively. 

### Evaluate the LFSP's TAMP solution on the Structformer dataset
For Line Pattern
```
python lgmcts/scripts/eval/lfsp_eval.py --pattern=line
```
Similarly generate the files for the other 3 patterns by replacing the command line argument 'pattern' with 'circle', 'dinner' and 'tower' respectively. 

## BUG

- Currently, the sequential sampling is not working. Need to be fixed.
- Is the sampling repeatable??

- Why do we need flip_xy in eval of mcts, but not in eval of mcts + GT?

## TODO:

- [ ] Theoretically, the action part should have bug, but what kind of bug is still unknown.
- [ ] Include the spatial pattern into pipeline.
- [ ] Change the seed structure.
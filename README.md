# Installation
## Conda environment setup
```
conda create -n lgmcts python=3.9 -y
conda activate lgmcts
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Under the lgmcts environment
pip install -r requirements
pip install -e .
```

## Adding the OPENAI API for LLM Parsing and using LLMs as Few Shot Planners
```
# In the root directory of LGMCTS-D directory
mkdir lgmcts/conf
```
Create a .pkl file with API keys stored as a list. Run the below python code. However, one API key is enough. 
```
# pickle package is already included in python3.9
import pickle
api_keys = [<api_key1>,<api_key2>, ..]
with open("lgmcts/conf/api_key.pkl", "wb") as fp:
    pickle.dump(fp, api_keys)
```

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

## Evaluating LLMs as Few Shot Planners (LFSP)  - Progprompt and Code as policies

### Evaluation on the Structformer dataset

#### Dataset preparation
```
# In the root directory of LGMCTS-D directory
mkdir -p output/lfsp/eval_single_pattern
```

Download the [Structformer dataset](https://drive.google.com/drive/folders/19k2ZTlgC0itD-BLl22J0AkN8Ej8UMWnX?usp=sharing) for each of the four patterns (line, circle, dinner, and tower). Extract them to the folder `output/lfsp/eval_single_pattern/`


#### Generating the files for few-shot prompting
For `Line` Pattern
```
python lgmcts/scripts/eval/lfsp_sformer_prompt.py --pattern=line
```
Similarly generate the files for the other 3 patterns by replacing the command line argument `pattern` with `circle`, `dinner` and `tower` respectively. 

#### Automated Evaluation on the full dataset
For `Line` Pattern
> Step 1:
```
python lfsp_sformer.py --pattern=line
```
> Step 2:
```
chmod +x lfsp_sformer_line.sh
./lfsp_sformer_line.sh
```
Results can be generated for the rest of the patterns similarly. 


### Evaluation on the ELGR Bench

#### Dataset preparation
Download the [ELGR Bench](https://drive.google.com/drive/folders/1QiwUofPF8rGkcZIraJ1WRBE0zHf4rjXV?usp=sharing). Extract them to the folder `output/lfsp/`

#### Automated Evaluation on the full dataset
> Step 1:
```
python lfsp_elgr.py
```
> Step 2:
```
chmod +x lfsp_elgr.sh
./lfsp_elgr.sh
``` 

## BUG

- Currently, the sequential sampling is not working. Need to be fixed.
- Is the sampling repeatable??

- Why do we need flip_xy in eval of mcts, but not in eval of mcts + GT?

## TODO:

- [ ] Theoretically, the action part should have bug, but what kind of bug is still unknown.
- [ ] Include the spatial pattern into pipeline.
- [ ] Change the seed structure.
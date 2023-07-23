# Data generate bed for LGMCTS

## How to run

```
python lgmcts/scripts/data_generation/run.py
```

The output will be show in `output`. Pointcloud and Pose will be saved inside `obs.pkl`.

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

- I don't know why there is a white line?

## TODO:

- Output pointcloud
- Output pose
import os
from omegaconf import OmegaConf
import importlib_resources
from lgmcts.tasks.base import BaseTask
from lgmcts.tasks.partition_files import *
from lgmcts.tasks.struct_rearrange import StructRearrange

__all__ = ["ALL_TASKS", "ALL_PARTITIONS", "PARTITION_TO_SPECS"]

_ALL_TASKS = {
    "struct_rearrange": [
        StructRearrange
    ],
}
ALL_TASKS = {
    f"{group}/{task.task_name}": task
    for group, tasks in _ALL_TASKS.items()
    for task in tasks
}
_ALL_TASK_SUB_NAMES = [
    task.task_name for tasks in _ALL_TASKS.values() for task in tasks
]

def _partition_file_path(fname) -> str:
    with importlib_resources.files("lgmcts.tasks.partition_files") as p:
        return os.path.join(str(p), fname)

def _load_partition_file(file: str):
    file = _partition_file_path(file)
    partition = OmegaConf.to_container(OmegaConf.load(file), resolve=True)
    partition_keys = set(partition.keys())
    for k in partition_keys:
        if k not in _ALL_TASK_SUB_NAMES:
            partition.pop(k)
    return partition

# test
TRAIN = _load_partition_file(file="train.yaml")
ALL_PARTITIONS = [
    "struct_rearrange"
]
PARTITION_TO_SPECS = {
    "train": TRAIN,
}
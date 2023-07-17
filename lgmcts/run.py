from lgmcts.tasks import BaseTask
from lgmcts.env.base import BaseEnv


if __name__ == '__main__':
    base_task = BaseTask(
        prompt_template='pick_place',
        modalities=['rgb'],
        obs_img_views=['front', 'top'],
        seed=0,
        debug=True)

    task = BaseEnv(
        task=base_task,
        modalities=['rgb'], 
        obs_img_views=['front', 'top'],
        seed=0,
        debug=True)
        
    task.reset()
from lgmcts.tasks import BaseTask


if __name__ == '__main__':
    task = BaseTask(
        prompt_template='Hello, world!',
        modalities=['rgb'], 
        obs_img_views=['front', 'top'],
        seed=0,
        debug=True)
    task.reset()
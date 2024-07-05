import os 
root_dir = os.getcwd()
fp = open(f"{root_dir}/lfsp_elgr.sh", "w")
for scene_seed in range(0, 20):
    for start in [0, 25, 50, 75]:
        fp.write('echo "Sleeping 10 seconds..."\n')
        fp.write('sleep 10\n')
        fp.write(f'echo "seed={scene_seed}" > "{root_dir}/LGMCTS-D/lgmcts/env/__init__.py"\n')
        fp.write(f'echo "LLM Result Generation for scene: {scene_seed}, start: {start}, end: {start+25}..."\n')
        fp.write(f'python "{root_dir}/lgmcts/scripts/eval/lfsp_elgr_res.py" --start={start} --end={start+25} --seed={scene_seed}\n')
        fp.write('echo "Sleeping 10 seconds..."\n')
        fp.write('sleep 10\n')
        fp.write(f'echo "Score Analysis scene: {scene_seed}, start: {start}, end: {start+25}..."\n')
        fp.write(f'python "{root_dir}/lgmcts/scripts/eval/lfsp_elgr_eval.py" --start={start} --end={start+25} --seed={scene_seed}\n')
        fp.write('echo "Sleeping 10 seconds..."\n')
        fp.write('sleep 10\n')
fp.close()

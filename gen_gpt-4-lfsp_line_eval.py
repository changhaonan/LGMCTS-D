pattern = "dinner"
num_scenes = {
        "line" : 430,
        "circle" : 342,
        "tower" : 134,
        "dinner" : 244
    }
fp = open(f"/media/kowndi/T7 Shield/RSS24/LGMCTS-D/gpt-4-lfsp_{pattern}_eval.sh", "w")
for start in range(0, num_scenes[pattern], 25):
    end = min(num_scenes[pattern], start+25)
    fp.write('echo "Sleeping 60 seconds..."\n')
    fp.write('sleep 60\n')
    fp.write(f'echo "LLM Result Generation for scene, start: {start}, end: {end}..."\n')
    fp.write(f'/home/kowndi/anaconda3/envs/lgmcts/bin/python "/media/kowndi/T7 Shield/RSS24/LGMCTS-D/lgmcts/scripts/eval/lfsp_sformer_res.py" --start={start} --end={end} --pattern={pattern}\n')
    fp.write('echo "Sleeping 60 seconds..."\n')
    fp.write('sleep 60\n')
    fp.write(f'echo "Score Analysis, start: {start}, end: {end}..."\n')
    fp.write(f'/home/kowndi/anaconda3/envs/lgmcts/bin/python "/media/kowndi/T7 Shield/RSS24/LGMCTS-D/lgmcts/scripts/eval/lfsp_sformer_eval.py" --start={start} --end={end} --pattern={pattern} \n')
    fp.write('echo "Sleeping 60 seconds..."\n')
    fp.write('sleep 60\n')
fp.close()

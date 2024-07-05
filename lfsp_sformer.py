import argparse
import os 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="line", help="Pattern")
    args = parser.parse_args()

    pattern = args.pattern
    num_scenes = {
            "line" : 430,
            "circle" : 342,
            "tower" : 134,
            "dinner" : 244
        }
    root_dir = os.getcwd()
    fp = open(f"{root_dir}/lfsp_sformer_{pattern}.sh", "w")
    for start in range(0, num_scenes[pattern], 25):
        end = min(num_scenes[pattern], start+25)
        fp.write('echo "Sleeping 60 seconds..."\n')
        fp.write('sleep 60\n')
        fp.write(f'echo "LLM Result Generation for scene, start: {start}, end: {end}..."\n')
        fp.write(f'python "{root_dir}/lgmcts/scripts/eval/lfsp_sformer_res.py" --start={start} --end={end} --pattern={pattern}\n')
        fp.write('echo "Sleeping 60 seconds..."\n')
        fp.write('sleep 60\n')
        fp.write(f'echo "Score Analysis, start: {start}, end: {end}..."\n')
        fp.write(f'python "{root_dir}/lgmcts/scripts/eval/lfsp_sformer_eval.py" --start={start} --end={end} --pattern={pattern} \n')
        fp.write('echo "Sleeping 60 seconds..."\n')
        fp.write('sleep 60\n')
    fp.close()

"""Parse the language prompt using LLM"""
import ast
import os
import numpy as np
import pickle
import random
from lgmcts.components.llm_chatgpt import ChatGPTAPI
from lgmcts.env import seed

def gen_prompt_goal_from_llm(prompt_path: str, n_epoches: int = 0, checkpoint_list: list = [], use_llm: bool = True, run_llm: bool = True, encode_ids_to_llm: bool = True, obj_id_reverse_mappings: list = None, num_save_digits: int = 6, debug: bool = False):
    prompt_goals = None
    if not encode_ids_to_llm:
        if use_llm:
            if run_llm:
                result = perform_llm_parsing(prompt_bg_file=f"{prompt_path}/prompt_bg.txt",
                                             prompt_str_file=f"{prompt_path}/prompt_str_list.txt", prompt_example_file=f"{prompt_path}/prompt_example_indirect.txt", debug=debug)
                res = [ast.literal_eval(r) for r in result]
                with open(os.path.join(os.path.dirname(prompt_path), "prompt", "llm_result.pkl"), "wb") as fp:
                    pickle.dump(res, fp)
                # read obj_id_reverse_mapping
                if obj_id_reverse_mappings is None:
                    obj_id_reverse_mappings = []
                    for i in range(n_epoches):
                        checkpoint_path = os.path.join(prompt_path, checkpoint_list[i])
                        with open(checkpoint_path, "rb") as f:
                            env_state = pickle.load(f)
                        obj_id_reverse_mappings.append(env_state["obj_id_reverse_mapping"])
                parse_llm_result(prompt_path, res, obj_id_reverse_mappings)
            with open(os.path.join(prompt_path, "goal.pkl"), "rb") as fp:
                prompt_goals = pickle.load(fp)
    else:
        if use_llm:
            if run_llm:
                result = perform_llm_parsing(prompt_bg_file=f"{prompt_path}/prompt_bg.txt", prompt_str_file=f"{prompt_path}/prompt_str_list_real.txt",
                                             prompt_example_file=f"{prompt_path}/prompt_example_direct.txt", encode_ids_to_llm=encode_ids_to_llm, obj_id_reverse_mappings=obj_id_reverse_mappings, num_save_digits=num_save_digits, debug=debug)
                prompt_goals = [ast.literal_eval(r.split("```")[1].replace("\n", "")) for r in result]
                with open(os.path.join(prompt_path, "goal.pkl"), "wb") as fp:
                    pickle.dump(prompt_goals, fp)
            else:
                with open(os.path.join(prompt_path, "goal.pkl"), "rb") as fp:
                    prompt_goals = pickle.load(fp)
    return prompt_goals


def parse_llm_result(dataset_path: str, llm_result: str, obj_id_reverse_mappings: list):
    """Parse the result from LLM"""
    # generate prompt
    prompt_folder = os.path.join(os.path.dirname(dataset_path), "prompt")
    if not os.path.exists(prompt_folder):
        os.makedirs(prompt_folder)

    goals = []
    for ind, res in enumerate(llm_result):
        obj_id_reverse_mapping = obj_id_reverse_mappings[ind]
        goal = []
        for entry in res:
            goal_entry = dict()
            if entry["pattern"] != "spatial":
                goal_entry["type"] = f"pattern:{entry['pattern']}"
                goal_entry["obj_ids"] = []
                goal_entry["anchor_id"] = []    
                anchor_color = None
                for item in obj_id_reverse_mapping:
                    if obj_id_reverse_mapping[item]["obj_name"] == entry["anchor"]:
                        goal_entry["anchor_id"].append(item)
                        anchor_color = obj_id_reverse_mapping[item]["texture_name"]
                        break

                if entry["anchor_relation"] == "same":
                    for item in obj_id_reverse_mapping:
                        if obj_id_reverse_mapping[item]["texture_name"] == anchor_color:
                            goal_entry["obj_ids"].append(item)
                else:
                    for item in obj_id_reverse_mapping:
                        if obj_id_reverse_mapping[item]["texture_name"] != anchor_color:
                            goal_entry["obj_ids"].append(item)
                goal.append(goal_entry)
            else:
                goal_entry["type"] = f"pattern:{entry['pattern']}"
                goal_entry["obj_ids"] = []
                for obj_name in entry["objects"]:
                    for item in obj_id_reverse_mapping:
                        if obj_id_reverse_mapping[item]["obj_name"] == obj_name:
                            goal_entry["obj_ids"].append(item)
                            break
                goal_entry["obj_ids"] = goal_entry["obj_ids"]
                goal_entry["spatial_label"] = np.array([0, 0, 0, 0], dtype=np.int32)
                for sp_label in entry["spatial_label"]:
                    if "left" in sp_label:
                        goal_entry["spatial_label"][0] = 1
                    if "right" in sp_label:
                        goal_entry["spatial_label"][1] = 1
                    if "front" in sp_label:
                        goal_entry["spatial_label"][2] = 1
                    if "behind" in sp_label:
                        goal_entry["spatial_label"][3] = 1
                goal_entry["spatial_str"] = entry["spatial_str"]
                goal.append(goal_entry)
        goals.append(goal)
    with open(f"{dataset_path}/goal.pkl", "wb") as fp:
        pickle.dump(goals, fp)


def perform_llm_parsing(prompt_bg_file: str, prompt_str_file: str, prompt_example_file: str = None, encode_ids_to_llm: bool = False, obj_id_reverse_mappings: list = None, num_save_digits: int = 6, debug: bool = False):
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(prompt_bg_file))), "lgmcts", "conf", "api_key.pkl"), "rb") as fp:
        api_keys = pickle.load(fp)
    api_key = random.choice(api_keys)
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(prompt_bg_file)))
    prompt_db = ""
    prompt_db += open(prompt_bg_file, "r").read()
    prompt_db += open(prompt_example_file, "r").read()
    prompts = []
    with open(prompt_str_file, "r") as fp:
        iter = 0
        for line in fp.readlines():
            if encode_ids_to_llm:
                prompt_prior = f"Please be mindful of the object ids, names, and their colors accordingly."
                prompt_prior += f"There are {len(fp.readlines())} objects in the scene."
                if obj_id_reverse_mappings is None:
                    with open(f"{root_path}/output/struct_rearrange_{seed}/checkpoint_{iter:0{num_save_digits}d}.pkl", "rb") as f:
                        iter += 1
                        checkpoint = pickle.load(f)
                        ids = []
                        names = []
                        textures = []
                        for k, v in checkpoint["obj_id_reverse_mapping"].items():
                            ids.append(k)
                            names.append(v["obj_name"])
                            textures.append(v["texture_name"])
                else:
                    ids = []
                    names = []
                    textures = []
                    for k, v in obj_id_reverse_mappings[iter].items():
                        ids.append(k)
                        names.append(v["obj_name"])
                        textures.append(v["texture_name"])
                prompt_prior += f"The obj_ids are {ids}"
                prompt_prior += f"and object names are {names}.\n"
                prompt_prior += f"And the corresponding object colors are {textures}.\n"
                prompts.append(prompt_prior + "<user>\n" + line.strip() + "\n</user>")
            else:
                prompts.append(line.strip())
    # gpt-3.5-turbo-16k-0613
    chatgpt = ChatGPTAPI(model="gpt-4", api_key=api_key, db=prompt_db)
    ret = chatgpt.chat(str_msg=prompts)
    # print(ret)
    return ret[0]


if __name__ == "__main__":
    pass

"""Parse the language prompt using LLM"""
import os
import pickle
import random
from lgmcts.components.llm_chatgpt import ChatGPTAPI


def perform_llm_parsing(prompt_bg_file: str, prompt_str_file: str, prompt_example_file: str = None, encode_ids_to_llm: bool = False, num_save_digits: int = 6, debug: bool = False):
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(prompt_bg_file))), "lgmcts", "conf", "api_key.pkl"), "rb") as fp:
        api_keys = pickle.load(fp)
    api_key = random.choice(api_keys)
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(prompt_bg_file)))
    # prompt_example_file = f"{root_path}/lgmcts/scripts/data_generation/prompt_example.txt"
    prompt_db = ""
    prompt_db += open(prompt_bg_file, "r").read()
    prompt_db += open(prompt_example_file, "r").read()
    prompts = []
    with open(prompt_str_file, "r") as fp:
        iter = 0
        for line in fp.readlines():
            if encode_ids_to_llm:
                prompt_prior = f"Please be mindful of the object ids, names, and their colors accordingly."
                prompt_prior += "There are {len(fp.readlines())} objects in the scene."
                with open(f"{root_path}/output/struct_rearrange/checkpoint_{iter:0{num_save_digits}d}.pkl", "rb") as f:
                    iter += 1
                    checkpoint = pickle.load(f)
                    ids = []
                    names = []
                    textures = []
                    for k,v in checkpoint["obj_id_reverse_mapping"].items():
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

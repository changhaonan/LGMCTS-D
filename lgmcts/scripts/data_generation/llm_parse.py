"""Parse the language prompt using LLM"""
import os
import pickle
import random 
from lgmcts.components.llm_chatgpt import ChatGPTAPI

def perform_llm_parsing(prompt_bg_file: str, prompt_str_file: str, prompt_example_file: str = None, debug: bool = False):
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(prompt_bg_file))), "lgmcts", "conf", "api_key.pkl"), "rb") as fp:
       api_keys = pickle.load(fp)
    api_key = random.choice(api_keys)
    prompt_example_file = "/media/exx/T7 Shield/ICLR23/LGMCTS-D/lgmcts/scripts/data_generation/prompt_example.txt"
    prompt_db = ""
    prompt_db += open(prompt_bg_file, "r").read()
    prompt_db += open(prompt_example_file, "r").read()
    prompts = []
    with open(prompt_str_file, "r") as fp:
        for line in fp.readlines():
            prompts.append(line.strip())
    # gpt-3.5-turbo-16k-0613
    chatgpt = ChatGPTAPI(model="gpt-4", api_key=api_key, db=prompt_db)
    ret = chatgpt.chat(str_msg=prompts)
    # print(ret)
    return ret[0]

if __name__ == "__main__":
    pass
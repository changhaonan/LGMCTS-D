"""Parse the language prompt using LLM"""
import os
from lgmcts.components.llm_chatgpt import ChatGPTAPI


if __name__ == "__main__":
    api_key = "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"
    prompt_example_file = os.path.join(os.path.dirname(__file__), "prompt_example.txt")
    prompt_bg_file = os.path.join(os.path.dirname(__file__), "prompt_bg.txt")
    prompt_db = ""
    prompt_db += open(prompt_bg_file, "r").read()
    prompt_db += open(prompt_example_file, "r").read()

    prompts = []
    with open("/media/exx/T7 Shield/ICLR23/LGMCTS-D/nl-prompts.txt", "r") as fp:
        for line in fp.readlines():
            prompts.append(line.strip())

    chatgpt = ChatGPTAPI(model="gpt-4", api_key=api_key, db=prompt_db)
    ret = chatgpt.chat(str_msg=prompts)
    for res in ret[0]:
        print(res)
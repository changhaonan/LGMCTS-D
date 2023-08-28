"""Large language model chat with GPT API"""
import time
import openai
from typing import Dict, Any, Tuple, Union, List
from PIL import Image
import numpy as np
from lgmcts.utils.user import print_type_indicator
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed


class LLM:
    """Abstract class for large language model"""

    def __init__(self):
        self._is_api = False

    @property
    def is_api(self):
        return self._is_api

    @is_api.setter
    def is_api(self, is_api):
        self._is_api = is_api


class ChatGPTAPI(LLM):
    """Chat with GPT API"""
    def __init__(self, model: str="gpt-4", api_key: str="", db: str = None):
        super().__init__()
        self.is_api = True
        openai.api_key = api_key
        self.gpt_model = model
        self.system_prompt = {"role": "system", "content": db}

    def chat(
        self,
        str_msg: Union[str, List[Any]],
        img_msg: Union[List[Image.Image], List[np.ndarray], None] = None,
        **kwargs
    ) -> Tuple[str, bool]:
        # Print typing indicator
        print_type_indicator("LLM")
        if isinstance(str_msg, list):
            return self.talk_prompt_list(str_msg), True
        elif isinstance(str_msg, str):
            return self.talk_prompt_string(str_msg), True
        
    def _threaded_talk_prompt(self, prompt: Dict[str, Any], max_retries: int=1) -> Tuple[str, Any]:
        # print("Threaded execution of prompt: {}".format(prompt))
        retries = 0
        conversation = [self.system_prompt]
        while retries <= max_retries:
            try:
                # assert len(prompt) == 1
                
                # for key, value in prompt.items():
                conversation.append({"role": "system", "content": prompt})
                reply = openai.ChatCompletion.create(
                    model=self.gpt_model,
                    messages=conversation,
                    timeout=10 # Timeout in seconds for the API call
                    )
                reply_content = reply["choices"][0]["message"]["content"]
                conversation.append({"role": "assistant", "content": reply_content})
                return reply_content, None 
            except Exception as e:
                retries += 1
                if retries <= max_retries:
                    time.sleep(10) # Wait for 10 seconds before retrying
                else:
                    return None, str(e)

    def talk_prompt_list(self, prompt_list: List[Dict[str, Any]], batch_size: int=4) -> List[str]:
        """prompt_list is a list of dict, each dict has one key and one value"""
        results = []
        errors = []
        for i in range(0, len(prompt_list), batch_size):
            # Create thread pool
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                print("Batch Execution of prompts {} to {}".format(i, min(i+batch_size-1, len(prompt_list))))
                future_to_prompt = {executor.submit(self._threaded_talk_prompt, prompt, max_retries=2): prompt for prompt in prompt_list[i:i+batch_size]}
                for future in as_completed(future_to_prompt):
                    prompt = future_to_prompt[future]
                    try:
                        reply_content, error = future.result()
                        if reply_content is not None:
                            results.append(reply_content)
                        else:
                            errors.append(f"Error for prompt {prompt}: {error}")
                    except TimeoutError:
                        errors.append(f"Timeout error for prompt {prompt}")
            for error in errors:
                print(error)
            
        return results


# Main function
if __name__ == "__main__":
    prompt_db = "Assume you are a language-based motion planner. You will parse user's requirement into goal configuration and constraints."
    prompt_db += "Object_id of the objects in the scene are: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 for letter A, L-shaped block, letter V, triangle, letter M, block, letter T, letter G, ring, pentagon, letter E, heart, flower, cross"
    prompt_db += "And colors of the objects in the scene are:  yellow_and_blue_stripe, pink, red_and_yellow_stripe, green, pink, yellow_and_blue_stripe, green, yellow, pink, red, orange, green, yellow, red_and_yellow_stripe for letter A, L-shaped block, letter V, triangle, letter M, block, letter T, letter G, ring, pentagon, letter E, heart, flower, cross"
    
    prompt_db += '''


                    <root>
                    <round>
                    <user>
                    Leave Objects whose color is not identical to letter A at a line pattern
                    </user>
                    <assistant>
                    {"type" : "pattern:line", "obj_ids" : [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13], "position_pixel" : [None, None, None], "rotation" : [None, None, None]}
                    </assistant>
                    </round>

                    <round>
                    <user>
                    Place Objects whose color is different from cross on a circle pattern
                    </user>
                    <assistant>
                    {"type" : "pattern:circle", "obj_ids" : [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "position_pixel" : [None, None, None], "rotation" : [None, None, None]}
                    </assistant>
                    </round>
                    </root>

                    <round>
                    <user>
                    Leave Objects whose color is same as triangle on a line pattern
                    </user>
                    <assistant>
                    {"type" : "pattern:line", "obj_ids" : [3, 6, 11], "position_pixel" : [None, None, None], "rotation" : [None, None, None]}
                    </assistant>
                    </round>
                    </root>

                 '''
    fp = open("/media/exx/T7 Shield/ICLR23/LGMCTS-D/nl-prompts.txt", "r")
    prompts = []
    for line in fp.readlines():
        prompts.append(line.strip())
    fp.close()
    chatgpt = ChatGPTAPI(prompt_db)
    ret  = chatgpt.chat(str_msg=prompts)
    for res in ret[0]:
        print(res)
    pass
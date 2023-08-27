import time
from typing import List


def user_input(end_token: List[str]) -> str:
    """Talk with LLM"""
    print(">> User: ", end="")
    lines = []
    while True:
        line = input()
        # if this line is endwith one of the end token, stop
        for token in end_token:
            if line.endswith(token):
                content = line.split(token)[0]
                if content:
                    lines.append(content)
                return ("\n".join(lines)).strip()
        lines.append(line)


def print_type_indicator(agent_name="LLM"):
    # print typing indicator
    print(f"* {agent_name} is typing...", end="", flush=True)
    for i in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()

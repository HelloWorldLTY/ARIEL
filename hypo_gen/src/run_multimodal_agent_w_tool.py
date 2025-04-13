import os
import re
import sys
import srsly
from tqdm import tqdm

sys.path.insert(0, "../")

from typing import List, Dict, Callable
from collections import OrderedDict
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from langchain.agents import Tool
from src.agents.dialogue_agent_with_tool import DialogueAgentWithTools
from src.agents.dialogue_agent import DialogueAgent
from src.environment import DialogueSimulator
from src.prompts import prompt_critic, prompt_analyst, prompt_engineer, prompt_scientist, prompt_env
from src.tools.pubmed import PubMedAPIWrapper

pubmed_api = PubMedAPIWrapper(mindate="2024/01/01",
                              maxdate="2025/01/01",
                              top_k_results=3,
                              doc_content_chars_max=2000)
# arxiv_api = ArxivAPIWrapper(mindate="20000101", maxdate="20230101", top_k_results=10, doc_content_chars_max=8000)

tools = [
    Tool(
        name="pubmed",
        func=pubmed_api.run,
        description=
        "Searches articles on pubmed.org. Input multiple related topics or keywords. Avoid hypotheses. Input: string."
    ),
]

def main(background,image_id):
    names = OrderedDict({
        "analyst": {
            "name": "Analyst",
            "system_message": prompt_analyst,
            "tools": None
        },
        "engineer": {
            "name": "Engineer",
            "system_message": prompt_engineer,
            "tools": tools
        },
        "scientist": {
            "name": "Scientist",
            "system_message": prompt_scientist,
            "tools": None
        },
        "critic": {
            "name": "Critic",
            "system_message": prompt_critic,
            "tools": None
        }
    })

    # we set `top_k_results`=2 as part of the `tool_kwargs` to prevent results from overflowing the context limit
    agents = [
        DialogueAgent(name=names["analyst"]["name"],
                      system_message=SystemMessage(
                          content=names["analyst"]["system_message"]),
                      model=ChatOpenAI(model_name="o1", temperature=1.0)),
        DialogueAgentWithTools(
            name=names["engineer"]["name"],
            system_message=SystemMessage(
                content=names["engineer"]["system_message"]),
            model=ChatOpenAI(model_name="o1", temperature=1.0),
            tools=names["engineer"]["tools"],
        ),
        DialogueAgent(name=names["scientist"]["name"],
                      system_message=SystemMessage(
                          content=names["scientist"]["system_message"]),
                      model=ChatOpenAI(model_name="o1", temperature=1.0)),
        DialogueAgent(name=names["critic"]["name"],
                      system_message=SystemMessage(
                          content=names["critic"]["system_message"]),
                      model=ChatOpenAI(model_name="o1", temperature=1.0))
    ]

    def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
        idx = (step) % len(agents)
        return idx

    conversation_description = prompt_env.format(background=background)
    
    if image_id != 9:
        conversation_description = HumanMessage(
        content=[
            {"type": "text", "text": conversation_description},
            {"type": "image_url", "image_url": {"url": f"../../figure_dict/fig_test_{image_id}.png"}},
        ],
    )
    
    else:
        conversation_description = HumanMessage(
        content=[
            {"type": "text", "text": conversation_description},
            {"type": "image_url", "image_url": {"url": f"../../figure_dict/fig_test_{image_id}.jpg"}},
        ],
    )

    simulator = DialogueSimulator(agents=agents,
                                  selection_function=select_next_speaker)
    simulator.reset()
    simulator.inject("Moderator", conversation_description)
    print(f"(Moderator): {conversation_description}")
    print("\n")

    max_iters = 20
    n = 0
    hypothesis = None
    messages = []
    while n < max_iters:
        name, message = simulator.step()
        messages.append({"name": name, "message": message})
        print(f"({name}): {message}")
        print("\n")
        n += 1
        if name == "Critic" and "Final Answer:" in message:
            # Adjusting the regular expression to extract all hypothesis points between the delimiters
            hypothesis = message.split("Final Answer:")[-1].replace("\n", "")
            print(">>>>>>>>>>>>>>>\n", hypothesis, "\n\n")
            break
    return hypothesis, messages
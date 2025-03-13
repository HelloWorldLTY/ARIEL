"""
self correct - verify skeleton
"""

import base64, os, json
import requests, re
from collections import Counter
import pandas as pd
# OpenAI API Key
api_key = ""
prompts_dir = "prompts"
VERIFY_CORRECT = "The proposed solution is correct"
END = "end"
CONTINUE = "continue"
max_iter = 10
model = "o1"
human = "H1"

def call_model(instruction, model, figure_id):
    # OpenAI API Key
    api_key = ""

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Path to your image
    image_path = f"figure_dict/fig_test_{figure_id}.png"

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    print(model)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "seed": 2024,
        # "temperature": 0.7
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # print(response)
    print(response)
    content = response.json()['choices'][0]["message"]["content"]
    print(content)
    print("**********")
    return content


def solve(question, meta_prompt, model, figure_id):
    with open(os.path.join(prompts_dir, "answer_prompt.txt"), "r") as f:
        answer_prompt = " ".join(f.readlines())

    answer_prompt = answer_prompt.format(QUESTION=question)
    print(answer_prompt)

    answer_output = call_model(answer_prompt, model, figure_id)

    return answer_output


def self_correct(question, model_output_round_last, model, figure_id):
    with open(os.path.join(prompts_dir, "self_correct_prompt.txt"), "r") as f:
        self_correct_prompt = " ".join(f.readlines())

    prompt_self_correct = self_correct_prompt.format(question=question, solution_and_analysis=model_output_round_last)

    print(prompt_self_correct)
    self_correct_output = call_model(prompt_self_correct, model, figure_id)

    return self_correct_output


def self_verify(question, model_output_round_last, model, figure_id):

    with open(os.path.join(prompts_dir, "self_verify_prompt.txt"), "r") as f:
        self_verify_prompt = " ".join(f.readlines())

    prompt_self_verify = self_verify_prompt.format(question=question, solution=model_output_round_last)
    print(prompt_self_verify)
    self_verify_output = call_model(prompt_self_verify, model, figure_id)

    return self_verify_output
    # if VERIFY_CORRECT in self_verify_output:
    #     return END
    # else:
    #     return CONTINUE

def load_data(figure_ind):
    with open(f"prompts/questions_{figure_ind}.txt", "r") as f:
        questions = f.readlines()
    with open(f"prompts/meta_prompt_{figure_ind}.txt", "r") as f:
        meta_prompt = "".join(f.readlines())

    return questions, meta_prompt

def read_human_answers(uncertainty):
    # Read the Excel file
    df = pd.read_excel('human_outputs/human_eval_output_correct.xlsx', sheet_name=human)  # If sheet_name is not specified, the first sheet will be read

    df.dropna(subset=['Answer'], inplace=True)
    answers = [[]]
    questions = [[]]

    for index, row in df.iterrows():
        if row['Answer'] != 'Answer':
            if uncertainty:
                answers[-1].append(f"{row['Answer']} I am not sure about this answer.")
            else:
                answers[-1].append(f"{row['Answer']}")
                print("appending")
            questions[-1].append(row['Figure 1'])
        else:
            # if re.match(r'Figure \d+', row['Figure 1']):
            answers.append([])
            questions.append([])

    # print(answers[0])
    #
    # for answer in answers:
    #     print(len(answer))

    return answers, questions

# read_human_answers()

def pipeline(model, figure_ind, figure_id, answers, questions_loaded, solved_once=False):
    if solved_once:
        answers_ind = answers[figure_id-1]
        questions = questions_loaded[figure_id-1]
        questions = [q.split(". ")[1] for q in questions]
        _, meta_prompt = load_data(figure_ind)
    else:
        questions, meta_prompt = load_data(figure_ind)

    with open(f"outputs_human_machine/{human}_{figure_ind}_output_{model}_uncertain_{uncertainty}.jsonl", "w") as f:
        for ind, question in enumerate(questions):
            # print(question)
            question = f"{meta_prompt}\n{question}"
            answer_list = []
            output_list = []

            if not solved_once:
                answer_output = solve(question, meta_prompt, model, figure_id)
                answer_list.append(answer_output)
                output_list.append(answer_output)
            else:
                answer_output = answers_ind[ind]
                answer_list.append(answer_output)
                output_list.append(answer_output)
            for i in range(max_iter):

                self_verify_output = self_verify(question, answer_output, model, figure_id)
                output_list.append(self_verify_output)

                if VERIFY_CORRECT in self_verify_output:
                    break
                else:
                    self_correct_output = self_correct(question, answer_output, model, figure_id)
                    answer_output = self_correct_output
                    answer_list.append(answer_output)
                    output_list.append(answer_output)

            line = {
                "question": question,
                "output_multiple_steps": output_list,
                "answers_multiple_rounds": answer_list,
            }

            f.write(json.dumps(line) + "\n")

    # majority vote
    # freq = Counter(answer_list)
    # final_answer = freq.most_common(1)

from_human = True
uncertainty = True

if from_human:
    answers, questions = read_human_answers(uncertainty=uncertainty)
    # print(len(answers))
    # print(len(questions))

    for answer, question in zip(answers, questions):
        # print(question)
        assert len(answer) == len(question)

    # for i in range(8, 11):
    i = 8
    pipeline(model, f"figure{i}", i, answers, questions, solved_once=True)

else:
    answers = []
    questions = []
    for i in range(1, 6):
        pipeline(model, f"figure{i}", i, answers, questions, solved_once=False)

import json

with open("outputs_human_machine/H2_figure5_output_o1.jsonl", "r") as f:
    for line in f:
        output = json.loads(line)

        print(len(output["output_multiple_steps"]))
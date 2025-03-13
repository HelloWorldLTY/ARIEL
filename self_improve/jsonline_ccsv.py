import json
import csv


def jsonl_to_csv(jsonl_file, csv_file):
    # Open JSONL file and read line by line
    with open(jsonl_file, 'r', encoding='utf-8', errors="replace") as f:
        data = [json.loads(line) for line in f]

    # Ensure JSONL data is a list of dictionaries
    if not data:
        raise ValueError("JSONL file is empty or has incorrect format")

    # Get field names from the first dictionary
    fieldnames = data[0].keys()

    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8', errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")

# Example usage

import os

output_dir = "outputs_human_machine"
output_csv_dir = "outputs_human_machine_csv"
json_files = os.listdir(output_dir)

for json_file in json_files:
    print(json_file)
    file_name = json_file.split(".")[0]
    jsonl_to_csv(os.path.join(output_dir, json_file), os.path.join(output_csv_dir, f"{file_name}.csv"))

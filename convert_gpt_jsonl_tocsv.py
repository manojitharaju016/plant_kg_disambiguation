import json
import jsonlines
import csv

skip_lines = 0
input_file = "./gpt_final_outs/batch_67b043ba00648190ad3376b522acc12d_output.jsonl" # input batch jsonl output file from GPT
output_csv = "./gpt_final_outs/o3mini_test_max_clust_30.csv"

# output CSV file from .json file
with jsonlines.open(input_file) as reader, open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Entity Name", "Group Items"])
    for line in reader:
        # Navigate into the response body
        try:
            body = line["response"]["body"]
            # Get the JSON string from the assistant's "content"
            content_str = body["choices"][0]["message"]["content"]
            data = json.loads(content_str)
            #print(len(data.keys()))
            all_keys = list(data.keys())
            all_values = list(data.values())

            for i in range(len(all_keys)):
                writer.writerow([all_keys[i], json.dumps(all_values[i], ensure_ascii=False)])

        except (KeyError, IndexError, json.JSONDecodeError):
            # Skip lines that don't match the expected structure
            print("Skipping line:", line)
            skip_lines += 1
            print(line["response"]["body"]["choices"][0]["message"]["content"])
            pass
print(f"Skipped {skip_lines} lines")
              

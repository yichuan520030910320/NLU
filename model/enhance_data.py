import json

input_file = "/home/ycwang/NLI/data/snli_1.0/snli_1.0_train.jsonl"
output_file = "/home/ycwang/NLI/dataset/enhance_train.jsonl"

with open(input_file, "r") as in_f, open(output_file, "w") as out_f:
    for line in in_f:
        data = json.loads(line)
        new_data = {
            "text_a": data["sentence1"],
            "text_b": data["sentence2"],
            "label": data["gold_label"],
        }
        out_f.write(json.dumps(new_data) + "\n")
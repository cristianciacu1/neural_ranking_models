import json
import argparse
import pyterrier as pt

# Function to read JSON Lines file and extract 'text' values
def read_json_lines(file_path):
    texts = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            texts.append(data['text'])
    return texts

# Function to create new JSON Lines file with the desired structure
def create_new_json_lines(output_file_path, texts, ques_per_passage):
    pt.init(tqdm="notebook")
    with open(output_file_path, 'w') as outfile:
        dataset = pt.get_dataset(f'irds:beir/{args.dataset}')
        i = 0
        for doc in dataset.get_corpus_iter():
            queries = texts[i * ques_per_passage : (i + 1) * ques_per_passage]
            if queries:
                new_entry = {
                    "id": doc['docno'],
                    "queries": queries
                }
                outfile.write(json.dumps(new_entry) + '\n')
            i += 1

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-d', '--dataset')
parser.add_argument('-q', '--queries')
args = parser.parse_args()

# Example input and output file paths
input_file_path = f'/scratch/cciacu/test_dir/indexing/sprint/deepimpact/all_in_one/datasets/{args.dataset}/{args.dataset}-gen-{args.queries}-queries.jsonl'
output_file_path = f'/scratch/cciacu/test_dir/indexing/sprint/deepimpact/all_in_one/datasets/{args.dataset}/{args.dataset}-gen-{args.queries}-queries-reformatted.jsonl'

# Step 1: Read the JSON Lines file and extract 'text' values
texts = read_json_lines(input_file_path)

# Number of queries per passage
ques_per_passage = int(args.queries)

# Step 2: Create the new JSON Lines file
create_new_json_lines(output_file_path, texts, ques_per_passage)
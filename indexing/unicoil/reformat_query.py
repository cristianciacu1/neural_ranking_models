import argparse
import os
import csv
from sprint_toolkit.inference.utils import load_queries, load_qrels

def load_queries_msmarco(path):
    queries = {}
    with open(path, 'r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        for row in reader:
            queries[row[0]] = row[1]

    return queries


def load_qrels_msmarco(path):
    qrels = {}
    with open(path, 'r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        for row in reader:
            query_id, corpus_id, score = row[0], row[2], int(row[3])
            if query_id not in qrels:
                qrels[query_id] = {corpus_id: score}
            qrels[query_id][corpus_id] = score
    return qrels

def convert_beir_queries(data_dir, topic_split, output_dir=None):
    if output_dir is None:
        output_dir = data_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    queries = load_queries(os.path.join(data_dir, "queries.jsonl"))
    qrels = load_qrels(os.path.join(data_dir, 'qrels', f'{topic_split}.tsv'))
    queries = {qid: queries[qid] for qid in qrels}
    
    with open(os.path.join(output_dir, f'queries-{topic_split}.tsv'), 'w') as fOut:
        for query_id, query in queries.items():
            query = query.replace('\t', '')
            line = '\t'.join([str(query_id), query]) + '\n'
            fOut.write(line)

def convert_msmarco_queries(data_dir, topic_split, output_dir=None):
    if output_dir is None:
        output_dir = data_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    queries = load_queries_msmarco(os.path.join(data_dir, "queries.dev.small.tsv"))
    qrels = load_qrels_msmarco(os.path.join(data_dir, 'qrels.dev.small.tsv'))
    queries = {qid: queries[qid] for qid in qrels}
    
    with open(os.path.join(output_dir, f'queries-{topic_split}.tsv'), 'w') as fOut:
        for query_id, query in queries.items():
            query = query.replace('\t', '')
            line = '\t'.join([str(query_id), query]) + '\n'
            fOut.write(line)

    
def run(original_format, data_dir, topic_split, output_dir=None):
    original_format = original_format.lower()

    if original_format == 'beir':
        convert_beir_queries(data_dir, topic_split, output_dir)
    else:
        convert_msmarco_queries(data_dir, topic_split, output_dir)
    
    print(f'{__name__}: Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_format')
    parser.add_argument('--data_dir')
    parser.add_argument('--topic_split')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    run(**vars(args))

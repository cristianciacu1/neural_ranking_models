import torch
import pyterrier as pt
from pathlib import Path
from pyterrier.measures import RR, nDCG, MAP, R, MRR
from fast_forward.encoder import TCTColBERTQueryEncoder
from fast_forward import Mode
from custom_ff_disk import OnDiskIndex
from fast_forward.util.pyterrier import FFScore, FFInterpolate
import os
import pandas as pd
from tqdm import tqdm
import argparse

## How to run
# 0. conda activate rp-splade

# To get the TREC file of a run, use the command below
# python evaluate_performance.py -i 0 -s true

# To evaluate the models on a dataset, use the command below
# python evaluate_performance.py -i 0

class SwapQueries(pt.Transformer):
    def __init__(self):
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'query_0' in df:
            df_new = df.copy()
            df_new["query_copy"] = df_new["query"]
            df_new["query"] = df_new["query_0"]
            df_new["query_0"] = df_new["query_copy"]
            df_new = df_new.drop(columns=["query_copy"])
            return df_new
        return df
    

import ast
class DoNothing(pt.Transformer):
    def __init__(self):
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    
class Decoder(pt.Transformer):
    def __init__(self):
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df['docno'] = df['docno'].apply(lambda x: ast.literal_eval(x).decode('utf-8'))
        return df
    

def read_alpha(model_name: str, dataset_name: str):
    with open(f'alpha_tuning/{model_name}/{dataset_name}.txt', 'r') as f:
        for line in f:
            if 'Best alpha' in line:
                splits = line.split(": ")
                return float(splits[1])
        raise Exception("The input file does not match the known structure.")
    
# List of datasets (names, pyterrier compatible) that are going to be used in the evaluations
datasets_names = ['fiqa', 'nfcorpus', 'scifact', 'quora', 'hotpotqa', 'dbpedia', 'fever', 'msmarco-passage']

testset_names = ['irds:beir/fiqa/test', 'irds:beir/nfcorpus/test', 'irds:beir/scifact/test',
                 'irds:beir/quora/test', 'irds:beir/hotpotqa/test', 'irds:beir/dbpedia-entity/test',
                 'irds:beir/fever/test', 'irds:msmarco-passage/trec-dl-2019']

# List of sparse indexes (relative paths from the current directory) that are going to be used in the evaluations
sparse_indexes = ['sparse_indexes/sparse_index_fiqa', 'sparse_indexes/sparse_index_nfcorpus', 
                  'sparse_indexes/sparse_index_scifact', 'sparse_indexes/sparse_index_quora',
                  'sparse_indexes/sparse_index_hotpotqa',
                  'sparse_indexes/sparse_index_dbpedia-entity_with_encoding',
                  'sparse_indexes/sparse_index_fever_with_encoding',
                  'sparse_indexes/sparse_index_msmarco-passage']

dense_indexes = ['dense_indexes/ffindex_fiqa_tct.h5', 'dense_indexes/ffindex_nfcorpus_tct.h5',
                 'dense_indexes/ffindex_scifact_tct.h5', 'dense_indexes/ffindex_quora_tct.h5',
                 'dense_indexes/ffindex_hotpotqa_tct.h5', 'dense_indexes/ffindex_dbpedia-entity_tct.h5',
                 'dense_indexes/ffindex_fever_tct.h5',
                 'dense_indexes/ffindex_msmarco-passage_tct.h5']

def dataframe_chunker(df, chunk_size):
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start:start + chunk_size]

# Transformer name
transformer_name = 'castorini/tct_colbert-msmarco'

def adaptToFFIndexes(run: pd.DataFrame, dataset_name: str, dataset) -> pd.DataFrame:
    unnamed_column = run.iloc[:, 0]

    # Split the column by space
    split_data = unnamed_column.str.split(' ', expand=True)

    # Create a new dataframe with the desired columns
    new_df = split_data[[0, 2, 3, 4]]
    new_df.columns = ['qid', 'docno', 'rank', 'score']

    topics = dataset.get_topics('text')

    header_split = run.columns[0].split(' ')
    new_row = pd.DataFrame({
        'qid': [header_split[0]],
        'docno': [header_split[2]],
        'rank': [header_split[3]],
        'score': [float(header_split[4])]
    })

    df = pd.concat([new_row, new_df]).reset_index(drop=True)
    result = pd.merge(df, topics, on='qid', how='left')

    result['score'] = result['score'].astype(float)

    return result

def main():
    # Dual-encoder architecture
    q_encoder = TCTColBERTQueryEncoder(transformer_name)

    # Initialize PyTerrier
    if not pt.started():
        pt.init(tqdm="notebook")

    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('-i', '--index')
    parser.add_argument('-s', '--significance')
    args = parser.parse_args()

    index = int(args.index)
    significance = True if args.significance is not None else False
    print(f"Significance: {significance}")

    dense_index = dense_indexes[index]
    sparse_index = sparse_indexes[index]
    dataset_name = datasets_names[index]
    testset_name = testset_names[index]

    import local_pyt_splade
    factory = local_pyt_splade.SpladeFactory()

    # Load ffindex for the current dataset
    ff_index = OnDiskIndex.load(
        Path(dense_index).resolve(), query_encoder=q_encoder, mode=Mode.MAXP
    )

    # ff_index = ff_index.to_memory()

    # Initialize Fast-Forward Indexes
    ff_score = FFScore(ff_index)

    # Test set
    testset = pt.get_dataset(testset_name)

    # Load sparse indexes
    index_ref = pt.IndexFactory.of(Path(sparse_index).resolve().as_posix())
    deepct_index_ref = pt.IndexFactory.of(Path(sparse_index + "_deepct").resolve().as_posix())
    splade_index_ref = pt.IndexFactory.of(Path(sparse_index + "_splade").resolve().as_posix())

    # Retrieval models
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25") % 1000
    tf_idf = pt.BatchRetrieve(index_ref, wmodel="TF_IDF") % 1000
    deep_ct = pt.BatchRetrieve(deepct_index_ref, wmodel="BM25") % 1000
    splade = factory.query() >> pt.BatchRetrieve(splade_index_ref, wmodel="Tf") % 1000

    # Alpha values
    alpha_bm25 = read_alpha('bm25', dataset_name)
    alpha_tfidf = read_alpha('tf_idf', dataset_name)
    alpha_deepct = read_alpha('deepct', dataset_name)
    alpha_splade = read_alpha('splade', dataset_name)

    if dataset_name in ['dbpedia', 'fever']:
        ff_bm25 = bm25 % 1000 >> Decoder() >> ff_score >> FFInterpolate(alpha_bm25)
        ff_tf_idf = tf_idf % 1000 >> Decoder() >> ff_score >> FFInterpolate(alpha_tfidf)
        ff_deepct = deep_ct % 1000 >> Decoder() >> SwapQueries() >> ff_score >> FFInterpolate(alpha_deepct)
        ff_splade = splade % 1000 >> Decoder() >> SwapQueries() >> ff_score >> FFInterpolate(alpha_splade)
    else:
        ff_bm25 = bm25 % 1000 >> ff_score >> FFInterpolate(alpha_tfidf)
        ff_tf_idf = tf_idf % 1000 >> ff_score >> FFInterpolate(alpha_tfidf)
        ff_deepct = deep_ct % 1000 >> SwapQueries() >> ff_score >> FFInterpolate(alpha_deepct)
        ff_splade = splade % 1000 >> SwapQueries() >> ff_score >> FFInterpolate(alpha_splade)

    models = [ff_bm25, ff_tf_idf, ff_deepct, ff_splade]
    models_name = ['BM25', 'TF-IDF', 'DeepCT', 'Splade', 'unicoil', 'deepimpact']
    
    # The commented lines show the setup if TREC runs are evaluated
    # models = [pt.io.read_results(
    #     Path(f'trec_runs/BM25_{dataset_name}.trec').resolve().as_posix()),
    #     pt.io.read_results(
    #     Path(f'trec_runs/TF-IDF_{dataset_name}.trec').resolve().as_posix()),
    #     pt.io.read_results(
    #     Path(f'trec_runs/DeepCT_{dataset_name}.trec').resolve().as_posix()),
    #     pt.io.read_results(
    #     Path(f'trec_runs/Splade_{dataset_name}.trec').resolve().as_posix()),
    #     pt.io.read_results(
    #     Path(f'trec_runs/unicoil_{dataset_name}.trec').resolve().as_posix()),
    #     pt.io.read_results(
    #     Path(f'trec_runs/deepimpact_{dataset_name}.trec').resolve().as_posix())]

    write_to_path = f'performance/{dataset_name}.csv'

    eval_metrics = [R@1000, MRR@10, MAP@1000, nDCG@10]
    if dataset_name == 'msmarco-passage':
        eval_metrics = [R@1000, MRR(rel=2)@10, MAP(rel=2)@1000, nDCG@10]

    with tqdm(total=len(models), desc=f"{dataset_name}", unit="model") as pbar:
        for model, model_name in tqdm(zip(models, models_name)):
            if not significance:
                curr_df: pd.DataFrame = pt.Experiment(
                    [model],
                    testset.get_topics('text'),
                    testset.get_qrels(),
                    batch_size=100,
                    filter_by_qrels=True,
                    eval_metrics=eval_metrics,
                    names=[f"{model_name}, {dataset_name}"]
                )

                if not os.path.exists(write_to_path):
                    curr_df.to_csv(write_to_path, index=False)
                else:
                    curr_df.to_csv(write_to_path, mode='a', header=False, index=False)
            else:
                # Compute significance testing
                for queries in dataframe_chunker(testset.get_topics('text'), 100):
                    significance_testing = model(queries)
                    final_significance_testing = pt.model.add_ranks(significance_testing)
                    pt.io.write_results(final_significance_testing,
                                        f'trec_runs/{model_name}_{dataset_name}.trec', append=True)

            pbar.update(1)
       
    try:
        alpha_unicoil = read_alpha('unicoil', dataset_name)
    except:
        alpha_unicoil = None
    
    try:
        alpha_deepimpact = read_alpha('deepimpact', dataset_name)
    except:
        alpha_deepimpact = None
    
    alphas = [alpha_unicoil, alpha_deepimpact]
    models = ['unicoil', 'deepimpact']

    with tqdm(total=len(models), desc=f"{dataset_name}", unit="model") as pbar:
        for model_name, alpha in zip(models, alphas):
            if alpha is None:
                print(f'Pair ({dataset_name}, {model_name}) was skipped as alpha is missing.')
                continue

            if dataset_name == 'dbpedia':
                path_to_read_from = f'runs/{model_name}_{dataset_name}-entity_test_run.tsv'
            elif dataset_name == 'msmarco-passage':
                path_to_read_from = f'runs/{model_name}_{dataset_name}_trec-dl_run.tsv'
            else:
                path_to_read_from = f'runs/{model_name}_{dataset_name}_test_run.tsv'
                    
            run = pd.read_csv(Path(path_to_read_from).resolve().as_posix(), sep='\t')
            adapted_run = adaptToFFIndexes(run, dataset_name, testset)

            adapted_run['rank'] = pd.to_numeric(adapted_run['rank'], errors='coerce')

            model = DoNothing() % 1000 >> ff_score >> FFInterpolate(alpha)

            if not significance:
                curr_df = pt.Experiment(
                    [model],
                    testset.get_topics('text'),
                    testset.get_qrels(),
                    batch_size=100,
                    filter_by_qrels=True,
                    eval_metrics=eval_metrics,
                    names=[f'{model_name}, {dataset_name}']
                )

                if not os.path.exists(write_to_path):
                    curr_df.to_csv(write_to_path, index=False)
                else:
                    curr_df.to_csv(write_to_path, mode='a', header=False, index=False) 
            else:
                # Compute significance testing
                for queries in dataframe_chunker(testset.get_topics('text'), 100):
                    subset = pd.merge(adapted_run, queries, on='qid', how='left')
                    subset = pt.Transformer.from_df(subset)
                    significance_testing = (subset >> ff_score >> FFInterpolate(alpha))(queries)
                    final_significance_testing = pt.model.add_ranks(significance_testing)
                    pt.io.write_results(final_significance_testing, 
                                        f'trec_runs/{model_name}_{dataset_name}.trec', append=True)

            pbar.update(1)
    print(f"Experiment on {dataset_name} was successfully saved.")

if __name__ == '__main__':
    main()
import pyterrier as pt
from pathlib import Path
from pyterrier.measures import RR, nDCG, MAP, R, MRR
from fast_forward.encoder import TCTColBERTQueryEncoder
from fast_forward import Mode
from custom_ff_disk import OnDiskIndex
from fast_forward.util.pyterrier import FFScore, FFInterpolate
import ast
import pandas as pd
import argparse

## How to run
# 0. conda activate rp-splade

# 1. python hyperparameter_optimization.py -i x
# where x is the index of the desired dataset from the list below

# Dataset names (human readable format)
dataset_names = ['fiqa', 'nfcorpus', 'scifact', 'quora', 'hotpotqa', 'dbpedia', 'fever', 'msmarco-passage']

# List of datasets (names, pyterrier compatible) that are going to be used in the evaluations
eval_dataset_names = ['irds:beir/fiqa/dev', 'irds:beir/nfcorpus/dev', 'irds:beir/scifact/train', 'irds:beir/quora/dev',
                      'irds:beir/hotpotqa/dev', 'irds:beir/dbpedia-entity/dev', 'irds:beir/fever/dev', 'irds:msmarco-passage/dev']

# List of sparse indexes (relative paths from the current directory) that are going to be used in the evaluations
sparse_indexes = ['sparse_indexes/sparse_index_fiqa', 'sparse_indexes/sparse_index_nfcorpus', 
                  'sparse_indexes/sparse_index_scifact', 'sparse_indexes/sparse_index_quora',
                  'sparse_indexes/sparse_index_hotpotqa',
                  'sparse_indexes/sparse_index_dbpedia-entity_with_encoding',
                  'sparse_indexes/sparse_index_fever_with_encoding',
                  'sparse_indexes/sparse_index_msmarco-passage']

# List of dense indexes (relative paths from the current directory) that are going to be used in the evaluations
dense_indexes = ['dense_indexes/ffindex_fiqa_tct.h5', 'dense_indexes/ffindex_nfcorpus_tct.h5',
                 'dense_indexes/ffindex_scifact_tct.h5', 'dense_indexes/ffindex_quora_tct.h5',
                 'dense_indexes/ffindex_hotpotqa_tct.h5', 'dense_indexes/ffindex_dbpedia-entity_tct.h5',
                 'dense_indexes/ffindex_fever_tct.h5',
                 'dense_indexes/ffindex_msmarco-passage_tct.h5']

alphas = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

# Transformer name
transformer_name = 'castorini/tct_colbert-msmarco'

# Dual-encoder architecture
q_encoder = TCTColBERTQueryEncoder(transformer_name)

class Decode(pt.Transformer):
    def __init__(self):
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df['docno'] = df['docno'].apply(lambda x:  ast.literal_eval(x).decode('utf-8'))
        return df

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
    
class DoNothing(pt.Transformer):
    def __init__(self):
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

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

def write_to_file(model_name, model, dataset_name, evalset, topics, qrels, alphas, ff_score, what_to_optimize='AP@1000'):
    import os.path
    if os.path.isfile(Path(f"alpha_tuning/{model_name}/{dataset_name}.txt").resolve().as_posix()):
        print(f"Skipped 'alpha_tuning/{model_name}/{dataset_name}.txt'")
        return
    best_alpha = -1
    best_score = -1

    eval_metrics = [R@1000, MRR@10, MAP@1000, nDCG@10]

    from tqdm import tqdm
    with tqdm(total=len(alphas), desc=f"{model_name}, {dataset_name}", unit="alpha") as pbar:
        for alpha in alphas:
            if model_name == 'unicoil' or model_name == 'deepimpact':
                # Edge case with scifact which uses the train set instead of the dev set.
                split = 'train' if 'scifact' in dataset_name else 'dev'
                if dataset_name == 'dbpedia':
                    path_to_read_from = f'runs/{model_name}_{dataset_name}-entity_{split}_run.tsv'
                else:
                    path_to_read_from = f'runs/{model_name}_{dataset_name}_{split}_run.tsv'
                
                run = pd.read_csv(Path(path_to_read_from).resolve().as_posix(), sep='\t')
                adapted_run = adaptToFFIndexes(run, dataset_name, evalset)
                adapted_run['rank'] = pd.to_numeric(adapted_run['rank'], errors='coerce')
                adapted_run = adapted_run.merge(topics[['qid']], on='qid')
                qrels = qrels.merge(topics[['qid']], on='qid')
                df = pt.Experiment(
                    [DoNothing() % 1000 >> ff_score >> FFInterpolate(alpha)],
                    adapted_run,
                    qrels,
                    batch_size=100,
                    filter_by_qrels=True,
                    eval_metrics=eval_metrics,
                    names=[f'{model_name}, {dataset_name}']
                )
            else:
                if 'fever' in dataset_name or 'dbpedia' in dataset_name:
                    transformer = [model >> Decode() >> SwapQueries() >> ff_score >>
                                    FFInterpolate(alpha)]
                else:
                    transformer = [model >> SwapQueries() >> ff_score >> FFInterpolate(alpha)]
                
                df: pd.DataFrame = pt.Experiment(
                    transformer,
                    topics,
                    qrels,
                    batch_size=100,
                    filter_by_qrels=True,
                    eval_metrics=eval_metrics,
                    names=[f"{model_name}, {dataset_name}"]
                )

            assert what_to_optimize in df, f"Expected {what_to_optimize} to be in the output, but it was not."

            recall = df['R@1000'].to_list()
            map = df[what_to_optimize].to_list()
            mrr = df['RR@10'].to_list()
            ndcg = df['nDCG@10'].to_list()
            
            assert len(map) == 1, "The experiment has more models than expected."

            recall = float(recall[0])
            map = float(map[0])
            mrr = float(mrr[0])
            ndcg = float(ndcg[0])

            with open(f"alpha_tuning/{model_name}/{dataset_name}.txt", "a") as file1:
                file1.write(f"Alpha: {str(alpha)} \n")
                file1.write(f"R@1000: {str(recall)} \n")
                file1.write(f"{what_to_optimize}: {str(map)} \n")
                file1.write(f"MRR@10: {str(mrr)} \n")
                file1.write(f"ndcg@10: {str(ndcg)} \n\n")

            if map > best_score:
                best_score = map
                best_alpha = alpha

            pbar.update(1)
    with open(f"alpha_tuning/{model_name}/{dataset_name}.txt", "a") as file1:
        file1.write(f"\nBest alpha: {str(best_alpha)} \n")
        file1.write(f"Best score: {str(best_score)}")

def main():
    if not pt.started():
        pt.init(tqdm="notebook")

    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

    parser.add_argument('-i', '--index')
    args = parser.parse_args()

    index = int(args.index)

    dense_index = dense_indexes[index]
    sparse_index = sparse_indexes[index]
    dataset_name = dataset_names[index]
    eval_dataset_name = eval_dataset_names[index]

    ff_index = OnDiskIndex.load(
        Path(dense_index).resolve(), query_encoder=q_encoder, mode=Mode.MAXP
    )

    # Initialize Fast-Forward Indexes
    ff_score = FFScore(ff_index)

    evalset = pt.get_dataset(eval_dataset_name)

    topics: pd.DataFrame = evalset.get_topics('text')
    qrels: pd.DataFrame = evalset.get_qrels()
    if topics.size > 1000:
        number_of_samples = 500
        topics = topics.sample(n=number_of_samples)
        qrels = qrels.merge(topics[['qid']], on='qid')
        print(f"Note: for {dataset_name}, {number_of_samples} samples were randomly selected from the topics.")
    
    # Load sparse indexes and run experiments
    try:
        index_ref = pt.IndexFactory.of(Path(sparse_index).resolve().as_posix())
        bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
        tf_idf = pt.BatchRetrieve(index_ref, wmodel="TF_IDF")
        write_to_file('bm25', bm25, dataset_name, evalset, topics, qrels, alphas, ff_score)
        write_to_file('tf_idf', tf_idf, dataset_name, evalset, topics, qrels, alphas, ff_score)
    except Exception as e:
        print(e)

    try:    
        deepct_index_ref = pt.IndexFactory.of(Path(sparse_index + "_deepct").resolve().as_posix())
        deep_ct = pt.BatchRetrieve(deepct_index_ref, wmodel="BM25")
        write_to_file('deepct', deep_ct, dataset_name, evalset, topics, qrels, alphas, ff_score)
    except Exception as e:
        print(e)
    
    try:
        # Splade
        splade_index_ref = pt.IndexFactory.of(Path(sparse_index + "_splade").resolve().as_posix())
        import pyt_splade
        factory = pyt_splade.SpladeFactory()
        splade = factory.query() >> pt.BatchRetrieve(splade_index_ref, wmodel="Tf")
        write_to_file('splade', splade, dataset_name, evalset, topics, qrels, alphas, ff_score)
    except Exception as e:
        print(e)

    try:
        write_to_file('unicoil', None, dataset_name, evalset, topics, qrels, alphas, ff_score)
    except Exception as e:
        print(e)

    try:
        write_to_file('deepimpact', None, dataset_name, evalset, topics, qrels, alphas, ff_score)
    except Exception as e:
        print(e)

if __name__=='__main__':
    main()
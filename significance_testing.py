import pyterrier as pt
from pathlib import Path
import argparse

from ranx import compare, Run, Qrels

datasets_names = ['fiqa', 'nfcorpus', 'scifact', 'quora', 'hotpotqa', 'dbpedia',
                  'fever', 'msmarco-passage']
testset_names = ['irds:beir/fiqa/test', 'irds:beir/nfcorpus/test', 'irds:beir/scifact/test',
                 'irds:beir/quora/test', 'irds:beir/hotpotqa/test', 'irds:beir/dbpedia-entity/test',
                 'irds:beir/fever/test', 'irds:msmarco-passage/trec-dl-2019']

def main():
    # Initialize PyTerrier
    if not pt.started():
        pt.init(tqdm="notebook")

    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('-i', '--index')
    parser.add_argument('--ff', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    index = int(args.index)
    ff = args.ff

    input_dir = 'trec_runs' if ff else 'trec_runs_first_stage'

    bm25 = Run.from_file(Path(f"{input_dir}/BM25_{datasets_names[index]}.trec").resolve().as_posix()) 
    tf_idf = Run.from_file(Path(f"{input_dir}/TF-IDF_{datasets_names[index]}.trec").resolve().as_posix())
    deepct = Run.from_file(Path(f"{input_dir}/DeepCT_{datasets_names[index]}.trec").resolve().as_posix())
    splade = Run.from_file(Path(f"{input_dir}/Splade_{datasets_names[index]}.trec").resolve().as_posix())
    unicoil = Run.from_file(
        Path(f"{input_dir}/unicoil_{datasets_names[index]}.trec").resolve().as_posix())
    deepimpact = Run.from_file(
        Path(f"{input_dir}/deepimpact_{datasets_names[index]}.trec").resolve().as_posix())

    qrels = pt.get_dataset(testset_names[index]).get_qrels()

    qrels = Qrels.from_df(qrels, q_id_col='qid', doc_id_col='docno', score_col='label')
    runs = [bm25, tf_idf, deepct, deepimpact, unicoil, splade]

    metrics = ["recall@1000", "map@1000", "mrr@10", "ndcg@10"]
    if not ff:
        metrics = ["recall@1000", "ndcg@10"]

    report = compare(qrels, runs, metrics=metrics,
                      max_p=0.05, stat_test='student', make_comparable=True)

    print(report)

    output_path = f"significance_tests/{datasets_names[index]}_ff.json" if ff else (
        f"significance_tests/{datasets_names[index]}_first_stage.json")

    report.save(Path(output_path))


if __name__ == '__main__':
    main()


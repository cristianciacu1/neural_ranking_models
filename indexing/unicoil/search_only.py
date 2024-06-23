import pyterrier as pt
import pandas as pd

def reformat_query(dataset_name, topic_split):
    if not pt.started():
        pt.init(tqdm="notebook")
    
    dataset = pt.get_dataset(dataset_name)
    queries = dataset.get_topics('text')
    
    queries.to_csv(f'queries-{topic_split}.tsv', sep='\t', header=False, index=False)

def main():
    
    # from sprint_toolkit.inference import search
    topic_split = 'trec-dl'
    tsv_queries_path = f'/Users/cciacu/Desktop/school/rp/experiments/indexing/unicoil/msmarco/queries-{topic_split}.tsv'
    encoder_name = 'unicoil'
    query_ckpt = 'castorini/unicoil-noexp-msmarco-passage'
    output_dir_index = '/Users/cciacu/Desktop/school/rp/experiments/indexing/unicoil/msmarco/index'
    output_path_search = f'/Users/cciacu/Desktop/school/rp/experiments/indexing/unicoil/msmarco/search/unicoil_msmarco_{topic_split}_run.tsv'
    hits = 1000
    batch_size = 32
    nprocs = 12
    output_format_search = 'trec'
    min_idf = -1

    # 1.
    # reformat_query('irds:msmarco-passage/trec-dl-2019', topic_split)

    from sprint_toolkit.inference import search

    # 2.
    search.run(
        topics=tsv_queries_path,
        encoder_name=encoder_name,
        ckpt_name=query_ckpt,
        index=output_dir_index,
        output=output_path_search,
        impact=True,
        hits=hits + 1,
        batch_size=batch_size,
        threads=nprocs,
        output_format=output_format_search,
        min_idf=min_idf,
    )

if __name__ == '__main__':
    main()
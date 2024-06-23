import pyterrier as pt
import os
from pathlib import Path
import time

def docs_iter(dataset):
    for d in dataset.get_corpus_iter():
        yield {'docno': str(d['docno'].encode('utf-8')), 'text': d['text']}

def main():
    start_time = time.time()

    if not pt.started():
        pt.init(tqdm="notebook")
    
    DATASET_NAME = 'irds:beir/fever'
    INDEX_NAME = 'sparse_index_fever_with_encoding'
    MAX_ID_LENGTH = 260

    # Get the current dataset from PyTerrier
    dataset = pt.get_dataset(DATASET_NAME)

    if not os.path.isdir(INDEX_NAME):
        print(f"Compute index for {DATASET_NAME}.")
        indexer = pt.IterDictIndexer(index_path=Path(INDEX_NAME).resolve().as_posix(), meta={'docno': MAX_ID_LENGTH})
        # TODO: check if `fields=["test"]` works for any dataset.
        indexer.index(docs_iter(dataset), fields=["text"])
    
    print(f'Finished indexing. Time elapsed: {time.time() - start_time}')


if __name__ == '__main__':
    main()

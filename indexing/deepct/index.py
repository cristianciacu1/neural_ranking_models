from pyterrier_deepct import DeepCT, Toks2Text
import pyterrier as pt
from pathlib import Path

def docs_iter(dataset):
    for d in dataset.get_corpus_iter():
        yield {'docno': str(d['docno'].encode('utf-8')), 'text': d['text']}

def main():
    if not pt.started():
        pt.init(tqdm="notebook")

    INDEX_NAME = "fever_with_encoding"
    DATASET_NAME = "irds:beir/fever"
    MAX_ID_LENGTH = 260

    deepct = DeepCT(batch_size=16)
    dataset = pt.get_dataset(DATASET_NAME)
    indexer = deepct >> Toks2Text() >> pt.IterDictIndexer(
        index_path=Path("sparse_index_" + INDEX_NAME + "_deepct").resolve().as_posix(), meta={'docno': MAX_ID_LENGTH})
    indexer.index(docs_iter(dataset))


if __name__ == '__main__':
    main()
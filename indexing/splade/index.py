import pyterrier as pt
from pathlib import Path

def docs_iter(dataset):
    for d in dataset.get_corpus_iter():
        yield {'docno': str(d['docno'].encode('utf-8')), 'text': d['text']}

def main():
    if not pt.started():
        pt.init(tqdm="notebook", version='snapshot')

    import pyt_splade

    INDEX_NAME = "fever_with_encoding"
    DATASET_NAME = "irds:beir/fever"
    MAX_ID_LENGTH = 260

    factory = pyt_splade.SpladeFactory()
    doc_encoder = factory.indexing()
    dataset = pt.get_dataset(DATASET_NAME)

    indexer = pt.IterDictIndexer(Path('sparse_index_' + INDEX_NAME + '_splade').resolve().as_posix(), overwrite=True, meta={'docno': MAX_ID_LENGTH})
    indexer.setProperty("termpipelines", "")
    indexer.setProperty("tokeniser", "WhitespaceTokeniser")

    indxr_pipe = (doc_encoder >> pyt_splade.toks2doc() >> indexer)
    indxr_pipe.index(docs_iter(dataset), batch_size=32)


if __name__ == '__main__':
    main()
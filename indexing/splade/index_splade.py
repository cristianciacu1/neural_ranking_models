import pyterrier as pt
from pathlib import Path

def main():
    if not pt.started():
        pt.init(tqdm="notebook", version='snapshot')

    import pyt_splade

    INDEX_NAME = "nfcorpus"
    DATASET_NAME = "irds:beir/nfcorpus"
    MAX_ID_LENGTH = 9

    factory = pyt_splade.SpladeFactory()
    doc_encoder = factory.indexing()
    dataset = pt.get_dataset(DATASET_NAME)

    indexer = pt.IterDictIndexer(Path('../../sparse_indexes/sparse_index_' + INDEX_NAME + '_splade').resolve().as_posix(), overwrite=True, meta={'docno': MAX_ID_LENGTH})
    indexer.setProperty("termpipelines", "")
    indexer.setProperty("tokeniser", "WhitespaceTokeniser")

    indxr_pipe = (doc_encoder >> pyt_splade.toks2doc() >> indexer)
    indxr_pipe.index(dataset.get_corpus_iter(), batch_size=128)


if __name__ == '__main__':
    main()


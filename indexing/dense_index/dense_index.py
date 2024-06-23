import pyterrier as pt
from pathlib import Path
from fast_forward.encoder import TCTColBERTQueryEncoder, TCTColBERTDocumentEncoder
import torch
from fast_forward import OnDiskIndex, Mode, Indexer
import time

def docs_iter_encode(dataset):
    for d in dataset.get_corpus_iter():
        yield {"doc_id": d["docno"], "text": d["text"]}

def docs_iter(dataset):
    for d in dataset.get_corpus_iter():
        yield {"doc_id": d["docno"].encode('utf-8'), "text": d["text"]}


def main():
    if not pt.started():
        pt.init(tqdm="notebook")

    DATASET_NAME = 'irds:beir/msmarco'
    INDEX_NAME = 'msmarco-beir'
    MAX_ID_LENGTH = 8

    print('Started indexing')
    start_time = time.time()

    dataset = pt.get_dataset(DATASET_NAME)

    q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")
    d_encoder = TCTColBERTDocumentEncoder(
        "castorini/tct_colbert-msmarco",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    ff_index = OnDiskIndex(
        Path("ffindex_" + INDEX_NAME + "_tct.h5"), dim=768, query_encoder=q_encoder, mode=Mode.MAXP, max_id_length=MAX_ID_LENGTH
    )

    ff_indexer = Indexer(ff_index, d_encoder, batch_size=16)
    ff_indexer.index_dicts(docs_iter(dataset))

    print(f'Finished indexing. Time elapsed: {time.time() - start_time}')
    

if __name__ == '__main__':
    main()

### Neural Ranking Models
This code was submitted together with my Bachelor Thesis "The Impact of the Retrieval Stage in Interpolation-based Re-Ranking". In my the thesis, I explored different sparse retrievers, retrievers that employ standard statistical term-weighting and retrievers that employ neural networks for term-weighting and document expansion, on various datasets. Furthermore, I used these models in Fast-Forward Indexes, an interpolation-based re-ranking setting, with the goal to understand the impact of the retrieval stage on the overall performance of the framework.

#### Dependencies
Running the experiments only requires having Conda installed on your machine. Given that Conda is installed, the following three Conda environments need to be created:
- py_deepct (used for indexing datasets with DeepCT)
- rp-splade (used for indexing datasets with SPLADE, and it is also used for all the evaluations)
- sprint_env (used for indexing datasets with uniCOIL and DeepImpact)

Furthermore, it is important to note that indexing using either uniCOIL or DeepImpact requires a CUDA-compatible GPU on the machine. This is a hard requirement of the SPRINT library. Yet, the other indexers do not require a GPU, but indexing documents for a neural retriever on a CPU is slow. Hence, using a GPU is recommended for indexing. On the other hand, evaluations on all models, except for the bi-encoder-based ones, is rather fast, while for the bi-encoder models, a GPU acceleration would be beneficial.

#### Datasets
The datasets used for the evaluations were retrieved from IR-Datasets(https://ir-datasets.com/) using the functionality provided by PyTerrier (https://github.com/terrier-org/pyterrier). 


#### Created indexes
At the moment of submitting the thesis, the created indexes could not be made publicly available due to the short available time and the large size of the indexes (around 400GB). Yet, in the near future, the indexes might be uploaded to https://www.4tu.nl/en/ to help other researchers.


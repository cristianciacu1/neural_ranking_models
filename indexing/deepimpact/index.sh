####
export dataset=$1
####

export stage=index

export encoder_name='deepimpact'
export encoder_ckpt_name='deepimpact_d2q'
export data_name=beir_$dataset

export quantization='b8'

export long_idenitifer=$data_name-$encoder_ckpt_name-$quantization

export collection_dir=$long_idenitifer/collection
export output_dir=$long_idenitifer/$stage

export log_name=$stage.$long_idenitifer.log

srun time -v python -m sprint_toolkit.inference.$stage \
    -collection JsonVectorCollection \
    -input $collection_dir \
    -index $output_dir \
    -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
    -threads 12 > $log_name
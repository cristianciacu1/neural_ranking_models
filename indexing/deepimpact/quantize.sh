####
export dataset=$1
####

export stage=quantize

export encoder_name='deepimpact'
export encoder_ckpt_name='deepimpact_d2q'
export data_name=beir_$dataset

export quantization_from='float'
export quantization_to='b8'

export quantization_method='range-nbits'
export original_score_range=-1
export quantization_nbits=8

export long_idenitifer_from=$data_name-$encoder_ckpt_name-$quantization_from
export long_idenitifer_to=$data_name-$encoder_ckpt_name-$quantization_to

export collection_dir=$long_idenitifer_from/collection
export output_dir=$long_idenitifer_to/collection

export log_name=$stage.$long_idenitifer_to.log

srun time -v python -m sprint_toolkit.inference.$stage \
    --collection_dir $collection_dir \
    --output_dir $output_dir \
    --method $quantization_method \
    --original_score_range $original_score_range \
    --quantization_nbits $quantization_nbits \
    --nprocs 12 > $log_name
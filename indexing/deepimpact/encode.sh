####
export dataset=$1
####

export stage='encode'

export encoder_name='deepimpact'
export encoder_ckpt_name='deepimpact_d2q'
export data_name=beir_$dataset
export quantization='float'

export ckpt_name='/scratch/cciacu/test_dir/indexing/sprint/deepimpact/deepimpact-bert-base'
export long_idenitifer=$data_name-$encoder_ckpt_name-$quantization

export output_dir_encode=$long_idenitifer/collection
export log_name=$stage.$long_idenitifer.log
export gpus="0"

export preprocessed_documents_path=/scratch/cciacu/test_dir/indexing/sprint/deepimpact/all_in_one/datasets/$dataset/deepimpact

srun time -v python -m sprint_toolkit.inference.$stage \
    --encoder_name $encoder_name \
    --ckpt_name $ckpt_name \
    --data_name $data_name \
    --data_dir $preprocessed_documents_path \
    --gpus $gpus \
    --output_dir $output_dir_encode \
    > $log_name
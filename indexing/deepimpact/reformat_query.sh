####
export dataset=$1
export topic_split=$2
####

export stage=reformat_query

export encoder_name='deepimpact'
export encoder_ckpt_name='deepimpact_d2q'
export data_name=beir_$dataset

export data_dir=/scratch/cciacu/test_dir/indexing/sprint/deepimpact/all_in_one/datasets/$dataset

export long_idenitifer=$data_name
export log_name=$stage.$long_idenitifer.log

srun time -v python -m sprint_toolkit.inference.$stage --original_format 'beir' --data_dir $data_dir --topic_split $topic_split
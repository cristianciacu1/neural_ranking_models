####
export dataset=$1
export topic_split=$2
####

export stage=search

export encoder_name='deepimpact'
export encoder_ckpt_name='deepimpact_d2q'
export data_name=beir_$dataset

export quantization='b8'

export ckpt_name='/scratch/cciacu/test_dir/indexing/sprint/deepimpact/deepimpact-bert-base'
export long_idenitifer="$data_name-$encoder_ckpt_name-$quantization"
export log_name=$stage.$long_idenitifer.log
export index_dir=$long_idenitifer/index

export output_format=trec
export output_path=$long_idenitifer/$stage/$output_format-format/$dataset-$topic_split-run.tsv
export data_dir=/scratch/cciacu/test_dir/indexing/sprint/deepimpact/all_in_one/datasets/$dataset
export queries_path=$data_dir/queries-$topic_split.tsv

srun time -v python -m sprint_toolkit.inference.$stage --topics $queries_path --encoder_name $encoder_name --ckpt_name $ckpt_name --index $index_dir --output $output_path --impact --hits 1000 --batch 36 --threads 12 --output-format $output_format > $log_name
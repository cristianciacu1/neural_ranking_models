#####
export data_name='fever'
export topic='test'
export long_idenitifer=/scratch/cciacu/test_dir/indexing/sprint/unicoil/runs/$data_name/test-quantized
#####

export stage=search  # Adapted from the Pyserini README for reproducing uniCOIL on MSMARCO

export encoder_name='unicoil'
export encoder_ckpt_name='unicoil_noexp'
export quantization='b8'

export ckpt_name='castorini/unicoil-noexp-msmarco-passage'
export log_name=$stage.$data_name.log
export index_dir=$long_idenitifer/index

export output_format=trec  # Could also be 'msmarco'. These formats are from Pyserini. 'trec' will keep also the scores
export output_path=$long_idenitifer/$stage/$output_format-format/unicoil_$data_name_$topic_run.tsv
export data_dir=datasets/beir/$data_name
export queries_path=$data_dir/queries-$topic.tsv  # This is the input queries

time -v python -m sprint_inference.reformat_query \
    --original_format 'beir' \
    --data_dir $data_dir \
    --topic_split $topic
    > /scratch/cciacu/test_dir/indexing/sprint/unicoil/runs/fever/test-quantized/reformat.log

time -v python -m sprint_inference.$stage \
    --topics $queries_path \
    --encoder_name $encoder_name \
    --ckpt_name  $ckpt_name \
    --index $index_dir \
    --output $output_path \
    --impact \
    --hits 1000 --batch 36 --threads 12 \
    --output-format $output_format \
    > $log_name

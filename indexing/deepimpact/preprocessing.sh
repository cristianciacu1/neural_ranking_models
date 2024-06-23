####
export dataset=$1
export queries=$2
####

export model_name_or_path="/scratch/cciacu/test_dir/indexing/sprint/deepimpact/deepimpact-bert-base"
export split="test"

export data_path=/scratch/cciacu/test_dir/indexing/sprint/deepimpact/all_in_one/datasets/$dataset
export d2q_filepath=/scratch/cciacu/test_dir/indexing/sprint/deepimpact/all_in_one/datasets/$dataset/$dataset-gen-$queries-queries-reformatted.jsonl
export output_path=/scratch/cciacu/test_dir/indexing/sprint/deepimpact/all_in_one/datasets/$dataset/deepimpact

srun time -v python preprocessing.py \
    --dataset_dir ${data_path} \
    --output_path ${output_path} \
    --d2q_filepath ${d2q_filepath}
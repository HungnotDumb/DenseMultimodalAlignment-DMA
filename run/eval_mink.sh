set -x

exp_dir=$1
config=$2

mkdir -p ${exp_dir}

model_dir=${exp_dir}/model
result_dir=${exp_dir}/result_eval

export PYTHONPATH=.
python -u run/eval_mink.py \
  --conf
# !/usr/bin/env bash
python run_recbole.py --model=CDIB --dataset=ml-100k --train_neg_sample_args=None  --gpu_id=$1
python run_recbole.py --model=CDIB --dataset=beauty --train_neg_sample_args=None  --gpu_id=$1
python run_recbole.py --model=CDIB --dataset=retailrocket-view --train_neg_sample_args=None  --gpu_id=$1
python run_recbole.py --model=CDIB --dataset=sports --train_neg_sample_args=None  --gpu_id=$1
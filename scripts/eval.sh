#!/bin/bash

#model_path="/path/to/projector"
model_path="/root/lht/LLaGA/checkpoints/cora/llaga-vicuna-7b-simteg-2-10-linear-projector_nc_selecttmp"
model_base="/root/autodl-tmp/lht/huggingface_model/lmsys/vicuna-7b-v1.5-16k" #meta-llama/Llama-2-7b-hf
mode="v1" # use 'llaga_llama_2' for llama and "v1" for others
dataset="cora" #test dataset
task="nc" #test task
emb="simteg"
use_hop=2
sample_size=10
template="ND" # or ND
output_path="/root/autodl-tmp/lht/LLaGA/eval_results/vicuna_cora-nc_cora-nc_selecttmp.jsonl" #"/path/to/output"

python eval/eval_pretrain_GNN.py \
--model_path ${model_path} \
--model_base ${model_base} \
--conv_mode  ${mode} \
--dataset ${dataset} \
--pretrained_embedding_type ${emb} \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--answers_file ${output_path} \
--task ${task} \
--cache_dir ../../checkpoint \
--template ${template}
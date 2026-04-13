#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set -x

exp_dir=''
model_path=''
nj=4
gpus="[0,1]"
data_type="shard"  # shard/raw/feat
data=''
num_ref_utts=1
fusion_type='xattn_add'
xattn_heads=4


. tools/parse_options.sh
data_name_array=("train" "dev" "eval")
data_list_path_array=("${data}/train/${data_type}.list" "${data}/dev/${data_type}.list" "${data}/eval/${data_type}.list")
data_scp_path_array=("${data}/train/raw.list" "${data}/dev/raw.list" "${data}/eval/raw.list")
# nj_array=($nj $nj )
# batch_size_array=(1 1) # batch_size of test set must be 1 !!!
# num_workers_array=(1 1)
nj_array=($nj )
batch_size_array=(1) # batch_size of test set must be 1 !!!
num_workers_array=(1)
count=${#data_name_array[@]}


for i in $(seq 0 $(($count - 1))); do
  wavs_num=$(wc -l ${data_scp_path_array[$i]} | awk '{print $1}')
  bash tools/extract_embedding.sh --exp_dir ${exp_dir} \
    --model_path $model_path \
    --data_type ${data_type} \
    --data_list ${data_list_path_array[$i]} \
    --wavs_num ${wavs_num} \
    --store_dir ${data_name_array[$i]} \
    --batch_size ${batch_size_array[$i]} \
    --num_workers ${num_workers_array[$i]} \
    --nj ${nj_array[$i]} \
    --gpus $gpus \
    --num_ref_utts ${num_ref_utts} \
    --fusion_type ${fusion_type} \
    --xattn_heads ${xattn_heads} 

done

wait

echo "Embedding dir is (${exp_dir}/embeddings)."
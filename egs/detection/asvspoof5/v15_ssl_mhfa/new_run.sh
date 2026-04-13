#!/bin/bash
# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2025 Johan Rohdin, Lin Zhang (rohdin@fit.vut.cz, partialspoof@gmail.com)
#           2025 Junyi Peng (pengjy@fit.vut.cz)

#SBATCH --job-name=num_ref_dur #job name
#SBATCH --gpus=1
#SBATCH --exclude=c14,c05
#SBATCH --partition=gpu
# #SBATCH --partition=gpu-a100   #queue
# #SBATCH --account=a100acct
#SBATCH --mail-user="ktan17@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --error=/export/fs05/ktan17/ssl/new_new_file/ASVspoof5.dev.%j.err
#SBATCH --output=/export/fs05/ktan17/ssl/new_new_file/ASVspoof5.dev.%j.out
#SBATCH --array=0-3
# #SBATCH --dependency
#unset PYTHONPATH
#unset PYTHONHOME

echo "[$(date)] Starting Job ID: $SLURM_JOB_ID" 

source ~/.bashrc
echo `date`
echo $SSL_CERT_FILE
ls -l /etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export TMPFIR=/export/fs05/ktan17/tmp
nvidia-smi
hostname
conda activate /home/ktan17/miniconda3/envs/wedefense

#-------------------------------------------------------------------------
#LR=(1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)
#LR=(5e-3 1e-3 5e-4 1e-4)

# LR=(0.1 0.01 0.001 0.0001)
LR=0.001
initial_lr=0.001
echo "initial_lr: $initial_lr"
# final_lr=${LR[$SLURM_ARRAY_TASK_ID]}
# echo "final_lr:$final_lr"
#-------------------------------------------------------------------------
# TRIAL_RATIO=('35235' 3w8_ratio)
# trial_ratio=${TRIAL_RATIO[$SLURM_ARRAY_TASK_ID]}
# echo "trial_ratio:$trial_ratio"

# TRIALS_MODE=('s1' 's2' 's3_T31')
trial_mode='s2'
# trial_mode=${TRIALS_MODE[$SLURM_ARRAY_TASK_ID]}
echo "trial_model:$trial_mode"

# FUSION_TYPE=('xattn_add' 'feat_cat' 'time_cat')
fusion_type='xattn_add'
# fusion_type=${FUSION_TYPE[$SLURM_ARRAY_TASK_ID]}
echo "fusion_type:$fusion_type"

# XATTN_HEADS=(1 2 4 8)
xattn_heads=4
# xattn_heads=${XATTN_HEADS[$SLURM_ARRAY_TASK_ID]}
echo "xattn_heads:$xattn_heads"

NUM_REFS=(4 3 2 1)
# num_ref_utts=1
num_ref_utts=${NUM_REFS[$SLURM_ARRAY_TASK_ID]}
echo "num_ref_utts:$num_ref_utts"

# TARGET_DUR=(2.0 3.0 5.0)
target_dur=2.0
# target_dur=${TARGET_DUR[$SLURM_ARRAY_TASK_ID]}
echo "target_dur:$target_dur"

# NUM_AVG=(2 3 4 5 6 7 8)
num_avg=2
# num_avg=${NUM_AVG[$SLURM_ARRAY_TASK_ID]}
echo "num_avg:$num_avg" # how many models you want to average

# set -x
. ./path.sh || exit 1

stage=6
stop_stage=8
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

python -c "import torch; print('GPU count:', torch.cuda.device_count())"
ASVspoof5_dir=/export/fs05/arts/dataset/ASVspoof5
data=data/hard_pair/ASVspoof5/ratio111/${trial_mode}
# data=data/hard_pair/ASVspoof5/$trial_ratio # data folder

trial=/export/fs05/ktan17/dataset/ASVspoof5/train/hp_3w6_ratio_modular_${trial_mode}.tsv
data_type="raw"  # shard/raw
config=conf/MFA_Conformer.yaml #wespeaker version
# config=conf/MHFA_wav2vec2.yaml
exp_dir=exp/num_refs_${num_ref_utts}
model_init=/export/fs05/ktan17/SASV2_Baseline/pretrained_weight/MFA_Conformer_stage_123.model
gpus="[0]"
checkpoint=
score_norm_method="asnorm"  # asnorm/snorm
top_n=300
early_stop_patience=10
# setup for large margin fine-tuning
lm_config=conf/campplus_lm.yaml

. tools/parse_options.sh || exit 1

#######################################################################################
# Stage 1. Preparing data folder for partialspoof: wav.scp, utt2lab, lab2utt, reco2dur
#######################################################################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  # ./local/prepare_data.sh ${ASVspoof5_dir} ${data}
  ./local/prepare_data_track2.sh ${ASVspoof5_dir} ${data} ${trial}
fi

#######################################################################################
# Stage 2. Preapring shard data for partialspoof and musan/rirs
#######################################################################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."

  cd ${data}
  # ln -s flac_T train
  # ln -s flac_D dev
  # ln -s flac_E_eval eval
  ln -s flac_T_all train
  ln -s flac_D_all dev
  ln -s flac_E_eval_all eval
  cd -
  # We don't use VAD here

  for dset in train;do
  # for dset in train dev eval;do
      if [ $data_type == "shard" ]; then
          python tools/make_shard_list.py --num_utts_per_shard 1000 \
              --num_threads 8 \
              --prefix shards \
              --shuffle \
              ${data}/$dset/wav.scp ${data}/$dset/utt2lab \
              ${data}/$dset/shards ${data}/$dset/shard.list
      else
          python tools/make_raw_list_new.py \
              --num_ref_utts ${num_ref_utts} \
              ${data}/$dset/wav.scp \
              ${data}/$dset/utt2lab \
              ${data}/$dset/spk2utt \
              ${data}/$dset/raw.list
      fi
  done

  #TODO: wespeaker doesn't support multi-channel wavs.
  #MUSAN_dir=/export/fs05/arts/dataset/musan
  #find ${MUSAN_dir} -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > data/musan/wav.scp
  #RIRs_dir=/export/fs05/arts/dataset/RIRS_NOISES/RIRS_NOISES
  #find ${RIRs_dir} -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort > data/rirs/wav.scp
  # Convert all musan data to LMDB. But note that lmdb does not work on NFS!
  # python tools/make_lmdb.py data/musan/wav.scp ${HOME}/local_lmdb/musan/lmdb
  # rsync -av ${HOME}/local_lmdb/musan/lmdb data/musan/lmdb
  # Convert all rirs data to LMDB
  # python tools/make_lmdb.py data/rirs/wav.scp ${HOME}/local_lmdb/rirs/lmdb
  # rsync -av ${HOME}/local_lmdb/rirs/lmdb data/rirs/lmdb
fi

#######################################################################################
# Stage 3. Validation Set Preparation
#######################################################################################
USE_RANDOM_VAL=true  # true/false

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

  VAL_SEED=42
  VAL_NUM=5000                     # Total val samples
  VAL_RATIO_TARGET=1               # 1:8:7
  VAL_RATIO_NON=1
  VAL_RATIO_SPOOF=1

  VAL_DIR=${data}/val
  mkdir -p ${VAL_DIR}
  rm -f ${VAL_DIR}/*

  if [ "${USE_RANDOM_VAL}" = true ]; then
      echo "[VAL] Random sampling ENABLED (seed=${VAL_SEED})"

      python3 tools/make_val_set.py \
          --mode ratio \
          --dev_dir ${data}/dev \
          --val_dir ${VAL_DIR} \
          --val_num ${VAL_NUM} \
          --ratio ${VAL_RATIO_TARGET} ${VAL_RATIO_NON} ${VAL_RATIO_SPOOF} \
          --seed ${VAL_SEED}

      VAL_LIST=${VAL_DIR}/raw.list
      VAL_LAB=${VAL_DIR}/utt2lab

  else
      echo "[VAL] USE_RANDOM_VAL != true → Randomly selecting ${VAL_NUM} utterances"

      python3 tools/make_val_set.py \
          --mode flat \
          --dev_dir ${data}/dev \
          --val_dir ${VAL_DIR} \
          --val_num ${VAL_NUM} \
          --seed ${VAL_SEED}

      VAL_LIST=${VAL_DIR}/raw.list
      VAL_LAB=${VAL_DIR}/utt2lab
  fi

  echo "[VAL] Final val_data  = ${VAL_LIST}"
  echo "[VAL] Final val_label = ${VAL_LAB}"
fi


#######################################################################################
# Stage 4. Training
#######################################################################################
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Start training ..."
  num_gpus=1
  if [[ $(hostname -f) == *fit.vutbr.cz   ]]; then
     gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
  fi
    ##num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  # To avoid the randomly generated port is occuppied.
  while :
  do
    port=$(( (RANDOM % 100) + 29500 ))
    if ! lsof -i:$port >/dev/null; then
      break
    fi
  done

    torchrun --rdzv_backend=c10d --rdzv_endpoint=$(hostname):$((port)) --nnodes=1 --nproc_per_node=$num_gpus \
      wedefense/bin/train_MFA.py --config $config \
        --exp_dir ${exp_dir} \
        --gpus $gpus \
        --num_avg ${num_avg} \
        --data_type "${data_type}" \
        --train_data ${data}/train/${data_type}.list \
        --train_label ${data}/train/utt2lab \
        ${checkpoint:+--checkpoint $checkpoint} \
        --val_data ${data}/val/${data_type}.list \
        --val_label ${data}/val/utt2lab \
        --initial_lr ${initial_lr} \
        --early_stop_patience ${early_stop_patience} \
        --save_original_model False \
        --model_init ${model_init} \
        --num_ref_utts ${num_ref_utts} \
        --fusion_type ${fusion_type} \
        --xattn_heads ${xattn_heads} \
        --target_dur ${target_dur}
        #--reverb_data data/rirs/lmdb \
        #--noise_data data/musan/lmdb \
	#TODO, currently also moved from local/extract_emb.sh, flexible to control musan/rirs.
fi

avg_model=$exp_dir/models/avg_model.pt
model_path=$avg_model
#######################################################################################
# Stage 5. Averaging the model, and extract embeddings
#######################################################################################
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  if [[ "$(basename $avg_model)" != "best_model.pt" ]]; then
    echo "Do model average ..."
    python wedefense/bin/average_model.py \
      --dst_model $avg_model \
      --src_path $exp_dir/models \
      --num ${num_avg}
  else
    echo "avg_model is best_model.pt, skip model averaging."
  fi


  echo "Extract embeddings ..."
  num_gpus=1
  if [[ $(hostname -f) == *fit.vutbr.cz   ]]; then
     gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
  fi
  # 这里要改的话要进extract_emb.sh改！！！
  local/extract_emb.sh \
     --exp_dir $exp_dir --model_path $model_path \
     --nj $num_gpus --gpus $gpus --data_type $data_type --data ${data} --num_ref_utts ${num_ref_utts} --fusion_type ${fusion_type} --xattn_heads ${xattn_heads}
fi

#######################################################################################
# Stage 6. Extract logits and posterior
#######################################################################################
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Extract logits and posteriors ..."
  for dset in dev;do
  # for dset in train; do
      mkdir -p ${exp_dir}/posteriors/$dset
      echo $dset
      python wedefense/bin/infer.py --model_path $model_path \
	  --config ${exp_dir}/config.yaml \
	  --num_classes 3 \
	  --embedding_scp_path ${exp_dir}/embeddings/$dset/embedding.scp \
	  --out_path ${exp_dir}/posteriors/$dset \
    --data_type $data_type \
    --num_ref_utts ${num_ref_utts} \
    --fusion_type ${fusion_type} \
    --xattn_heads ${xattn_heads}
  done
fi


#######################################################################################
# Stage 7. Convert logits to llr
#######################################################################################
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Convert logits to llr ..."
  cut -f2 -d" " ${data}/train/utt2lab | sort | uniq -c | awk '{print $2 " " $1}' > ${data}/train/lab2num_utts
  for dset in dev; do
  # for dset in train; do
      echo $dset
      python wedefense/bin/logits_to_llr_new.py \
	  --logits_scp_path ${exp_dir}/posteriors/$dset/logits.scp \
	  --train_label ${data}/train/utt2lab 

  done
fi

#######################################################################################
# Stage 8. Measuring performance
#######################################################################################
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Measuring Performance ..."
  for dset in dev; do
  # for dset in train; do
    keyfile="${data}/${dset}/sasv_key_file.txt"

    # 判断文件是否存在
    if [ ! -f "$keyfile" ]; then
      echo "Generating sasv_key_file.txt for ${dset} ..."
      echo -e "spk\tfilename\tcm-label\tasv-label" > "$keyfile"

      while read -r line; do
        utt=$(echo "$line" | awk '{print $1}')
        lab=$(echo "$line" | awk '{print $2}')
        spk=${utt%%#*}
        fname=${utt#*#}

        if [ "$lab" = "spoof" ]; then
            cm="spoof"
            asv="spoof"
        elif [ "$lab" = "target" ]; then
            cm="bonafide"
            asv="target"
        elif [ "$lab" = "nontarget" ]; then
            cm="bonafide"
            asv="nontarget"
        else
            cm="unk"
            asv="unk"
        fi

        echo -e "${spk}\t${fname}\t${cm}\t${asv}"
      done < "${data}/${dset}/utt2lab" >> "$keyfile"
    else
      echo "sasv_key_file.txt already exists for ${dset}, skipping..."
    fi
    
    echo "Measuring " $dset
    python wedefense/metrics/detection/evaluation.py  \
    --m t2_single \
    --sasv ${exp_dir}/posteriors/${dset}/llr.txt \
    --sasv_keys ${data}/${dset}/sasv_key_file.txt
  done
fi

#######################################################################################
# Stage 9. Analyses
#######################################################################################
# TODO
# 1. significant test
# 2. boostrap testing
# 3. embedding visulization
exit 0
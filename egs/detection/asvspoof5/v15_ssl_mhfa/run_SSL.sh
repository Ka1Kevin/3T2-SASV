#!/bin/bash
# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2025 Johan Rohdin, Lin Zhang (rohdin@fit.vut.cz, partialspoof@gmail.com)
#           2025 Junyi Peng (pengjy@fit.vut.cz)

#SBATCH --job-name=spoofceleb #job name
# #SBATCH --job-name=spoofceleb #job name
#SBATCH --gpus=1
#SBATCH --exclude=c14
#SBATCH --partition=gpu
# #SBATCH --partition=gpu-a100   #queue
# #SBATCH --account=a100acct
#SBATCH --mail-user="ktan17@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --error=/export/fs05/ktan17/ssl/new_file/Spoofceleb.dev.%j.err
#SBATCH --output=/export/fs05/ktan17/ssl/new_file/Spoofceleb.dev.%j.out
#SBATCH --array=0-3

#unset PYTHONPATH
#unset PYTHONHOME

echo "[$(date)] Starting Job ID: $SLURM_JOB_ID" 

source ~/.bashrc
echo `date`
echo $SSL_CERT_FILE
ls -l /etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export TMPFIR=/export/fs05/ktan17/tmp

conda activate /home/ktan17/miniconda3/envs/wedefense

#-------------------------------------------------------------------------
#LR=(1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)
#LR=(5e-3 1e-3 5e-4 1e-4)

LR=(0.1 0.01 0.001 0.0001)
initial_lr=${LR[$SLURM_ARRAY_TASK_ID]}
echo "initial_lr: $initial_lr"
# final_lr=${LR[$SLURM_ARRAY_TASK_ID]}
# echo "final_lr:$final_lr"
#-------------------------------------------------------------------------

# set -x
. ./path.sh || exit 1

stage=3
stop_stage=3
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print('GPU count:', torch.cuda.device_count())"

# ASVspoof5_dir=/export/fs05/arts/dataset/ASVspoof5
spoofceleb_dir=/export/fs05/arts/dataset/SpoofCeleb/spoofceleb/
# data=data/ASVspoof5_18w # data folder
# data=data/hard_pair/ASVspoof5/170554/
data=data/spoofceleb
trial=/export/fs05/ktan17/dataset/spoofceleb/train/3w6_pairs.tsv
data_type="raw"  # shard/raw
num_ref_utts=1
config=conf/MHFA_wav2vec2.yaml #wespeaker version
# config=conf/MHFA_wav2vec2.yaml
# exp_dir=exp/softmax/stable_lr/lr_${initial_lr}
exp_dir=exp/SSL_Spoofceleb_50e_init${initial_lr} #1372053
# exp_dir=exp/CE_MFA_Conformer_HP_3w6_50e_init${initial_lr}_final0.00005_warmup10_from0 # 1372054
# exp_dir=exp/MFA_Conformer_HP_3w6_100e_init${initial_lr}_final0.00005_warmup10_from0 # 1372055
# exp_dir=exp/MFA_Conformer_HP_3w6_200e_init${initial_lr}_final0.00005_warmup10_from0 # 1372056
gpus="[0]"
num_avg=2 # how many models you want to average
checkpoint=
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/campplus_lm.yaml

. tools/parse_options.sh || exit 1

#######################################################################################
# Stage 1. Preparing data folder for partialspoof: wav.scp, utt2lab, lab2utt, reco2dur
#######################################################################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  # ./local/prepare_data.sh ${ASVspoof5_dir} ${data}
  ./local/prepare_spoof_celeb.sh ${spoofceleb_dir} ${data} ${trial}
fi

#######################################################################################
# Stage 2. Preapring shard data for partialspoof and musan/rirs
#######################################################################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."

  cd ${data}
  ln -s flac_train_all train
  ln -s flac_development_all dev
  ln -s flac_evaluation_all eval
  cd -
  # We don't use VAD here

  # for dset in train;do
  for dset in train dev eval;do
      if [ $data_type == "shard" ]; then
          python tools/make_shard_list.py --num_utts_per_shard 1000 \
              --num_threads 8 \
              --prefix shards \
              --shuffle \
              ${data}/$dset/wav.scp ${data}/$dset/utt2lab \
              ${data}/$dset/shards ${data}/$dset/shard.list
      else
          python tools/make_raw_list_spoofceleb.py \
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
# Stage 3. Training
#######################################################################################
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
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
      wedefense/bin/new_train.py --config $config \
        --exp_dir ${exp_dir} \
        --gpus $gpus \
        --num_avg ${num_avg} \
        --data_type "${data_type}" \
        --train_data ${data}/train/${data_type}.list \
        --train_label ${data}/train/utt2lab \
        ${checkpoint:+--checkpoint $checkpoint} \
        --val_data ${data}/dev/${data_type}.list \
        --val_label ${data}/dev/utt2lab \
        #--reverb_data data/rirs/lmdb \
        #--noise_data data/musan/lmdb \
	#TODO, currently also moved from local/extract_emb.sh, flexible to control musan/rirs.
fi

avg_model=$exp_dir/models/avg_model.pt
model_path=$avg_model
#######################################################################################
# Stage 4. Averaging the model, and extract embeddings
#######################################################################################
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then

  echo "Do model average ..."
  python wedefense/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}


  echo "Extract embeddings ..."
  num_gpus=1
  if [[ $(hostname -f) == *fit.vutbr.cz   ]]; then
     gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken )" $num_gpus | sed "s: ::g")
  fi

  local/extract_emb.sh \
     --exp_dir $exp_dir --model_path $model_path \
     --nj $num_gpus --gpus $gpus --data_type $data_type --data ${data}
fi

#######################################################################################
# Stage 5. Extract logits and posterior
#######################################################################################
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Extract logits and posteriors ..."
  # for dset in dev eval;do
  for dset in train; do
      mkdir -p ${exp_dir}/posteriors/$dset
      echo $dset
      python wedefense/bin/infer.py --model_path $model_path \
	  --config ${exp_dir}/config.yaml \
	  --num_classes 3 \
	  --embedding_scp_path ${exp_dir}/embeddings/$dset/embedding.scp \
	  --out_path ${exp_dir}/posteriors/$dset
  done
fi


#######################################################################################
# Stage 6. Convert logits to llr
#######################################################################################
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Convert logits to llr ..."
  cut -f2 -d" " ${data}/train/utt2lab | sort | uniq -c | awk '{print $2 " " $1}' > ${data}/train/lab2num_utts
  # for dset in dev eval; do
  for dset in train; do
      echo $dset
      python wedefense/bin/logits_to_llr_new.py \
	  --logits_scp_path ${exp_dir}/posteriors/$dset/logits.scp \
	  --train_label ${data}/train/utt2lab \

  done
fi

#######################################################################################
# Stage 7. Measuring performance
#######################################################################################
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Measuring Performance ..."
  # for dset in dev eval; do
  for dset in train; do
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
# Stage 8. Analyses
#######################################################################################
# TODO
# 1. significant test
# 2. boostrap testing
# 3. embedding visulization
exit 0
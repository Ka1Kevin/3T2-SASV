#!/bin/bash
# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2025 Johan Rohdin, Lin Zhang (rohdin@fit.vut.cz, partialspoof@gmail.com)
#           2025 Junyi Peng (pengjy@fit.vut.cz)

#SBATCH --job-name=wildspoof #job name
# #SBATCH --job-name=spoofceleb #job name
#SBATCH --gpus=1
#SBATCH --exclude=c14,c05
#SBATCH --partition=gpu
# #SBATCH --partition=gpu-a100   #queue
# #SBATCH --account=a100acct
#SBATCH --mail-user="ktan17@jh.edu"  #email for reporting
#SBATCH --mail-type=END,FAIL  #report types
#SBATCH --error=/export/fs05/ktan17/ssl/new_new_file/wildspoof.dev.%j.err
#SBATCH --output=/export/fs05/ktan17/ssl/new_new_file/wildspoof.dev.%j.out
# #SBATCH --array=0-3

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
set -euo pipefail

proto=/export/fs06/arts/dataset/wildspoof_test/protocol_test_label.tsv
llr=/export/fs05/ktan17/ssl/wedefense-main/egs/detection/asvspoof5/v15_ssl_mhfa/primary_full.txt
key=/export/fs05/ktan17/ssl/wedefense-main/egs/detection/asvspoof5/v15_ssl_mhfa/data/wildspoof_test/eval/sasv_key_file.txt

outdir=/export/fs05/ktan17/ssl/wedefense-main/egs/detection/asvspoof5/v15_ssl_mhfa/exp/ska_tdnn_wild_spoof/posteriors/eval/by_source
mkdir -p "$outdir"

# 1) 把 proto 统一成 tab 分隔：spk \t utt \t source
proto_map=$outdir/proto_map.tsv
awk '{print $1"\t"$2"\t"$NF}' "$proto" > "$proto_map"

# 2) sources 列表（第3列）
sources=$(awk -F'\t' '{print $3}' "$proto_map" | sort -u)

for s in $sources; do
  echo
  echo "=============================="
  echo "Evaluating source: $s"
  echo "=============================="

  llr_sub=$outdir/llr_${s}.txt
  key_sub=$outdir/key_${s}.txt

  # 过滤 llr：保留 header + 属于该 source 的 trial
  awk -F'\t' -v src="$s" '
    NR==FNR { m[$1 FS $2]=$3; next }   # proto_map: (spk,utt)->source
    FNR==1 { print; next }            # llr header
    ( ($1 FS $2) in m ) && m[$1 FS $2]==src { print }
  ' "$proto_map" "$llr" > "$llr_sub"

  # 过滤 key：保留 header + 属于该 source 的 trial
  awk -F'\t' -v src="$s" '
    NR==FNR { m[$1 FS $2]=$3; next }   # proto_map: (spk,utt)->source
    FNR==1 { print; next }            # key header
    ( ($1 FS $2) in m ) && m[$1 FS $2]==src { print }
  ' "$proto_map" "$key" > "$key_sub"

  # trial 数
  n_llr=$(awk -F'\t' 'NR>1{c++} END{print c+0}' "$llr_sub")
  n_key=$(awk -F'\t' 'NR>1{c++} END{print c+0}' "$key_sub")
  echo "Trials kept: llr=$n_llr, key=$n_key"

  # 三类计数（a-DCF 需要 tar/non/spoof 都非空）
  n_tar=$(awk -F'\t' 'NR>1 && $3=="bonafide" && $4=="target"{c++} END{print c+0}' "$key_sub")
  n_non=$(awk -F'\t' 'NR>1 && $3=="bonafide" && $4=="nontarget"{c++} END{print c+0}' "$key_sub")
  n_spf=$(awk -F'\t' 'NR>1 && $3=="spoof"{c++} END{print c+0}' "$key_sub")
  echo "Class counts: tar=$n_tar non=$n_non spoof=$n_spf"

  python wedefense/metrics/detection/evaluation.py \
    --m t2_single \
    --sasv "$llr_sub" \
    --sasv_keys "$key_sub"
done
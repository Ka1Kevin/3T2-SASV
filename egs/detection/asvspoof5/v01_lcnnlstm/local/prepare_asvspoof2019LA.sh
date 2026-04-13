#!/bin/bash
set -xe

ASVspoof2019LA_dir=$1     # /export/fs05/arts/dataset/ASVspoof2019/LA
data_dir=$2               # data/ASVspoof2019LA
trial_train=$3            # /export/fs05/ktan17/dataset/ASVSpoof2019LA/train_trial.txt

# 固定（你说的固定路径）
trial_dev=${ASVspoof2019LA_dir}/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt
trial_eval=${ASVspoof2019LA_dir}/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt

# 保持你原先的缩写命名
DSETs=(T D E)
DSETs_full=(train dev eval)

for i in "${!DSETs[@]}"; do
  dset=${DSETs[$i]}
  dset_full=${DSETs_full[$i]}
  out_dir=${data_dir}/flac_${dset}_all
  mkdir -p ${out_dir}

  # wav.scp: utt_id abs_path
  find ${ASVspoof2019LA_dir}/ASVspoof2019_LA_${dset_full}/flac -name "*.flac" \
    | awk -F'/' '{fn=$NF; sub(/\.flac$/,"",fn); print fn, $0}' \
    | sort -k1,1 > ${out_dir}/wav.scp

  # 选择 trial 文件（train 用你生成的；dev/eval 用固定路径）
  if [ "$dset_full" = "train" ]; then
    trial_file=${trial_train}
  elif [ "$dset_full" = "dev" ]; then
    trial_file=${trial_dev}
  else
    trial_file=${trial_eval}
  fi

  # 你的 trial 格式（四列）：
  # spk utt bonafide/spoof target|nontarget|spoof
  # 生成 utt2lab: key=spk#utt, value=第4列(label)
  awk '{print $1"#"$2, $4}' ${trial_file} > ${out_dir}/utt2lab

  # spk2utt：建议对每个 split 用“该 split 里 bonafide 的 utt”聚合
  # 避免把 nontarget 的 utt（来自别的 spk）混进 enrollment
  awk '$3=="bonafide"{print $1,$2}' ${trial_file} \
    | sort -k1,1 \
    | awk '{
        spk=$1; utt=$2;
        if (spk in arr) arr[spk]=arr[spk]","utt;
        else arr[spk]=utt;
      }
      END { for (spk in arr) print spk, arr[spk]; }' \
    > ${out_dir}/spk2utt

  ./tools/utt2lab_to_lab2utt.pl ${out_dir}/spk2utt > ${out_dir}/lab2spk
  python tools/wav2dur.py ${out_dir}/wav.scp ${out_dir}/utt2dur
done

echo "Prepared ASVspoof2019LA data: wav.scp, utt2lab, spk2utt, lab2spk, utt2dur"
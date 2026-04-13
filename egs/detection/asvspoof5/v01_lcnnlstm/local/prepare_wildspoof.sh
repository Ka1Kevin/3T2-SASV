#!/bin/bash
#
# Prepare wildspoof_test data dir:
#   wav.scp, utt2lab, lab2utt, utt2dur
#
# Usage:
#   local/prepare_data_wildspoof.sh /export/fs06/arts/dataset/wildspoof_test  <out_data_dir>  /export/fs06/arts/dataset/wildspoof_test/protocol_test_label.tsv

set -xe

WILDSPOOF_DIR=$1   # e.g., /export/fs06/arts/dataset/wildspoof_test
data_dir=$2        # e.g., data/wildspoof_test
trial_file=$3      # e.g., .../protocol_test_label.tsv

DSETs=(eval)

for dset in "${DSETs[@]}"; do

  out_dir=${data_dir}/flac_${dset}_all
  mkdir -p "${out_dir}"

  flac_root="${WILDSPOOF_DIR}/data_v2.0"

  # -------------------------
  # 1) wav.scp: utt_id path
  # -------------------------
  find "${flac_root}" -maxdepth 1 -type f -name "UTT_*.flac" | sort | \
    awk -F'/' 'BEGIN{OFS=" "}{
      fname=$NF;
      utt=fname;
      sub(/\.flac$/,"",utt);
      print utt, $0
    }' > "${out_dir}/wav.scp"

  # sanity check: wav.scp count == flac count
  num_flac=$(find "${flac_root}" -maxdepth 1 -type f -name "UTT_*.flac" | wc -l)
  num_wavscp=$(wc -l < "${out_dir}/wav.scp")
  echo "[CHECK] ${dset}: flac files = ${num_flac}, wav.scp entries = ${num_wavscp}"
  if [ "${num_flac}" -ne "${num_wavscp}" ]; then
      echo "[ERROR] ${dset}: wav.scp entry number mismatch!"
      exit 1
  fi

  # -------------------------
  # 2) utt2lab: spk#utt label
  # protocol_test_label.tsv format:
  #   SPK_00121556_000  UTT_00114655  target  spoofceleb
  # label choices:
  #   - use $3 only: target/non/spoof
  #   - or use $3"@"$4 to keep source too
  # -------------------------
  awk '
    BEGIN{OFS=" "}
    {
      spk=$1; utt=$2; lab=$3; src=$4;
      # Option A (recommended): 3-class label only
      print spk "#" utt, lab;

      # Option B: keep source domain too (uncomment if needed)
      # print spk "#" utt, lab "@" src;
    }
  ' "${trial_file}" > "${out_dir}/utt2lab"

  # check duplicate trial keys
  num_pairs=$(wc -l < "${out_dir}/utt2lab")
  num_uniq_pairs=$(awk '{print $1}' "${out_dir}/utt2lab" | sort | uniq | wc -l)
  if [ "${num_pairs}" -ne "${num_uniq_pairs}" ]; then
    echo "[ERROR] ${dset}: duplicate pair IDs in utt2lab!"
    exit 1
  fi

  ./tools/utt2lab_to_lab2utt.pl "${out_dir}/utt2lab" > "${out_dir}/lab2utt"

  # -------------------------
  # 3) utt2dur
  # -------------------------
  python tools/wav2dur.py "${out_dir}/wav.scp" "${out_dir}/utt2dur"

done

echo "Prepared data folder including wav.scp, utt2lab, lab2utt, utt2dur"
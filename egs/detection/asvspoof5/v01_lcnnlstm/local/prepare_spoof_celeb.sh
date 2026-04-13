#!/bin/bash
#
# Copyright 2025 Lin Zhang (partialspoof@gmail.com)
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

#local/prepare_data.sh [ASVspoof5_dir] [data_dir]
#
#Download ASVspoof5 database,
#and prepare data dir for partial spoof: wav.scp, utt2lab, lab2utt, utt2dur

set -xe

ASVspoof5_dir=$1
data_dir=$2
trial_file=$3

DSETs=(T D E_eval)
DSETs_full=(train dev eval)


if [ ! -d ${ASVspoof5_dir} ]; then
    mkdir -p ${ASVspoof5_dir}
    bash ./01_download_database.sh ${ASVspoof5_dir}
fi

for i in "${!DSETs[@]}"; do
  dset=${DSETs[$i]}
  dset_full=${DSETs_full[$i]}

  if [ ! -d ${data_dir}/flac_${dset}_all ]; then
     mkdir -p ${data_dir}/flac_${dset}_all
  fi

  find ${ASVspoof5_dir}/flac/${dset}/ -name "*.flac" | sort | \
  awk -F'/' '
    BEGIN{OFS=" "}
    {
      path=$0
      fname=$NF; gsub(/\.flac$/,"",fname)
      spkid=$(NF-1)
      attack=$(NF-2)
      print attack "@" spkid "@" fname, path
    }
  ' > ${data_dir}/flac_${dset}_all/wav.scp
  
  # check row number.
  num_flac=$(find ${ASVspoof5_dir}/flac/${dset}/ -name "*.flac" | wc -l)
  num_wavscp=$(wc -l < ${data_dir}/flac_${dset}_all/wav.scp)
  echo "[CHECK] ${dset}: flac files = ${num_flac}, wav.scp entries = ${num_wavscp}"
  if [ "${num_flac}" -ne "${num_wavscp}" ]; then
      echo "[ERROR] ${dset}: wav.scp entry number mismatch!"
      exit 1
  fi

  # produce utt2lab from protocols
  if [ "$dset" = "train"  ]; then
    awk '
      BEGIN{OFS=" "}
      {
        ref=$2; test=$4; lab=$5;

        # --- ref: attack@spk@fname ---
        n1=split(ref, arr1, "/");
        ref_fname=arr1[n1]; gsub(/\.flac$/,"",ref_fname);
        ref_spk=arr1[n1-1];
        ref_atk=arr1[n1-2];
        ref_id=ref_atk "@" ref_spk "@" ref_fname;

        # --- test: attack@spk@fname ---
        n2=split(test, arr2, "/");
        test_fname=arr2[n2]; gsub(/\.flac$/,"",test_fname);
        test_spk=arr2[n2-1];
        test_atk=arr2[n2-2];
        test_id=test_atk "@" test_spk "@" test_fname;

        print ref_id "#" test_id, lab;
      }
    ' $trial_file > ${data_dir}/flac_${dset}_all/utt2lab
  else
    proto="${ASVspoof5_dir}/protocol/sasv_${dset}_evaluation_protocol.csv"

    awk -F',' '
      BEGIN{OFS=" "}
      {
        ref=$1; test=$2; lab=$3;

        n1=split(ref, arr1, "/");
        ref_fname=arr1[n1]; gsub(/\.flac$/,"",ref_fname);
        ref_spk=arr1[n1-1];
        ref_atk=arr1[n1-2];
        ref_id=ref_atk "@" ref_spk "@" ref_fname;

        n2=split(test, arr2, "/");
        test_fname=arr2[n2]; gsub(/\.flac$/,"",test_fname);
        test_spk=arr2[n2-1];
        test_atk=arr2[n2-2];
        test_id=test_atk "@" test_spk "@" test_fname;

        print ref_id "#" test_id, lab;
      }
    ' $proto > ${data_dir}/flac_${dset}_all/utt2lab
  fi
  num_pairs=$(wc -l < ${data_dir}/flac_${dset}_all/utt2lab)
  num_uniq_pairs=$(awk '{print $1}' ${data_dir}/flac_${dset}_all/utt2lab | sort | uniq | wc -l)
  if [ "${num_pairs}" -ne "${num_uniq_pairs}" ]; then
    echo "[ERROR] ${dset}: duplicate pair IDs in utt2lab!"
    exit 1
  fi

  ./tools/utt2lab_to_lab2utt.pl ${data_dir}/flac_${dset}_all/utt2lab \
	  >${data_dir}/flac_${dset}_all/lab2utt

  #we are using wav2dur.py, but quite slow.
  python tools/wav2dur.py ${data_dir}/flac_${dset}_all/wav.scp ${data_dir}/flac_${dset}_all/utt2dur
done

echo "Prepared data folder including wav.scp, utt2lab, lab2utt"
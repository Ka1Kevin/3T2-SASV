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
trial=$3

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
  
  find ${ASVspoof5_dir}/flac_${dset}/ -name "*.flac" | awk -F"/" '{print $NF,$0}' |\
         sort > ${data_dir}/flac_${dset}_all/wav.scp
  sed -i 's/\.flac / /g' ${data_dir}/flac_${dset}_all/wav.scp


  # produce utt2lab from protocols
  if [ "$dset" = "T"  ]; then
    awk 'NR>1 {print $1"#"$4, $5}' ${trial} \
      > ${data_dir}/flac_${dset}_all/utt2lab

    awk -F ' +' '$9=="bonafide"{print $1, $2}' ${ASVspoof5_dir}/ASVspoof5.${dset_full}.tsv \
      | sort -k1,1 \
      | awk '{
          spk=$1; utt=$2;
          if (spk in arr) {
              arr[spk]=arr[spk]","utt;
          } else {
              arr[spk]=utt;
          }
      }
      END {
          for (spk in arr) {
              print spk, arr[spk];
          }
      }' > ${data_dir}/flac_${dset}_all/spk2utt
  else
    cp ${ASVspoof5_dir}/ASVspoof5.${dset_full}.track_2.enroll.tsv ${data_dir}/flac_${dset}_all/spk2utt
    awk -F ' +' '{ 
        key = $1"#"$2; 
        if ($4 == "bonafide") { 
            print key, $5 
        } else { 
            print key, "spoof" 
        } 
    }' ${ASVspoof5_dir}/ASVspoof5.${dset_full}.track_2.trial.tsv \
    > ${data_dir}/flac_${dset}_all/spk2utt
  fi

  ./tools/utt2lab_to_lab2utt.pl ${data_dir}/flac_${dset}_all/spk2utt \
	  >${data_dir}/flac_${dset}_all/lab2spk

  #we are using wav2dur.py, but quite slow.
  python tools/wav2dur.py ${data_dir}/flac_${dset}_all/wav.scp ${data_dir}/flac_${dset}_all/utt2dur
done

echo "Prepared data folder for partialspoof, including wav.scp, utt2lab, lab2utt"

# all是为track2准备的
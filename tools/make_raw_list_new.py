# Copyright (c) 2022 Binbin Zhang(binbzha@qq.com)
#               2023 Zhengyang Chen(chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import json
import os
import random

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_ref_utts', type=int, default=0,
                        help='number of reference utterances to sample (0 = only use test wav)')
    parser.add_argument('wav_file', help='wav file',)
    parser.add_argument('utt2lab_file', help='utt2lab file')
    parser.add_argument('spk2utt_file', help='spk2utt file')
    parser.add_argument('raw_list', help='output raw list file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            key = arr[0]  # os.path.splitext(arr[0])[0]
            wav_table[key] = ' '.join(arr[1:])

    spk2utt = {}
    with open(args.spk2utt_file, 'r', encoding='utf8') as fin:
        for line in fin:
            spk, utts = line.strip().split(maxsplit=1)
            spk2utt[spk] = utts.split(',')

    data = []
    with open(args.utt2lab_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]               # eg. T_0012#T_0000129935
            lab = arr[1]

            ref_spk, test_utt = key.split('#')
            assert test_utt in wav_table
            test_wav = wav_table[test_utt]

            # sample reference utterances
            ref_wavs = []
            if args.num_ref_utts > 0:
                if ref_spk in spk2utt:
                    utt_candidates = spk2utt[ref_spk]
                    # random sample
                    sampled_utts = random.sample(
                        utt_candidates,
                        min(args.num_ref_utts, len(utt_candidates))
                    )
                    ref_wavs = [wav_table[u] for u in sampled_utts if u in wav_table]

            line_dict = dict(key=key, lab=lab)
            if args.num_ref_utts > 0:
                line_dict["wav"] =  ref_wavs + [test_wav]
            else:
                line_dict["wav"] = test_wav

            data.append(line_dict)

    with open(args.raw_list, 'w', encoding='utf8') as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()

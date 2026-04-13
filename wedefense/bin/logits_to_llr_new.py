#!/usr/bin/env python
# Copyright (c) 2025 Lin Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fire
import kaldiio
import numpy as np
import os
from wedefense.utils.file_utils import read_table
from wedefense.utils.utils import lab2id
from scipy.special import logsumexp


def compute_llr_sasv(logits, pi_non=0.0095, pi_spf=0.05,
                     train_label_ratio=[0.33, 0.33, 0.33]):
    """
    Compute LLR for SASV task.

    Label mapping in logits:
        0 -> nontarget
        1 -> spoof
        2 -> target

    Args:
        logits: ndarray [N, 3] with columns [s_non, s_spf, s_tar]
        pi_non: prior prob. of nontarget (test-time)
        pi_spf: prior prob. of spoof (test-time)
        train_label_ratio: priors in training set [non, spf, tar]

    Returns:
        llr: ndarray [N]
    """
    logits = np.array(logits)
    s_non, s_spf, s_tar = logits[:, 0], logits[:, 1], logits[:, 2]

    # prior adjustment only for training set
    pi_train_non, pi_train_spf, pi_train_tar = train_label_ratio
    s_adj_tar = s_tar - np.log(pi_train_tar)

    # normalized weighted sum of non-target and spoof scores as denominator
    pi_total_rej = pi_non + pi_spf
    log_pi_non = np.log(pi_non / pi_total_rej)
    log_pi_spf = np.log(pi_spf / pi_total_rej)

    # log-sum-exp as denominator to avoid underflow
    den = logsumexp(
        np.stack([log_pi_non + s_non,
                  log_pi_spf + s_spf], axis=0),
        axis=0
    )

    # LLR calculation
    llr = s_adj_tar - den
    return llr


def main(logits_scp_path, train_label, pi_non=0.0095, pi_spf=0.05):
    """
    Args:
        logits_scp_path: path to logits.scp
        train_label: training label file
        pi_non: test-time prior of nontarget
        pi_spf: test-time prior of spoof
    """
    print(f"Loading logits from {logits_scp_path}")

    # read logits
    utt, logits = [], []
    for k, v in kaldiio.load_scp_sequential(logits_scp_path):
        utt.append(k)
        logits.append(v)
    logits = np.vstack(logits)
    print(f"logits shape: {logits.shape}")

    # read training label to get priors
    train_utt_lab_list = read_table(train_label)
    lab2id_dict = lab2id(train_utt_lab_list)

    counts = {k: 0 for k in lab2id_dict.keys()}
    for _, lab in train_utt_lab_list:
        counts[lab] += 1

    total = sum(counts.values())
    train_label_ratio = [
        counts["nontarget"] / total,  # index 0
        counts["spoof"] / total,      # index 1
        counts["target"] / total,     # index 2
    ]
    print(f"Training priors (non, spf, tar): {train_label_ratio}")

    # LLR calculation
    llr = compute_llr_sasv(
        logits,
        pi_non=pi_non,
        pi_spf=pi_spf,
        train_label_ratio=train_label_ratio
    )

    # Nan check
    nan_idx = np.where(np.isnan(llr))[0]
    if len(nan_idx):
        print(f"Warning: {len(nan_idx)} NaN LLRs → set to 0")
        llr[nan_idx] = 0

    # output LLR scores
    out_path = os.path.dirname(logits_scp_path)
    out_file = os.path.join(out_path, "llr.txt")
    with open(out_file, "w") as f:
        f.write("spk\tfilename\tcm-score\tasv-score\tsasv-score\n")
        for i, u in enumerate(utt):
            spk, fname = u.split("#", 1)
            f.write(f"{spk}\t{fname}\t-\t-\t{llr[i]}\n")

    print(f"LLR scores saved to {out_file}")



if __name__ == "__main__":
    fire.Fire(main)

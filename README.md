# WeDefense Inference Pipeline for SASV

This repository provides inference pipelines for spoofing-robust automatic speaker verification (SASV) using pretrained base checkpoints on:

- ASVspoof5  
- SpoofCeleb  
- WildSpoof  

Given a new trial file, the pipeline:

1. Prepares dataset metadata  
2. Extracts embeddings  
3. Runs posterior/logit inference  
4. Converts logits into log-likelihood ratio (LLR) scores  
5. Computes final SASV evaluation metrics  

This repository supports cross-dataset inference and evaluation under a unified three-class SASV formulation (**target / nontarget / spoof**).

---

# Repository Structure

```bash
.
├── egs/detection/asvspoof5/v15_ssl_mhfa/
│   ├── run_asvspoof5.sh
│   ├── run_spoofceleb.sh
│   └── run_wildspoof.sh
│
├── wedefense/bin/
│   ├── infer.py
│   ├── logits_to_llr_new.py
│   └── average_model.py
│
├── egs/detection/asvspoof5/v15_ssl_mhfa/local/
│   ├── prepare_data_track2.sh
│   ├── prepare_spoof_celeb.sh
│   └── prepare_wildspoof.sh
│
├── egs/detection/asvspoof5/v15_ssl_mhfa/conf/
│   ├── MFA_Conformer.yaml
│   └── SKA_TDNN.yaml
│
└── exp/
    └── checkpoints / outputs
```

---

# Requirements

## Environment

Create and activate a Python environment:

```bash
conda create -n wedefense python=3.10
conda activate wedefense
```

Install dependencies from the provided requirement file:

```bash
pip install -r requirements.txt
```

---

# Supported Datasets

This framework supports inference and evaluation on:

- ASVspoof5  
- SpoofCeleb  
- WildSpoof  

## Supported Models

Multiple backbones are supported, including:

- MFA-Conformer  
- SKA-TDNN  

---

# Input Format

## Trial File

Inference expects a tab-separated trial file with the format:

```text
ref_speaker   ref_utts   test_speaker   test_utt   label   num_ref
```

Columns:

| Column       | Description                       |
| ------------ | --------------------------------- |
| ref_speaker  | Enrollment speaker ID             |
| ref_utts     | Reference utterance(s) or path(s) |
| test_speaker | Test speaker ID                   |
| test_utt     | Test utterance or path            |
| label        | target / nontarget / spoof        |
| num_ref      | Number of reference utterances    |

Labels:

- `target` : same-speaker bonafide trial  
- `nontarget` : different-speaker bonafide trial  
- `spoof` : spoofed trial  

This repository uses a unified three-class SASV trial format.

---

## Example: ASVspoof5

ASVspoof5 trials use utterance IDs:

```text
T_3734	T_0000000011	T_3734	T_0000152869	target	1
```

Interpretation:

- reference speaker: `T_3734`
- reference utterance: `T_0000000011`
- test speaker: `T_3734`
- test utterance: `T_0000152869`
- label: target
- number of reference utterances: 1

---

## Example: SpoofCeleb

SpoofCeleb trials use full audio paths:

```text
id10830	/dataset/SpoofCeleb/spoofceleb/flac/train/a00/id10830/awlcO3BiN2I-00002-001.flac	id10061	/dataset/SpoofCeleb/spoofceleb/flac/train/a00/id10061/ut8dzAUnDp4-00004-003.flac	nontarget	1
```

Interpretation:

- reference speaker: `id10830`
- reference utterance: full enrollment waveform path
- test speaker: `id10061`
- test utterance: full test waveform path
- label: nontarget
- number of reference utterances: 1

Note:

- ASVspoof5 uses utterance IDs  
- SpoofCeleb uses absolute waveform paths  

Both formats are supported.

---

# Quick Start

All inference scripts are located in:

```bash
egs/detection/asvspoof5/v15_ssl_mhfa/
```

From the repository root:

## ASVspoof5

```bash
cd egs/detection/asvspoof5/v15_ssl_mhfa
bash run_asvspoof5.sh
```

## SpoofCeleb

```bash
cd egs/detection/asvspoof5/v15_ssl_mhfa
bash run_spoofceleb.sh
```

## WildSpoof

```bash
cd egs/detection/asvspoof5/v15_ssl_mhfa
bash run_wildspoof.sh
```

---

# Inference Pipeline

## Step 1: Extract embeddings

```bash
local/extract_emb.sh \
  --exp_dir $exp_dir \
  --model_path $model_path \
  --data $data
```

Output:

```text
exp/.../embeddings/<split>/embedding.scp
```

---

## Step 2: Run inference

Compute posterior logits:

```bash
python wedefense/bin/infer.py \
  --model_path ${model_path} \
  --config ${exp_dir}/config.yaml \
  --num_classes 3 \
  --embedding_scp_path \
      ${exp_dir}/embeddings/${dset}/embedding.scp \
  --out_path \
      ${exp_dir}/posteriors/${dset} \
  --data_type raw \
  --num_ref_utts 1 \
  --fusion_type feat_cat \
  --xattn_heads 4
```

Outputs:

Outputs:

```text
logits.scp
logits.ark
posteriors.scp
posteriors.ark
```

---

## Step 3: Convert logits to LLR scores

```bash
python wedefense/bin/logits_to_llr_new.py \
  --logits_scp_path \
      ${exp_dir}/posteriors/${dset}/logits.scp \
  --train_label \
      ${data}/train/utt2lab
```

Output:

```text
llr.txt
```

This is the final SASV score file.

---

## Step 4: Evaluate

```bash
python wedefense/metrics/detection/evaluation.py \
  --m t2_single \
  --sasv ${exp_dir}/posteriors/${dset}/llr.txt \
  --sasv_keys ${data}/${dset}/sasv_key_file.txt
```

Metrics may include:

- SASV-EER  
- minDCF  
- Other detection metrics

---

# Running Inference on New Data

Given a new trial file:

Replace in the corresponding shell script:

```bash
trial=/path/to/new_trials.tsv
```

Set checkpoint:

```bash
model_path=/path/to/checkpoint.pt
```

Then run:

```bash
cd egs/detection/asvspoof5/v15_ssl_mhfa
bash run_xxx.sh
```

Pipeline:

```text
embeddings
→ logits
→ llr scores
→ final metrics
```

---

# Example Outputs

After inference, outputs are generated under:

```bash
exp/MFA_Conformer/posteriors/dev/
```

Example:

```text
llr.txt
logits.ark
logits.scp
posteriors.ark
posteriors.scp
```

## Output Files

### Posterior Outputs

Posterior probabilities:

```text
posteriors.scp
posteriors.ark
```

### Logit Outputs

Raw model logits before LLR conversion:

```text
logits.scp
logits.ark
```

### Final SASV Scores

Final scores after logits-to-LLR conversion:

```text
llr.txtllr.txt
```

This is the main scoring file used for evaluation.

Format:

```text
spk	filename	cm-score	asv-score	sasv-score
```

Example:

```text
D_0062	D_0000000001	-	-	-4.054442405700684
```

Fields:

- `spk` : speaker ID  
- `filename` : test utterance  
- `cm-score` : countermeasure score (if used)  
- `asv-score` : ASV score (if used)  
- `sasv-score` : final SASV score produced by this system  

The final evaluation uses the `sasv-score` column.

Higher SASV scores indicate stronger target-speaker support.

---

# Stage Control

Scripts support staged execution:

```bash
stage=
stop_stage=
```

Examples:

Run only inference + scoring:

```bash
stage=6
stop_stage=8
```

Run from embedding extraction:

```bash
stage=5
stop_stage=8
```

---

# # Citation

If you use this repository or build upon this work, please cite:

```bibtex
@misc{tan2026integratedspoofingrobustautomaticspeaker,
  title={Integrated Spoofing-Robust Automatic Speaker Verification via a Three-Class Formulation and LLR},
  author={Kai Tan and Lin Zhang and Ruiteng Zhang and Johan Rohdin and Leibny Paola García-Perera and Zexin Cai and Sanjeev Khudanpur and Matthew Wiesner and Nicholas Andrews},
  year={2026},
  eprint={2603.13780},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2603.13780}
}
```

Paper:
https://arxiv.org/abs/2603.13780

---
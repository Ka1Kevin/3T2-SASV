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


---
# Quick Start for Inference
## Stage 1: Prepare Inference Data

For inference, prepare the following three files:

```text
data/test/
├── raw.list
├── utt2lab
└── sasv_key_file.txt
```

---

### `raw.list` Format

`raw.list` should contain one JSON object per line:

```json
{"key": "<trial_key>", "lab": "<label>", "wav": ["<reference_wav_1>", "<reference_wav_2>", "<test_wav>"]}
```

Notes:

- `key` can be manually defined.
- `lab` should be one of: `target`, `nontarget`, or `spoof`.
- `wav` contains reference utterance(s) followed by the test utterance.
- The test utterance must always be the last item in `wav`.

---

### `utt2lab` Format

```text
<trial_key> <label>
```

The `<trial_key>` must match the `key` field in `raw.list`.

### `sasv_key_file` Format

```text
spk	filename	cm-label	asv-label
```

Label mapping:

| Trial Label | cm-label   | asv-label   |
| ----------- | ---------- | ----------- |
| `spoof`     | `spoof`    | `spoof`     |
| `target`    | `bonafide` | `target`    |
| `nontarget` | `bonafide` | `nontarget` |

---

### Example

`raw.list`

```json
{"key": "D_0062#D_0000000001", "lab": "spoof", "wav": ["/export/fs05/arts/dataset/ASVspoof5/flac_D/D_A0000001158.flac", "/export/fs05/arts/dataset/ASVspoof5/flac_D/D_0000000001.flac"]}
```

`utt2lab`

```text
D_0062#D_0000000001 spoof
```

In this example:

```text
/export/fs05/arts/dataset/ASVspoof5/flac_D/D_A0000001158.flac → reference utterance
/export/fs05/arts/dataset/ASVspoof5/flac_D/D_0000000001.flac  → test utterance
```

`sasv_key_file.txt`

```text
spk	filename	cm-label	asv-label
D_0062	D_0000000001	spoof	spoof
D_0062	D_0000000002	bonafide	target
D_0062	D_0000000003	bonafide	nontarget
```

## Stage 2: Extract Embeddings

After preparing `raw.list` and `utt2lab`, extract embeddings using:

```bash
local/extract_emb.sh \
  --exp_dir <experiment_output_dir> \       # directory to save embeddings and later outputs
  --model_path <checkpoint_path> \          # pretrained checkpoint used for embedding extraction
  --nj <num_jobs> \                         			# number of parallel jobs
  --gpus "<gpu_ids>" \                     			  # GPU IDs, e.g., "[0]" or "[0,1]"
  --data_type "raw" \                      				# input data format; use "raw" for raw.list
  --data <data_dir> \                       			# data directory containing raw.list and utt2lab
  --num_ref_utts <num_reference_utts> \  # number of reference utterances per trial
  --fusion_type <fusion_method> \       	  # fusion method, e.g., feat_cat or xattn_add
  --xattn_heads <num_attention_heads>   # number of cross-attention heads
```

Expected input:

```text
exp/test/raw.list
```

Expected outputs:

```text
exp/test/embeddings/
├── embedding_000.ark
├── embedding_000.scp
├── extract.result
└── embedding.scp
```
### Example

```bash
local/extract_emb.sh \
  --exp_dir exp/test \
  --model_path checkpoints/ska_tdnn_asvspoof5_best_model.pt \
  --nj 1 \
  --gpus "[0]" \
  --data_type "raw" \
  --data data/test \
  --num_ref_utts 1 \
  --fusion_type "xattn_add" \
  --xattn_heads 4
```

### Output: `embedding.scp`

After embedding extraction, the main output file is:

```text
exp/test/embeddings/embedding.scp
```

Format:

```text
<trial_key> <ark_path>:<byte_offset>
```

Example:

```text
D_0062#D_0000000001 exp/test/embeddings/embedding_000.ark:20
```

Fields:

| Field | Description |
|---|---|
| `trial_key` | The same trial key defined in `raw.list` and `utt2lab` |
| `ark_path` | Path to the binary embedding archive file |
| `byte_offset` | Offset position of the embedding entry inside the `.ark` file |

The corresponding `.ark` file stores the actual embedding vectors, while `embedding.scp` provides an index for locating each embedding.

## Stage 3: Run Inference

After embedding extraction, run `wedefense/bin/infer.py` to generate logits and posterior outputs.

### Format

```bash
mkdir -p <posterior_output_dir>

python wedefense/bin/infer.py \
  --model_path <checkpoint_path> \             				 # pretrained checkpoint used for inference
  --config <config_path> \                      					  # model config file
  --num_classes 3 \                             						# number of classes: target / nontarget / spoof
  --embedding_scp_path <embedding_scp_path> \   # embedding.scp generated in Stage 2
  --out_path <posterior_output_dir> \           			   # directory to save logits and posteriors
  --data_type "raw" \                          							 # input data type
  --num_ref_utts <num_reference_utts> \         		  # number of reference utterances per trial
  --fusion_type <fusion_method> \               			   # fusion method, e.g., feat_cat or xattn_add
  --xattn_heads <num_attention_heads>           		 # number of cross-attention heads
```

Expected input:

```text
exp/test/embeddings/embedding.scp
```

Expected outputs:

```text
exp/test/posteriors/
├── logits.ark
├── logits.scp
├── posteriors.ark
└── posteriors.scp
```

### Example

```bash
mkdir -p exp/test/posteriors

python wedefense/bin/infer.py \
  --model_path checkpoints/ska_tdnn_asvspoof5_best_model.pt \
  --config exp/test/config.yaml \
  --num_classes 3 \
  --embedding_scp_path exp/test/embeddings/embedding.scp \
  --out_path exp/test/posteriors \
  --data_type "raw" \
  --num_ref_utts 1 \
  --fusion_type "xattn_add" \
  --xattn_heads 4
```

### Output: `logits.scp`

Format:

```text
<trial_key> <ark_path>:<byte_offset>
```

Example:

```text
D_0062#D_0000000001 exp/test/posteriors/logits.ark:20
```

### Output: `posteriors.scp`

Format:

```text
<key> <ark_path>:<byte_offset>
```

Example:

```text
D exp/test/posteriors/posteriors.ark:2
```

Notes:

- `.scp` files are index files.
- `.ark` files store the actual tensor values.
- `logits.scp` indexes raw logits.
- `posteriors.scp` indexes posterior probabilities.

## Stage 4: Convert Logits to LLR Scores

After inference, convert `logits.scp` into final SASV LLR scores using `wedefense/bin/logits_to_llr_new.py`.

### Format

```bash
python wedefense/bin/logits_to_llr_new.py \
  --logits_scp_path <logits_scp_path> \   # logits.scp generated in Stage 3
  --train_label <label_file>              			# label file used to compute class priors
```

Expected input:

```text
└── exp/test/posteriors/logits.scp
└── data/test/utt2lab
```

Expected output:

```text
exp/test/posteriors/llr.txt
```

### Example

```bash
python wedefense/bin/logits_to_llr_new.py \
  --logits_scp_path exp/test/posteriors/logits.scp \
  --train_label data/test/utt2lab
```

The output file `llr.txt` has the format:

```text
spk	filename	cm-score	asv-score	sasv-score
```

Example:

```text
D_0062	D_0000000001	-	-	-4.054442405700684
```

## Stage 5: Compute a-DCF

After generating `llr.txt`, compute the final SASV evaluation metric using `evaluation.py`.

### Format

```bash
python wedefense/metrics/detection/evaluation.py \
  --m <evaluation_mode> \              # evaluation mode; use t2_single for SASV Track 2 single-system scoring
  --sasv <sasv_score_file> \           	# path to llr.txt generated in Stage 4
  --sasv_keys <sasv_key_file>        # path to the prepared SASV key file
```

---

### Example

```bash
python wedefense/metrics/detection/evaluation.py \
  --m t2_single \
  --sasv exp/test/posteriors/llr.txt \
  --sasv_keys data/test/sasv_key_file.txt
```

The evaluation results are printed in the terminal.
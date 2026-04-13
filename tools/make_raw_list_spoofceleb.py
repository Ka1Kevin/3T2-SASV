import argparse
import logging
import json


def get_args():
    parser = argparse.ArgumentParser(description='Fixed make_raw_list for SpoofCeleb')
    parser.add_argument('wav_file', help='wav.scp file')
    parser.add_argument('utt2lab_file', help='utt2lab file (refutt#testutt lab)')
    parser.add_argument('spk2utt_file', help='spk2utt file (ignored except for checks)')
    parser.add_argument('raw_list', help='output raw list')
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # 1. Load wav.scp
    wav_table = {}
    utt2spk = {}

    with open(args.wav_file, 'r', encoding='utf-8') as fin:
        for ln, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            arr = line.split()
            if len(arr) < 2:
                logging.warning(f"Skip bad wav.scp line {ln}: {line}")
                continue

            utt = arr[0]
            wav_path = " ".join(arr[1:])
            wav_table[utt] = wav_path

            # extract speaker ID from path (your current assumption)
            parts = wav_path.split('/')
            if len(parts) >= 2:
                spk = parts[-2]
            else:
                spk = "unknown"
            utt2spk[utt] = spk

    logging.info(f"Loaded {len(wav_table)} entries from wav.scp")

    data = []
    skipped = 0

    # 2) Process utt2lab
    with open(args.utt2lab_file, 'r', encoding='utf-8') as fin:
        for ln, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            # Must have at least: "<ref>#<test> <lab>"
            if '#' not in line:
                logging.warning(f"Skip bad utt2lab line {ln} (no #): {line}")
                skipped += 1
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                logging.warning(f"Skip bad utt2lab line {ln} (no lab): {line}")
                skipped += 1
                continue

            key, lab = parts[0], parts[1]

            # key is "refutt#testutt"
            if '#' not in key:
                logging.warning(f"Skip bad utt2lab line {ln} (bad key): {line}")
                skipped += 1
                continue

            ref_utt, test_utt = key.split('#', 1)

            # Robustly skip header / dirty line like "@@ref_utt#@@test_utt label"
            # by checking existence in wav.scp
            if ref_utt not in wav_table or test_utt not in wav_table:
                logging.warning(f"Skip invalid utt2lab line {ln} (utt not in wav.scp): {line}")
                skipped += 1
                continue

            ref_wav = wav_table[ref_utt]
            test_wav = wav_table[test_utt]

            item = {
                "key": key,
                "lab": lab,
                "wav": [ref_wav, test_wav],  # direct use, no sampling
            }
            data.append(item)

    logging.info(f"Generated {len(data)} items; skipped {skipped} lines from utt2lab")

    # 3) Save jsonlines
    with open(args.raw_list, 'w', encoding='utf-8') as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info(f"Saved raw list to {args.raw_list}")


if __name__ == "__main__":
    main()

import argparse
import os
import json
import random

def read_utt2lab(path):
    utt2lab = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 2:
                utt2lab.append((p[0], p[1]))
    return utt2lab


def load_rawlist(path):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            mapping[obj["key"]] = line.rstrip("\n")
    return mapping


def sample_ratio(utt2lab, num, ratio, seed):
    random.seed(seed)
    rt, rn, rs = ratio
    total_r = rt + rn + rs

    n_t = num * rt // total_r
    n_n = num * rn // total_r
    n_s = num * rs // total_r

    print(f"[VAL] Sampling with ratio 1:8:7")
    print(f"  target     = {n_t}")
    print(f"  nontarget  = {n_n}")
    print(f"  spoof      = {n_s}")

    targets = [u for u, l in utt2lab if l == "target"]
    nons    = [u for u, l in utt2lab if l == "nontarget"]
    spoofs  = [u for u, l in utt2lab if l == "spoof"]

    sel_t = random.sample(targets, min(n_t, len(targets)))
    sel_n = random.sample(nons,    min(n_n, len(nons)))
    sel_s = random.sample(spoofs,  min(n_s, len(spoofs)))

    merged = sel_t + sel_n + sel_s
    random.shuffle(merged)

    return merged


def sample_flat(utt2lab, num, seed):
    random.seed(seed)
    all_utts = [u for u, _ in utt2lab]

    print(f"[VAL] Flat-sampling: selecting {num} / {len(all_utts)} utterances")

    if num > len(all_utts):
        return all_utts

    return random.sample(all_utts, num)


def main(args):

    os.makedirs(args.val_dir, exist_ok=True)

    dev_utt2lab = os.path.join(args.dev_dir, "utt2lab")
    dev_rawlist = os.path.join(args.dev_dir, "raw.list")

    utt2lab = read_utt2lab(dev_utt2lab)
    raw_map = load_rawlist(dev_rawlist)

    label_map = dict(utt2lab)

    # -------------------------
    # SELECT MODE
    # -------------------------
    if args.mode == "ratio":
        selected = sample_ratio(utt2lab, args.val_num, args.ratio, args.seed)
    else:
        selected = sample_flat(utt2lab, args.val_num, args.seed)

    print(f"[VAL] Total selected = {len(selected)}")

    # -------------------------
    # Save utt2lab
    # -------------------------
    out_utt2lab = os.path.join(args.val_dir, "utt2lab")
    with open(out_utt2lab, "w") as f:
        for u in selected:
            f.write(f"{u} {label_map[u]}\n")

    print(f"[VAL] Saved utt2lab → {out_utt2lab}")

    # -------------------------
    # Save raw.list
    # -------------------------
    out_raw = os.path.join(args.val_dir, "raw.list")
    missing = 0

    with open(out_raw, "w") as f:
        for u in selected:
            if u in raw_map:
                f.write(raw_map[u] + "\n")
            else:
                missing += 1

    print(f"[VAL] Saved raw.list → {out_raw}")
    if missing > 0:
        print(f"[WARN] Missing {missing} utt keys in raw.list")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ratio", "flat"], required=True)
    parser.add_argument("--dev_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--val_num", type=int, default=5000)
    parser.add_argument("--ratio", nargs=3, type=int, default=[1, 8, 7])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

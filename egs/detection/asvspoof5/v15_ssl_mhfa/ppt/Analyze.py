import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")

########################################
# 1. 路径与阈值配置
########################################

trial_path = "/export/fs05/arts/dataset/ASVspoof5/ASVspoof5.dev.track_2.trial.tsv"

target_durs = {
    2.0: {
        "score": "/export/fs05/ktan17/ssl/wedefense-main/egs/detection/asvspoof5/v15_ssl_mhfa/exp/target_dur/target_dur_2.0/posteriors/dev/llr.txt",
        "thr": 1.32629
    },
    3.0: {
        "score": "/export/fs05/ktan17/ssl/wedefense-main/egs/detection/asvspoof5/v15_ssl_mhfa/exp/target_dur/target_dur_3.0/posteriors/dev/llr.txt",
        "thr": -0.66266
    },
    5.0: {
        "score": "/export/fs05/ktan17/ssl/wedefense-main/egs/detection/asvspoof5/v15_ssl_mhfa/exp/target_dur/target_dur_5.0/posteriors/dev/llr.txt",
        "thr": -0.93526
    }
}

out_dir = Path("/export/fs05/ktan17/ssl/wedefense-main/egs/detection/asvspoof5/v15_ssl_mhfa/analysis/time_2.0_analysis")
out_dir.mkdir(exist_ok=True)

########################################
# 2. 读取 trial 文件（拿 attack 标签）
########################################

trial_df = pd.read_csv(
    trial_path,
    sep=r"\s+",
    header=None,
    names=["spk", "filename", "gender", "attack", "label"]
)

# 只关心 spoof attack
trial_df = trial_df[trial_df["label"] == "spoof"]
trial_df = trial_df[["spk", "filename", "attack"]]

########################################
# 3. 加载 sasv-score 并 merge attack
########################################

def load_sasv_with_attack(score_path: str) -> pd.DataFrame:
    df = pd.read_csv(score_path, sep="\t")

    # 防止列名不一致
    df.columns = [c.strip() for c in df.columns]

    df = df.merge(
        trial_df,
        on=["spk", "filename"],
        how="inner"
    )

    return df

########################################
# 4. 图 1：SASV-score 分布曲线（KDE）+ threshold
########################################

for target_dur, cfg in target_durs.items():
    df = load_sasv_with_attack(cfg["score"])
    thr = cfg["thr"]

    plt.figure(figsize=(9, 5))

    sns.kdeplot(
        data=df,
        x="sasv-score",
        hue="attack",
        common_norm=False,
        linewidth=1.8
    )

    plt.axvline(
        thr,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {thr:.2f}"
    )

    plt.title(f"SASV-score Distribution by Attack (target_dur={target_dur})")
    plt.xlabel("SASV-score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_dir / f"sasv_score_kde_target_dur{target_dur}.png", dpi=300)
    plt.close()

########################################
# 5. 统计 spoof → target 的误判比例
########################################

records = []

for target_dur, cfg in target_durs.items():
    df = load_sasv_with_attack(cfg["score"])
    thr = cfg["thr"]

    for attack, g in df.groupby("attack"):
        mis_rate = (g["sasv-score"] >= thr).mean()
        records.append({
            "target_dur": target_dur,
            "attack": attack,
            "mis_rate": mis_rate
        })

stat_df = pd.DataFrame(records)

########################################
# 6. 图 2：Attack × Head 误判比例柱状图
########################################

plt.figure(figsize=(11, 5))

sns.barplot(
    data=stat_df,
    x="attack",
    y="mis_rate",
    hue="target_dur"
)

plt.ylabel("Misclassification rate (spoof → target)")
plt.xlabel("Attack type")
plt.title("Attack-wise SASV Misclassification Rate")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(out_dir / "attack_misclassification_rate.png", dpi=300)
plt.close()

########################################
# 7. 同时导出统计表（方便写论文）
########################################

stat_df.to_csv(out_dir / "attack_misclassification_rate.csv", index=False)

print("✅ Done. Figures and statistics saved to:", out_dir)

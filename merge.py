import os
import glob
import pandas as pd

BASE_DIR = "/data/Forever_Pan/AGI_sources/CastMaster_new/datasets/SFT_RL_bank"
OUT_PATH = os.path.join(BASE_DIR, "grok_merge.csv")

def merge_and_shuffle_grok(base_dir: str, out_path: str):
    csv_paths = glob.glob(os.path.join(base_dir, "*", "grok.csv"))
    if not csv_paths:
        raise FileNotFoundError("未在任何子目录中找到 grok.csv")

    dfs = []
    for path in csv_paths:
        print(f"读取: {path}")
        df = pd.read_csv(path)
        # 可选：加一列数据集名称，方便以后区分来源
        df["dataset_name"] = os.path.basename(os.path.dirname(path))
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    # 打乱行
    shuffled = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    shuffled.to_csv(out_path, index=False)
    print(f"已保存合并并打乱后的文件到: {out_path}")
    print(f"总行数: {len(shuffled)}")

if __name__ == "__main__":
    merge_and_shuffle_grok(BASE_DIR, OUT_PATH)
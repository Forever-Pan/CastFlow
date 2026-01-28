import os
import pandas as pd

# 修改为你自己的根目录
BASE_DIR = "/data/Forever_Pan/AGI_sources/CastMaster_new/datasets/SFT_RL_bank"

# 新增的列名，可以按需要改，比如 "dataset" 或 "source"
NEW_COL_NAME = "dataset_name"

def add_dataset_name_column(base_dir: str):
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname in ("grok.csv", "grok_final.csv"):
                csv_path = os.path.join(root, fname)
                dataset_name = os.path.basename(root)  # 用当前文件夹名作为数据集名称

                print(f"处理文件: {csv_path}  -> 数据集名称: {dataset_name}")

                # 读入 CSV
                df = pd.read_csv(csv_path)

                # 新增或覆盖一列为数据集名称
                df[NEW_COL_NAME] = dataset_name

                # 写回原文件（覆盖保存）
                df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    add_dataset_name_column(BASE_DIR)
import pandas as pd

# 读取输入文件
input_df = pd.read_csv('datasets/windy_power_test.csv')
# 假设输入文件没有 idx 列，我们需要按顺序生成 idx (0, 1, 2...)
# 或者如果 CastMaster 的 load_data_from_csv 是怎么生成 idx 的？
# load_data_from_csv 从 0 开始递增 idx。
# 这里我们可以假设总共有 len(input_df) // 96 个样本（如果是滑动窗口的话），或者直接看 load_data_from_csv 的逻辑。
# 不过最简单的是读取 output_use_memory_test.log 中的 "Processing sample ... idx=..." 信息。

# 更好的方法是读取输出文件，看 idx 有哪些
output_df = pd.read_csv('results/windy/test_predictions_use_memory.csv', names=['idx', 'model_name', 'answer', 'ground_truth', 'prompt', 'full_answer'], header=0)

# 输出文件中的 idx 列表
existing_idxs = sorted(output_df['idx'].unique())

print(f"Total samples found in output: {len(existing_idxs)}")
print(f"Existing IDs: {existing_idxs}")

# 假设样本 ID 是连续的 0 到 49 (共 50 个)
expected_idxs = list(range(50))

missing_idxs = [idx for idx in expected_idxs if idx not in existing_idxs]

print(f"Missing IDs: {missing_idxs}")

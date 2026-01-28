# 数据集配置参数指南

本文档列出除了Prompt外，需要根据数据集精细设计的所有参数及其调整位置。

## 一、核心评估阈值（最重要）

### 1. **EXCELLENT_MSE_THRESHOLD** - MSE阈值
- **位置**: `castmaster/castmaster_agent.py` 第60行
- **当前默认值**: `5000`
- **作用**: 判断训练阶段的样本是否足够优秀（excellent），用于决定是否保存到Memory库
- **调整方法**:
  ```bash
  # 方法1: 环境变量（推荐）
  export EXCELLENT_MSE_THRESHOLD=5000
  
  # 方法2: 直接修改代码
  EXCELLENT_MSE_THRESHOLD: float = float(os.getenv("EXCELLENT_MSE_THRESHOLD", 5000))  # 修改默认值
  ```
- **使用位置**: 
  - 第1238行：`metrics_acceptable = (mse < EXCELLENT_MSE_THRESHOLD) and (mae < EXCELLENT_MAE_THRESHOLD)`
  - 第1257-1258行：打印excellent样本信息

### 2. **EXCELLENT_MAE_THRESHOLD** - MAE阈值
- **位置**: `castmaster/castmaster_agent.py` 第61行
- **当前默认值**: `50`
- **作用**: 与MSE阈值一起判断样本是否优秀
- **调整方法**:
  ```bash
  # 方法1: 环境变量（推荐）
  export EXCELLENT_MAE_THRESHOLD=50
  
  # 方法2: 直接修改代码
  EXCELLENT_MAE_THRESHOLD: float = float(os.getenv("EXCELLENT_MAE_THRESHOLD", 50))  # 修改默认值
  ```
- **使用位置**: 同MSE阈值（第1238行）

**建议调整策略**:
- 根据数据集的数值范围调整。例如：
  - 如果数据范围是0-100，MAE阈值可以设为10-20
  - 如果数据范围是0-1000，MAE阈值可以设为100-200
  - MSE阈值通常是MAE阈值的平方的2-5倍

---

## 二、时间序列窗口参数

### 3. **预测窗口长度 (forecast/predicted_window)**
- **硬编码位置**:
  - `castmaster/castmaster_agent.py` 第458行: `forecast = 96`
  - `castmaster/castmaster_agent.py` 第83行: `CASES_LIBRARY_PREDICTED_WINDOW` (默认96)
  - `castmaster/castmaster_agent.py` 多处硬编码96（第1110, 1114, 1015, 1020, 1021, 1044, 1146, 1597行）
- **作用**: 定义需要预测多少个时间点
- **调整方法**: 
  - 需要修改多处硬编码，建议：
    1. 在文件顶部定义常量：`PREDICTED_WINDOW = int(os.getenv("PREDICTED_WINDOW", 96))`
    2. 替换所有硬编码的96为这个常量

### 4. **历史窗口长度 (look_back)**
- **位置**: `castmaster/castmaster_agent.py` 第82行
- **当前默认值**: `96`
- **作用**: 用于案例库构建，定义历史窗口长度
- **调整方法**:
  ```bash
  export CASES_LIBRARY_LOOK_BACK=96
  ```

### 5. **季节性周期 (seasonal/season_length)**
- **硬编码位置**: `castmaster/castmaster_agent.py` 第460行: `seasonal = 24`
- **作用**: 用于时间序列模型的季节性参数
- **调整方法**: 
  - 需要根据数据集的周期性调整
  - 例如：15分钟数据，日周期=96，周周期=672
  - 建议改为可配置：`seasonal = int(os.getenv("SEASONAL_PERIOD", 24))`

---

## 三、Memory相似度参数

### 6. **MEMORY_SIM_THRESHOLD_TRAIN** - 训练阶段相似度阈值
- **位置**: `castmaster/castmaster_agent.py` 第67行
- **当前默认值**: `0.8`
- **作用**: 训练阶段，从Memory中检索相似样本的阈值（余弦相似度）
- **调整方法**:
  ```bash
  export MEMORY_SIM_THRESHOLD_TRAIN=0.9
  ```
- **使用位置**: 第614行

### 7. **MEMORY_SIM_THRESHOLD_TEST** - 测试阶段相似度阈值
- **位置**: `castmaster/castmaster_agent.py` 第70行
- **当前默认值**: `0.9`
- **作用**: 测试阶段，从Memory中检索相似样本的阈值
- **调整方法**:
  ```bash
  export MEMORY_SIM_THRESHOLD_TEST=0.9
  ```
- **使用位置**: 第639行

### 8. **MEMORY_TOPK_TEST** - 测试阶段Top-K
- **位置**: `castmaster/castmaster_agent.py` 第73行
- **当前默认值**: `3`
- **作用**: 测试阶段最多参考的相似样本数
- **调整方法**:
  ```bash
  export MEMORY_TOPK_TEST=3
  ```

### 9. **MemoryLibrary.similarity_threshold** - Memory库相似度阈值
- **位置**: `castmaster/memory_library.py` 第32行
- **硬编码位置**: `castmaster/castmaster_agent.py` 第1474行: `similarity_threshold=0.1`
- **当前默认值**: `0.1`
- **作用**: Memory库初始化时的相似度阈值（用于`add_memory`时的去重判断）
- **合并逻辑**: 
  - 在`add_memory`方法中，如果`dedup=True`（默认），会检查新记忆与已有记忆的相似度
  - 合并条件：`similarity > (1.0 - similarity_threshold)`
  - 当`similarity_threshold=0.1`时，条件为`similarity > 0.9`（余弦相似度超过90%才会合并）
- **重要说明**:
  - 在训练阶段（`castmaster_agent.py`第1270行），调用`add_memory`时设置了`dedup=False`，**因此不会发生合并**
  - 只有在其他场景下调用`add_memory`且`dedup=True`时，才会根据此阈值判断是否合并
- **调整方法**: 修改第1474行
  ```python
  memory_library = MemoryLibrary(similarity_threshold=0.1, use_faiss=True)  # 修改0.1
  ```
- **调整建议**:
  - 更严格去重（更不容易合并）：减小阈值，如`0.05` → 条件变为`similarity > 0.95`
  - 更宽松去重（更容易合并）：增大阈值，如`0.2` → 条件变为`similarity > 0.8`

---

## 四、案例库构建参数

### 10. **CASES_LIBRARY_SLIDING_WINDOW** - 滑动窗口步长
- **位置**: `castmaster/castmaster_agent.py` 第84行
- **当前默认值**: `24`
- **作用**: 构建案例库时的滑动窗口步长
- **调整方法**:
  ```bash
  export CASES_LIBRARY_SLIDING_WINDOW=24
  ```

### 11. **CASES_LIBRARY_NUM_CLUSTERS** - 聚类数量
- **位置**: `castmaster/castmaster_agent.py` 第86行
- **当前默认值**: `6`
- **作用**: K-medoid聚类的簇数
- **调整方法**:
  ```bash
  export CASES_LIBRARY_NUM_CLUSTERS=6
  ```

### 12. **CASES_LIBRARY_METHOD** - 聚类方法
- **位置**: `castmaster/castmaster_agent.py` 第85行
- **当前默认值**: `"weighted"`
- **作用**: 聚类方法（weighted/unweighted等）
- **调整方法**:
  ```bash
  export CASES_LIBRARY_METHOD=weighted
  ```

---

## 五、时间序列嵌入参数

### 13. **时间序列降采样点数 (n_samples)**
- **位置**: `castmaster/memory_library.py` 第85行
- **当前硬编码值**: `24`
- **作用**: 将时间序列嵌入为向量时的降采样点数
- **调整方法**: 直接修改代码
  ```python
  n_samples = 24  # 修改为合适的值，建议为预测窗口长度的1/4到1/2
  ```
- **建议**: 如果预测窗口是96，可以设为24或48

---

## 六、控制流程参数

### 14. **MAX_REFLECTION_LOOPS** - 最大反思轮数
- **位置**: `castmaster/castmaster_agent.py` 第64行
- **当前默认值**: `3`
- **作用**: 反思-规划循环的最大轮数
- **调整方法**:
  ```bash
  export MAX_REFLECTION_LOOPS=3
  ```

### 15. **PARALLEL_PLAN_K** - 并行策略数
- **位置**: `castmaster/castmaster_agent.py` 第76行
- **当前默认值**: `4`
- **作用**: 训练阶段每一轮探索的候选工具策略数量
- **调整方法**:
  ```bash
  export PARALLEL_PLAN_K=4
  ```

---

## 七、时间间隔参数（硬编码）

### 16. **时间戳间隔**
- **硬编码位置**: `castmaster/castmaster_agent.py` 第1016行: `"Timestamps increment by 15 minutes."`
- **作用**: 在Prompt中告知LLM时间戳的间隔
- **调整方法**: 需要修改Prompt中的描述

---

## 八、数据集名称

### 17. **DATASET_NAME**
- **位置**: `castmaster/castmaster_agent.py` 第80行
- **当前默认值**: `"windy_power"`
- **作用**: 数据集名称，用于案例库目录结构
- **调整方法**:
  ```bash
  export DATASET_NAME=your_dataset_name
  ```

---

## 快速配置示例

创建一个 `.env` 文件或设置环境变量：

```bash
# 核心评估阈值（最重要！）
export EXCELLENT_MSE_THRESHOLD=5000
export EXCELLENT_MAE_THRESHOLD=50

# 时间窗口参数
export CASES_LIBRARY_LOOK_BACK=96
export CASES_LIBRARY_PREDICTED_WINDOW=96
export PREDICTED_WINDOW=96  # 如果添加了这个参数

# Memory相似度参数
export MEMORY_SIM_THRESHOLD_TRAIN=0.9
export MEMORY_SIM_THRESHOLD_TEST=0.9
export MEMORY_TOPK_TEST=3

# 案例库参数
export CASES_LIBRARY_SLIDING_WINDOW=24
export CASES_LIBRARY_NUM_CLUSTERS=6
export CASES_LIBRARY_METHOD=weighted

# 控制流程参数
export MAX_REFLECTION_LOOPS=3
export PARALLEL_PLAN_K=4

# 数据集名称
export DATASET_NAME=windy_power
```

---

## 优先级建议

### 高优先级（必须调整）
1. **EXCELLENT_MSE_THRESHOLD** 和 **EXCELLENT_MAE_THRESHOLD** - 直接影响Memory库的质量
2. **预测窗口长度** - 需要与数据集匹配
3. **季节性周期** - 影响模型预测效果

### 中优先级（建议调整）
4. **MEMORY_SIM_THRESHOLD_TRAIN/TEST** - 影响Memory检索的准确性
5. **CASES_LIBRARY_SLIDING_WINDOW** - 影响案例库的密度
6. **时间序列降采样点数** - 影响相似度计算的准确性

### 低优先级（可选调整）
7. **MAX_REFLECTION_LOOPS** - 影响训练时间
8. **PARALLEL_PLAN_K** - 影响探索策略数量
9. **CASES_LIBRARY_NUM_CLUSTERS** - 影响聚类结果

---

## 注意事项

1. **MSE/MAE阈值调整建议**:
   - 先运行少量样本，观察MSE和MAE的分布
   - 设置阈值使得约10-20%的样本能被判定为excellent
   - 阈值太严格：Memory库样本太少
   - 阈值太宽松：Memory库质量下降

2. **预测窗口长度**:
   - 目前代码中多处硬编码96，建议重构为统一常量
   - 需要同时修改Prompt中的描述

3. **季节性周期**:
   - 需要根据数据集的真实周期性设置
   - 15分钟数据：日周期=96，周周期=672
   - 1小时数据：日周期=24，周周期=168

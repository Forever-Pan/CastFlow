CastMaster: 时序预测大模型智能体基础框架

功能概述
- 规划模块：使用 gpt-4o 生成 YAML 规划，长期/短期规划。
- 行动模块：按规划执行特征工程与训练/推理，计划-执行-反馈循环。
- 预测模块：包含季节性 Naive、ARIMA、树模型包装、LSTM。
- 反思模块：MSE/MAE/SMAPE 指标评估与回退策略。
- 记忆模块：向量存储（FAISS/NearestNeighbors）与策略图检索。
- 流程：训练集 70% 滚动回测选择最佳方案，测试集 30% 单次滚动预测。
 - LLM 主预测（可选）：测试阶段由 LLM 作为主体预测，综合最佳小模型的辅助预测、外生变量与特征信息进行修正与输出。

使用步骤
1) 安装依赖：`pip install -r requirements.txt`
2) 准备数据 CSV（含时间戳、18 外生变量、目标列）。
3) 运行：
   - 推荐：`python -m castmaster --data path/to/wind.csv --output output/`
   - 亦可：`python -m castmaster.cli --data path/to/wind.csv --output output/`
4) 如需 LLM 规划，使用 .env 自动配置（无需每次手动 export）：
   - 在项目根或工作目录创建 `.env` 文件，示例如下：
     ```
     OPENAI_BASE_URL="https://api2.aigcbest.top/v1"
     OPENAI_API_KEY="你的密钥"
     MODEL="gpt-4o"
     ```
   - 包在导入时会自动加载 `.env`（依赖 python-dotenv），也会尝试读取示例路径：`/data/Forever_Pan/AGI_sources/CastMaster_new/.env`。
   - 规划模块会使用 `.env` 中的 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 与 `MODEL`。
5) 启用 LLM 主预测与外生特征：
   - 在 `configs/default.yaml` 的 `planning` 部分设置：
     - `llm_as_primary: true` 让 LLM 成为主体预测器（测试阶段）。
     - `top_k_exog: 3` 选取与目标皮尔逊相关性最高的若干外生变量。
     - `use_featuretools_time_features: true` 使用 featuretools 的时间特征增强。
   - 运行后，`output/test_metrics.csv` 会包含 `llm_api_called`（是否成功调用 LLM API）与 `llm_confidence` 字段。

特征提取说明
- 目标特征：统计特征（均值/方差/偏度/峰度）、谱熵、tsfeatures（如 acf_features/entropy/lumpiness/flat_spots 等，自动回退）。
- 外生变量：对数值型外生变量逐一提取相同特征，并计算与目标的皮尔逊相关系数；自动选取相关性最高的 `top_k_exog` 个变量传入 LLM。
- 时间特征：若可用，使用 `featuretools` 提取 weekday/month/hour/day/week/is_weekend 等时间特征；否则回退到内置日历 + 傅里叶特征。

目录结构
- castmaster/：核心代码
- configs/default.yaml：默认配置
- output/：结果输出目录

内存（Memory）格式与作用
- 向量存储（`castmaster/memory/vector_store.py`）
  - 存储格式：在运行结束时保存到 `output/memory_vectors.json`，结构为 `{dim: int, items: [{vec: List[float], meta: dict}, ...]}`。
  - 存储信息：
    - `vec` 为窗口统计嵌入（均值+标准差）。
    - `meta` 为该窗口的描述，如 `{"desc": "last train window"}`。
  - 作用：在新一次运行开始时加载（自动读取 `output/memory_vectors.json`），用于检索相似上下文并作为 LLM 规划的记忆输入。
- 策略图（`castmaster/memory/strategy_graph.py`）
  - 存储格式：可保存为 `{edges: [[from, to, weight], ...]}`（在代码中提供 `save/load`，默认运行未写出该文件）。
  - 存储信息：工具/策略之间的有向边与权重，用于记录“从某分析选择某工具”的转移强度。
  - 作用：为后续基于图的工具选择与经验复用做准备，便于统计何时使用差分、傅里叶等工具。
- 运行记忆日志（`output/memory_log.json`）
  - 存储格式：`{"analysis": {...}, "tool_suggestions": [...], "best_model": str, "llm_api_called": bool, "llm_confidence": float}`。
  - 存储信息：本次运行的序列分析（趋势/季节性/平稳性指标）、基于分析推荐的工具清单、回测选出的最佳小模型，以及 LLM 调用状态与置信度。
  - 作用：为后续运行提供可读的经验摘要，LLM 也可据此调整规划。

分析辅助工具（新增）
- 趋势分析：`analyze_trend(y)` 返回 `trend_slope/trend_r2/trend_strength/trend_direction`。
- 季节性分析：`analyze_seasonality(y, period, pandas_freq)` 返回 `seasonal_acf/seasonal_strength/stl_seasonal_strength` 等。
- 平稳性分析：`analyze_stationarity(y)` 返回 `adf_pvalue/is_stationary`。
- 综合分析：`analyze_series(y, period, pandas_freq)` 汇总趋势/季节性/平稳性。
- 工具建议器：`select_tools_based_on_analysis(analysis)` 基于分析自动建议 `detrend_linear/fourier_features/difference/calendar_features`。

LLM 的自适应规划与工具选择
- 在 `planning.llm_forecast_primary` 中，会将综合分析与工具建议一并发送给 LLM，鼓励其在推理中参考这些工具（如差分/去趋势/傅里叶）以修正辅助预测。
- `pipeline.run_pipeline` 在运行结束会将分析与工具建议写入 `output/memory_log.json`，并将向量存储保存为 `output/memory_vectors.json`。下次运行加载这些记忆以辅助规划。
- 如需扩展为“策略图记忆”，可在每次工具使用后调用 `StrategyGraph.add_edge` 增加边权，并通过 `save/load` 跨运行复用。

风力发电预测任务说明与物理先验（新增）
- 任务目标与范围：针对新能源风力发电场的短期功率预测，默认以 `15min` 采样，`H=96`（24 小时历史）与 `F=96`（24 小时前瞻）为基线。目标是在常见的切入/额定/切出风速条件、并网与限电约束下，结合物理先验与外生变量进行稳定、高可信的预测。
- 目标变量（`power/real_power`）：机组/场站在给定时刻的真实有功功率（单位一般为 `kW/MW`），受风速立方关系、空气密度、风机控制（桨距/偏航/转速）以及电网侧约束（限发、频率、电压）共同影响。
- 常见外生变量与物理含义（示例列名，具体以数据集为准）：
  - `wind_speed`/`wind_speed_80m`/`wind_speed_hub`：风速（米/秒，越接近轮毂高度越好），是功率的主导因子。
  - `wind_direction`/`nacelle_direction`：风向与机舱朝向（度），用于计算偏航失配（yaw misalignment）。
  - `air_density`/`temperature`/`pressure`/`humidity`：空气密度直接影响功率系数前的比例项，密度随气压升高、温度降低而增加；湿度略微降低密度。
  - `gust`/`gust_speed`：阵风强度，提示短时风速跃迁、切出风险与功率尖峰。
  - `turbulence_intensity`（TI）：湍流强度，高 TI 在高风速下会降低有效 `Cp` 并增加切出风险。
  - `rotor_speed`（转速，`rpm`）：与风速共同决定叶尖速比 `λ`，影响 `Cp` 与功率平台期的控制策略。
  - `pitch_angle`（桨距角，度）：接近或超过额定风速时增大桨距以限功，导致有效 `Cp` 下降。
  - `yaw_angle`（偏航角，度）：偏航误差增大时有功功率下降，近似余弦衰减。
  - `availability`/`curtailment_flag`：可用性与限电标志，直接约束输出功率。
  - `grid_frequency`/`grid_voltage`：电网侧约束与并网控制，对额定区间的功率输出有次级影响。
- 物理函数借鉴关系（可用于特征工程与 LLM 推理）：
  - 基本功率公式：`power ≈ 0.5 * ρ * A * Cp(λ, β) * v^3`，其中 `ρ` 为空气密度，`A` 为扫掠面积，`Cp` 为功率系数，`v` 为风速，`λ = (ωR)/v` 为叶尖速比，`β` 为桨距角。
  - 饱和与分段：功率随风速近似立方增长，至额定风速附近饱和（平台），超过切出风速功率降为零；可用 `v^3` 与分段/逻辑斯蒂（sigmoid）近似平台化。
  - 空气密度近似：`ρ ≈ p / (R_specific * T)`（湿空气时略小），在相同风速下，密度越高功率越大，可构造 `ρ * v^3` 作为先验强特征。
  - 偏航失配惩罚：`P_eff ≈ P * cos^k(Δθ)`，`Δθ` 为风向与机舱朝向差，`k ∈ [1, 2]`，建议对 `wind_direction` 做正余弦编码并加入失配项。
  - 桨距/转速控制：`Cp` 随 `β` 增大而下降，`λ` 接近最优值时 `Cp` 最大；可在高风速区引入单调递减的 `pitch_angle` 惩罚或以 `rotor_speed`（含 1–2 步滞后）代理控制状态。
  - 湍流影响：`Cp_eff ← Cp_eff * (1 - c * TI)`，`c` 为小常数（经验 0.1–0.3），高 TI 时对平台期与切出的风险提示更强。
  - 风向周期性：对 `wind_direction` 使用 `sin(θ)`/`cos(θ)` 以建模圆周结构；配合地形/来流主导方向可引入分段权重。
- 物理先验特征工程建议（与 CastMaster 的通用特征并行使用）：
  - 衍生强特征：`v^3`、`ρ * v^3`、`cos(Δθ)`、`TI` 与 `TI * v`、`pitch_angle` 的单调惩罚项、`rotor_speed / v` 近似 `λ`、`direction_sin/cos`。
  - 分段与饱和：基于 `cut_in/rated/cut_out` 风速（例如 3/12–15/25 m/s）进行分段，平台区约束预测不超过额定功率；可用 `min(power_pred, rated_power)` 或逻辑斯蒂近似。
  - 滞后/控制响应：对 `yaw_angle`、`pitch_angle`、`rotor_speed` 引入 1–3 步滞后特征以刻画控制延迟。
  - 稳健性：对异常的 `gust/TI` 使用 Winsorize/分位裁剪，配合 CastMaster 内置的异常值鲁棒合并与标准化。
- 先验约束（可供模型与 LLM 参考）：
  - 非负性与平台约束：`power ≥ 0` 且在额定区间上限受限。
  - 单调性（额定前）：对风速至额定前的区间，功率对 `v` 单调不减；密度提高时同向放大。
  - 物理不变性：风向是圆周变量，应使用正余弦；密度由气压与温度共同决定；限电/可用性直接约束输出。
- 在 CastMaster_new 中使用这些先验（不改动代码即可启用的做法）：
  - 列命名与映射：确保数据集中包含或映射为上述外生列名（若名称不同，可在导入前重命名）。系统会自动筛选数值外生变量并提取统计与 `tsfeatures`；无需强制在 `configs/default.yaml` 中列出。
  - 物理强特征的传递：管线在测试阶段会选择与目标皮尔逊相关性最高的若干外生变量，并连同统计特征发送给 LLM 主预测器，LLM 可依据上述物理先验做校正（如 `ρ v^3`、`cos(Δθ)`）。
  - 若需手工增强：可以在你自己的数据预处理环节（或扩展 `features.py`）添加 `v^3/ρ v^3/sin/cos(yaw mismatch)/滞后控制` 等特征列，CastMaster 会自动将其视为外生变量纳入特征提取与相关性评估。
- 配置示例（仅示例，不存在的列将被自动忽略）：
  - 在 `configs/default.yaml` 中，可按需添加：
    - `data.exogenous_cols: [wind_speed_hub, wind_direction, nacelle_direction, air_density, temperature, pressure, humidity, gust, turbulence_intensity, rotor_speed, pitch_angle, yaw_angle, availability, grid_frequency, grid_voltage, curtailment_flag]`
  - 若数据集未包含某些列，无需修改配置，系统会自动从现有数值列中选择外生变量。
- 单位与数据质量建议：
  - 风速 `m/s`，风向 `degree`，温度 `°C`，气压 `hPa`，功率 `kW/MW`；方向建议做正余弦编码；功率建议统一为同一量纲。
  - 建议保证轮毂高度的一致性，若有多个高度风速，可采用鲁棒合并（均值/中位数/去异常平均）。
  - 保留限电与可用性标志，避免将其视为噪声；异常值处理与标准化已在管线中提供鲁棒回退。
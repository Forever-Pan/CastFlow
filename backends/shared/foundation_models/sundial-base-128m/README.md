---
license: apache-2.0
metrics:
- mse
- mae
- mase
- wql
- crps
pipeline_tag: time-series-forecasting
datasets:
- thuml/UTSD
- Salesforce/lotsa_data
- autogluon/chronos_datasets
tags:
- time series
- time-series
- forecasting
- foundation models
- pretrained models
- generative models
- time series foundation models
---


# Sundial

🚩 **News (2025.06)** Sundial has been accepted as **ICML 2025 Oral** (Top 1%).

🚩 **News (2025.05)** Get **1st MASE** on the [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) Benchmark.

🚩 **News (2025.02)** Get **1st MSE/MAE** zero-shot performance on [Time-Series-Library](https://github.com/thuml/Time-Series-Library) datasets.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/xoSJYO6GSHeFKY9eLjNz2.png)

Sundial is a family of **generative** time series foundation models. The model can make zero-shot predictions for **point** and **probabilistic** forecasting. [[Slides]](https://cloud.tsinghua.edu.cn/f/8d526337afde465e87c9/) [[Poster]](https://cloud.tsinghua.edu.cn/f/cc2a156315e9453f99b3/) [[Intro (CN)]](https://mp.weixin.qq.com/s/y3sc2e2lmW1sqfnoK-ZdDA).

Not only the mean or quantiles, you can get any statistical predictions with a set of generated samples.

The base version is pre-trained on **1 trillion** time points with **128M** parameters. For more information, please refer to this [paper](https://arxiv.org/pdf/2502.00816).

**Sundial** can be viewed as an **ARMA** model (Auto-Regression and Moving-Average). Transformer learns auto-regressive token representations. Conditioned on them, TimeFlow transforms random noises into non-deterministic predictions.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/B5w-TNPnTBpChexIhsVOp.png)

**Overall Architecture**: The input time series is divided into patch tokens, which are embedded from the original continuous values. The patch embeddings are fed into a decoder-only Transformer, a stable and speedup version that learns token representations. The model is optimized using our TimeFlow Loss, a parameterized loss function that models per-token probability distribution conditioned on the learned representations, and generates multiple plausible predictions under the flow-matching framework.

## Quickstart
```
pip install transformers==4.40.1 # Use this version and Python 3.10 for stable compatibility
```

```
import torch
from transformers import AutoModelForCausalLM

# load pretrain model
# supports different lookback/forecast lengths
model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True) 

# prepare input
batch_size, lookback_length = 1, 2880 
seqs = torch.randn(batch_size, lookback_length)

# Note that Sundial can generate multiple probable predictions
forecast_length = 96 
num_samples = 20

output = model.generate(seqs, max_new_tokens=forecast_length, num_samples=num_samples)

# use raw predictions for mean/quantiles/confidence-interval estimation
print(output.shape) 
```

More examples for predicting quantiles or confidence intervals are provided in this [notebook](https://github.com/thuml/Sundial/blob/main/examples/quickstart_zero_shot_generation.ipynb).

## Evaluation

We evaluate performance on the following benchmarks:

- [GIFT-Eval (1st MASE)](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/3BxatwayhK5GAoqMf1oHv.png) [[Leaderboard]](https://huggingface.co/spaces/Salesforce/GIFT-Eval).
- [Time-Series-Library (1st MSE/MAE)](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/5VqnFwWTWoYz877Zkluiw.png).
- [FEV Leaderboard](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/mrKL9QmX-aX8rCiwxKgmA.png).

We are actively working around it and are glad to hear suggestions and noteworthy cases :)

## Inference Time

* Hardware: Apple M1 Pro CPU (16 GB)

| Lookback Length | Prediction Length | # Generated Samples | Inference Time | Accelerate By          |
| --------------- | ----------------- | ------------------- | -------------- | --------------         |
| 672             | 16                | 1                   | 249ms          |       -                |
| 2880            | 16                | 1                   | 510ms          | FlashAttention         |
| 2880            | 720               | 1                   | 510ms          | Multi-Patch Prediction |
| 2880            | 1440              | 1                   | 789ms          | KV Cache               |
| 2880            | 720               | 20                  | 949ms          | Shared Condition       |

* Hardware: A100-40G GPU, following [Chronos](https://arxiv.org/abs/2403.07815) paper.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/hCxzX2MbcNk1XfTe_wLfC.png)

## Specification

* **Architecture**: Causal Transformer (Decoder-only)
* **Pre-training Scale**: 1032B time points
* **Context Length**: up to 2880
* **ReNorm**: Default=True
* **Patch Length**: 16
* **Multi-Patch Prediction Lengt**h: 720
* **Parameter Count**: 128M
* **Number of Layers**: 12
* **Precision**: FP32
* **Speedup**: KV Cache & FlashAttention

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (62022050 and U2342217), the BNRist Innovation Fund (BNR2024RC01010), and the National Engineering Research Center for Big Data Software. 

The model is mostly built from the Internet public time series dataset, which comes from different research teams and providers. We sincerely thank all individuals and organizations who have contributed the data. Without their generous sharing, this model would not have existed.

## Citation

If you find Sundial helpful for your research, please cite our paper:
```
@article{liu2025sundial,
  title={Sundial: A Family of Highly Capable Time Series Foundation Models},
  author={Liu, Yong and Qin, Guo and Shi, Zhiyuan and Chen, Zhi and Yang, Caiyin and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2502.00816},
  year={2025}
}
```

## Contact

If you have any questions or want to use the code, feel free to contact:

* Yong Liu (liuyong21@mails.tsinghua.edu.cn)
* Guo Qin (qinguo24@mails.tsinghua.edu.cn)

## License

This model is licensed under the Apache-2.0 License.
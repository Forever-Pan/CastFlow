vllm serve ./models/merge/sft_qwen3_4B \
  --port 8003 \
  --max-model-len 18000 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --max-num-seqs 100
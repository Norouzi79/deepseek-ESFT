
export TOKENIZERS_PARALLELISM=false

exp_name="test/eval_intent"
base_model_path="deepseek-ai/ESFT-vanilla-lite"
torchrun --nproc-per-node=8 train_ep.py \
    --base_model_path=${base_model_path} \
    --expert_config=results/expert_configs/intent.json \
    --train_dataset=intent \
    --save_opt_states \
    --train_config=configs/base.yaml \
    --output_dir=results/checkpoints/${exp_name}



cp results/expert_configs/intent.json results/checkpoints/${exp_name}/checkpoint-1/expert_cfg.json


python eval_multigpu.py \
    --eval_dataset=intent \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --adapter_dir=results/checkpoints/${exp_name}/checkpoint-1 \
    --output_path=results/completions/token/intent.jsonl \
    --max_new_tokens=512 \
    --eval_batch_size=2 \
    --world_size=4 \
    --openai_api_key=REPLACE_WITH_YOUR_KEY \
    --gpus_per_rank=2
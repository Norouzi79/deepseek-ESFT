export PYTHONPATH=$PYTHONPATH:$(pwd)

python scripts/expert/get_expert_scores.py \
    --eval_dataset=intent \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --output_dir=results/expert_scores/intent \
    --n_sample_tokens=131072 \
    --world_size=4 \
    --gpus_per_rank=2

python scripts/expert/generate_expert_config.py \
    --eval_dataset=intent \
    --expert_scores_dir=results/expert_scores/intent \
    --output_path=results/expert_configs/intent.json \
    --score_function=token \
    --top_p=0.2 # the scoring function and top_p are hyperparameters
    # --train_shared_experts
    # --train_non_expert_modules

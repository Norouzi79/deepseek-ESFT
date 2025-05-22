# first: download adapter models and put them to the corresponding directories


python eval_multigpu.py \
    --eval_dataset=intent \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --adapter_dir=all_models/adapters/token/intent \
    --output_path=results/completions/token/intent.jsonl \
    --max_new_tokens=512 \
    --eval_batch_size=2 \
    --world_size=4 \
    --openai_api_key=REPLACE_WITH_YOUR_KEY \
    --gpus_per_rank=2


# below script is used for single-gpu training and has been deprecated. If you have only one gpu, you can set above world_size=1 and gpus_per_rank=1

# python scripts/eval.py \
#     --eval_datasets=translation \
#     --base_model_path=deepseek-ai/ESFT-vanilla-lite \
#     --adapter_dir=all_models/adapters/token \
#     --output_dir=results/completions/token \
#     --max_new_tokens=512 \
#     --openai_api_key=REPLACE_WITH_YOUR_KEY \
#     --eval_batch_size=2

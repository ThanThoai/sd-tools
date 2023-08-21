accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py \
    --train_data_dir data/crop_resize \
    --caption_extension caption \
    --output_dir exp1  \
    --sample_prompts samples.txt \
    --pretrained_model_name_or_path dreamshaper_8.safetensors \
    --in_json metadata_caption_subword.json \
    --save_model_as=safetensors \
    --prior_loss_weight 1.0 \
    --max_train_steps 2000 \
    --learning_rate 0.000005 --resolution 512 \
    --optimizer_type AdamW --xformers --train_batch_size 2 --sample_every_n_steps 100 --save_every_n_steps 500 --mixed_precision fp16 --token_string bxthoai  --num_vectors_per_token 1 --use_object_template --clip_skip 2
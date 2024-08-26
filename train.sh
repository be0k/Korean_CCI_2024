for i in {0..9}
do
    python src/train.py\
    --model_id kihoonlee/STOCK_SOLAR-10.7B\
    --tokenizer kihoonlee/STOCK_SOLAR-10.7B\
    --fold_mode True\
    --fold_num 10\
    --fold_idx $i\
    --batch_size 1\
    --gradient_accumulation_steps 4\
    --warmup_steps=-2\
    --lr 0.00005\
    --epoch 10\
    --weight_decay 0.1\
    --seed 42\
    --tokenizer_parallel True\
    --change_name True\
    --quant_allow False\
    --quant_4bit False\
    --quant_4bit_double False\
    --quant_4bit_compute_dtype bfloat16\
    --quant_8bit False\
    --model_dtype bfloat16\
    --lora_rank 16\
    --lora_alpha 32\
    --lora_dropout 0\
    --lora_bias none\
    --train_path ./data/train.json\
    --dev_path ./data/dev.json\
    --save_dir output/fold$i
done

python src/train.py\
    --model_id kihoonlee/STOCK_SOLAR-10.7B\
    --tokenizer kihoonlee/STOCK_SOLAR-10.7B\
    --fold_mode False\
    --fold_num 10\
    --fold_idx 0\
    --batch_size 1\
    --gradient_accumulation_steps 4\
    --warmup_steps=-2\
    --lr 0.00005\
    --epoch 10\
    --weight_decay 0.1\
    --seed 42\
    --tokenizer_parallel True\
    --change_name True\
    --quant_allow False\
    --quant_4bit False\
    --quant_4bit_double False\
    --quant_4bit_compute_dtype bfloat16\
    --quant_8bit False\
    --model_dtype bfloat16\
    --lora_rank 16\
    --lora_alpha 32\
    --lora_dropout 0\
    --lora_bias none\
    --train_path ./data/train.json\
    --dev_path ./data/dev.json\
    --save_dir output/fold10
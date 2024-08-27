from src.data import TrainDataset, DevDataset, DataCollatorForSupervisedDataset
from run.trainer import Llama3ForSFT
from src.utils import CustomMetrics, merge_data, get_directory_path

import wandb
wandb.init(mode="disabled")

import argparse
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig,TaskType
from trl import SFTTrainer, SFTConfig



# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, default='kihoonlee/STOCK_SOLAR-10.7B',  help="model file path")
g.add_argument("--tokenizer", type=str, default='kihoonlee/STOCK_SOLAR-10.7B', help="huggingface tokenizer path")
g.add_argument("--fold_mode", type=bool, default=True, help="k-fold mode")
g.add_argument("--fold_num", type=int, default=10, help="fold number (k)")
g.add_argument("--fold_idx", type=int, default=0, help="fold index (0 ~ k-1)")

##train parameters
g.add_argument("--batch_size", type=int, default=1, help="batch size (both train and eval)")
g.add_argument("--gradient_accumulation_steps", type=int, default=4, help="gradient accumulation steps")
g.add_argument("--warmup_steps", type=int, default=-1, help="scheduler warmup steps")
g.add_argument("--lr", type=float, default=5e-5, help="learning rate")
g.add_argument("--epoch", type=int, default=10, help="training epoch")
g.add_argument("--weight_decay", type=float, default=0.1, help="weight_decay")
g.add_argument("--seed", type=int, default=42, help="seed")

g.add_argument("--tokenizer_parallel", type=bool, default=True, help="set True if you want tokenizers_parrallelism")#
g.add_argument("--change_name", type=bool, default=True, help="change \"name\" to \"화자\" if True")
g.add_argument("--quant_allow", type=bool, default=False, help="Must be set to True for quantization.")
g.add_argument("--quant_4bit", type=bool, default=False, help="4bit quantization (load_in_4bit)")
g.add_argument("--quant_4bit_double", type=bool, default=False, help="4bit double quantization (bnb_4bit_use_double_quant)")
g.add_argument("--quant_4bit_compute_dtype", type=str, default='bfloat16', help="bnb_4bit_quant_type(float32, bfloat16, float16)")
g.add_argument("--quant_8bit", type=bool, default=False, help="8bit quantization (load_in_8bit)")
g.add_argument("--model_dtype", type=str, default="bfloat16", help="model dtype (torch_dtype)")

#lora parameters
g.add_argument("--lora_rank", type=int, default=16, help="lora rank")
g.add_argument("--lora_alpha", type=int, default=32, help="lora alpha")
g.add_argument("--lora_dropout", type=float, default=0., help="lora dropout")
g.add_argument("--lora_bias", type=str, default='none', help="lora bias ['none', 'lora_only', 'all']")

#data path parameters
g.add_argument("--train_path", type=str, default='./data/train.json', help="train dataset path (json)")
g.add_argument("--dev_path", type=str, default='./data/dev.json', help="dev dataset path (json)")
g.add_argument("--save_dir", type=str, default="test_git", help="model save path")




def main(args):
    #tokenizers parallelism
    if args.tokenizer_parallel:
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    else:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    #setting tokenizer
    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.padding_side = 'right'
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{% if message['content']%}{{'### System:\n' + message['content']+'\n\n'}}{% endif %}{% elif message['role'] == 'user' %}{{'### User:\n' + message['content']+'\n\n'}}{% elif message['role'] == 'assistant' %}{{'### Assistant:\n'  + message['content']}}{% endif %}{% if loop.last and add_generation_prompt %}{{ '### Assistant:\n' }}{% endif %}{% endfor %}"
    LABEL_IDS = [tokenizer(i, add_special_tokens=False)['input_ids'][0] for i in ['A','B','C']]

    #model config setting
    custom_model_kwargs = {}

    #model_kwargs['pretrained_model_name_or_path'] = args.model_id
    if args.model_dtype=="float16":
        custom_model_kwargs['torch_dtype'] = torch.float16
    else:
        custom_model_kwargs['torch_dtype'] = torch.bfloat16 

    custom_model_kwargs['device_map'] = "auto"

    #quantization config
    if args.quant_allow:
        if args.quant_4bit==False and args.quant_8bit==False:
            raise ValueError("set False both 4bit and 8bit with quant_allow is True. Please set quant_allow to False or set True one of them")
        
        if args.quant_4bit==True and args.quant_8bit==True:
            raise ValueError("Please set True one of them(quant_4bit, quant_8bit)")

        if args.quant_4bit_compute_dtype=="float16":
            quant_4bit_dtype = torch.float16
        else:
            quant_4bit_dtype = torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.quant_4bit,
            load_in_8bit=args.quant_8bit,
            bnb_4bit_use_double_quant=args.quant_4bit_double,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=quant_4bit_dtype
        )
        custom_model_kwargs['quantization_config'] = bnb_config

    #define model
    model = Llama3ForSFT.from_pretrained(
        args.model_id,
        **custom_model_kwargs
    )
    model.LABEL_IDS = LABEL_IDS
    model.is_train = True
    
    #lora fonfig
    lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=args.lora_dropout,
    bias=args.lora_bias,
    task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())
    
    #load dataset
    if args.fold_mode==False:
        train_dataset = TrainDataset(args.train_path, args.fold_mode, args.fold_num, args.fold_idx, tokenizer)
        valid_dataset = DevDataset(args.dev_path, args.fold_mode, args.fold_num, args.fold_idx, tokenizer)
    else:
        directory_path = get_directory_path(args.train_path)
        fold_directory_path = os.path.join(directory_path, 'merge.json')
        if not os.path.exists(fold_directory_path):
            merge_data(args.train_path, args.dev_path)
        train_dataset = TrainDataset(fold_directory_path, args.fold_mode, args.fold_num, args.fold_idx, tokenizer)
        valid_dataset = DevDataset(fold_directory_path, args.fold_mode, args.fold_num, args.fold_idx, tokenizer)

    from datasets import Dataset
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
        })
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    if args.warmup_steps < 0:
        args.warmup_steps = len(train_dataset)//args.gradient_accumulation_steps
        eval_mode = "steps"
    else:
        eval_mode = "epoch"

    #training config
    training_args = SFTConfig(
        output_dir=os.path.join('./output',args.save_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy=eval_mode,
        eval_steps=args.warmup_steps if eval_mode=='steps' else None,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epoch,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        log_level="info",
        logging_steps=1,
        save_strategy="epoch",
        bf16=False if args.model_dtype=="float16" else True,
        fp16= True if args.model_dtype=="float16" else False,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=1750,
        packing=True,
        seed=args.seed,
        report_to="none",
       # optim="paged_adamw_8bit"
    )

    #define custom metrics
    metrics_calculator = CustomMetrics(LABEL_IDS)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=metrics_calculator.compute_metrics,
    )

    #train
    trainer.train()

    
    
if __name__ == "__main__":
    exit(main(parser.parse_args()))

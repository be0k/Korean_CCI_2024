import sys
sys.path.append('C:\\Users\\Gachon\\Desktop\\jy_main\\kor_git')

from src.data import TestDataset
import argparse
import json
import tqdm
import torch
import numpy as np
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import os
# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output",type=str, default=f'quan8_auto.json',  help="output filename")
g.add_argument("--model_id", type=str, default='kihoonlee/STOCK_SOLAR-10.7B', help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, default='cuda', help="device to load the model")
g.add_argument("--device_number", type=int, default=1,help="device number(if < 0, not select)")
g.add_argument("--peft_model_dir", type=str, default="./test_git/checkpoint-614",help="peft_model_dir")
g.add_argument("--test_dir", type=str, default="data/test.json", help="test dir path")

# fmt: on


def main(args):
    if args.device_number >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"]= str(args.device_number)
    
    #model
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    use_cache=False,
    device_map='auto',
    quantization_config=bnb_config,
    )
    
    model = PeftModel.from_pretrained(model, args.peft_model_dir)
    model.eval()

    #tokenizer
    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.padding_side = 'right'
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{% if message['content']%}{{'### System:\n' + message['content']+'\n\n'}}{% endif %}{% elif message['role'] == 'user' %}{{'### User:\n' + message['content']+'\n\n'}}{% elif message['role'] == 'assistant' %}{{'### Assistant:\n'  + message['content']}}{% endif %}{% if loop.last and add_generation_prompt %}{{ '### Assistant:\n' }}{% endif %}{% endfor %}"

    dataset = TestDataset(args.test_dir, tokenizer)

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }

    with open(args.test_dir, "r", encoding="utf-8") as f:
        result = json.load(f)

    answer = []
    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(dataset))):
            inp, labels = dataset[idx]
            outputs = model(
                inp.to('cuda').unsqueeze(0),
                labels=labels.to('cuda')
            )
            logits = outputs.logits[:,-1].flatten()
            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer.vocab['A']],
                            logits[tokenizer.vocab['B']],
                            logits[tokenizer.vocab['C']],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            answer.append(np.argmax(probs))
            
    answer = np.array(answer)
    answer = answer.reshape(6, len(answer)//6)
    for i, custom_dict in enumerate(list(itertools.permutations([0,1,2]))):
        custom_dict = {value: index for index, value in enumerate(custom_dict)}
        print(custom_dict)
        answer[i] =  np.array([custom_dict[value] for value in answer[i].tolist()])
        
    from scipy import stats
    mode_values = stats.mode(answer, axis=0).mode
    for idx, label in enumerate(mode_values):
        result[idx]["output"] = answer_dict[label]
        print(answer_dict[label])
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    exit(main(parser.parse_args()))
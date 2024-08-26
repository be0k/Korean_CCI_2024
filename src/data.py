import random
from torch.utils.data import Dataset
import json
import torch
import itertools



class TrainDataset(Dataset):
    def __init__(self, fname, fold_mode, fold_num, fold_idx, tokenizer, mask_prob=0.):
        self.IGNORE_INDEX = -100
        self.data = []
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob

        self.PROMPT = '''You are an AI assistant that helps users analyze conversations and solve related problems. Please read the conversation carefully and select the most appropriate answer to the question based on the given options.'''
        self.answer_dict = {
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        with open(fname, "r", encoding='utf-8') as f:
            self.data = json.load(f)

        if fold_mode:
            fold_size = len(self.data) // fold_num
            start = fold_size*fold_idx
            end = start + fold_size
            self.data = self.data[:start] + self.data[end:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        inp = example["input"]
        chat = ["[Conversation]"]

        for cvt in inp['conversation']:
            speaker = cvt['speaker']
            utterance = cvt['utterance']
            str(utterance).replace('name', '화자')
            if random.random() < self.mask_prob:
                utterance = "[MASK]"
            chat.append(f"화자{speaker}: {utterance}")
        chat = "\n".join(chat)

        question = f"[Question]\n위 대화의 {inp['category']}"
        if (ord(inp['category'][-1]) - ord("가")) % 28 > 0:
            question += "으로"
        else:
            question = "로"
        question += " 올바른 지문은?"
                
        chat += "\n\n" + question + "\n\n[Option]\n"

        inferences = [
            inp['inference_1'],
            inp['inference_2'],
            inp['inference_3']
        ]
        label = self.answer_dict[example["output"]]

        order = list(range(len(inferences)))
        random.shuffle(order)
        
        shuffled_inferences = [inferences[i] for i in order]
        new_label = order.index(label)
        
        chat += f"A. {shuffled_inferences[0]}\n"
        chat += f"B. {shuffled_inferences[1]}\n"
        chat += f"C. {shuffled_inferences[2]}"

        message = [
            {"role": "system", "content": self.PROMPT},
            {"role": "user", "content": chat},
        ]

        source = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        target = f"{['A', 'B', 'C'][new_label]}. {shuffled_inferences[new_label]}{self.tokenizer.eos_token}"

        target = self.tokenizer(target,
                                return_attention_mask=False,
                                add_special_tokens=False,
                                return_tensors="pt")
        target["input_ids"] = target["input_ids"].type(torch.int64)

        input_ids = torch.concat((source[0], target["input_ids"][0]))
        labels = torch.concat((torch.LongTensor([self.IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
        
        return {
        'input_ids': input_ids,
        "labels": labels,
        }
    




class DevDataset(Dataset):
    def __init__(self, fname, fold_mode, fold_num, fold_idx, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        PROMPT = '''You are an AI assistant that helps users analyze conversations and solve related problems. Please read the conversation carefully and select the most appropriate answer to the question based on the given options.'''

        with open(fname, "r", encoding='utf-8') as f:
            data = json.load(f)

        if fold_mode:
            fold_size = len(data) // fold_num
            start = fold_size*fold_idx
            end = start + fold_size
            data = data[start:end]


        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                str(utterance).replace('name', '화자')
                chat.append(f"화자{speaker}: {utterance}")
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화의 {inp['category']}"
            if (ord(inp['category'][-1]) - ord("가")) % 28 > 0:
                question += "으로"
            else:
                question = "로"
            question += " 올바른 지문은?"
                
            chat = chat + "\n\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = ""
            if example["output"] == "inference_1":
                target = f"A. {example['input']['inference_1']}{tokenizer.eos_token}"
            elif example["output"] == "inference_2":
                target = f"B. {example['input']['inference_2']}{tokenizer.eos_token}"
            elif example["output"] == "inference_3":
                target = f"C. {example['input']['inference_3']}{tokenizer.eos_token}"
                
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.label[idx]


class TestDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.trg = []
        self.label = []

        PROMPT = '''You are an AI assistant that helps users analyze conversations and solve related problems. Please read the conversation carefully and select the most appropriate answer to the question based on the given options.'''
        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        with open(fname, "r", encoding='utf-8') as f:
            data = json.load(f)

        def make_chat(inp, idx):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                str(utterance).replace('name', '화자')
                chat.append(f"화자{speaker}: {utterance}")
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화의 {inp['category']}"
            if (ord(inp['category'][-1]) - ord("가")) % 28 > 0:
                question += "으로"
            else:
                question = "로"
            question += " 올바른 지문은?"
                
            chat = chat + "\n\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp[f'inference_{idx[0]}']}\n"
            chat += f"B. {inp[f'inference_{idx[1]}']}\n"
            chat += f"C. {inp[f'inference_{idx[2]}']}"

            return chat
        permutations = list(itertools.permutations([1,2,3]))
        for idx in permutations:
            for example in data:
                chat = make_chat(example["input"], idx)
                message = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": chat},
                ]
         
                source = tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
    
                target = ""

                target = tokenizer(target,
                          return_attention_mask=False,
                          add_special_tokens=False,
                          return_tensors="pt")
                target["input_ids"] = target["input_ids"].type(torch.int64)
    
                input_ids = torch.concat((source[0], target["input_ids"][0]))
                labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
                self.inp.append(input_ids)
                self.label.append(labels)
                self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.label[idx]



class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
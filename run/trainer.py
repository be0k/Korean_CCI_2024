from transformers import (
    LlamaPreTrainedModel,
    LlamaModel,
)
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F



class Llama3ForSFT(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.LABEL_IDS = None
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids= None,
        attention_mask= None,
        position_ids = None,
        past_key_values= None,
        inputs_embeds= None,
        labels= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states = None,
        return_dict= None,
        cache_position = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if self.is_train:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            label_tokens_ids = torch.tensor(self.LABEL_IDS,device=shift_labels.device)
            index_mapping = {value.item(): idx for idx, value in enumerate(label_tokens_ids)}
            true_labels = shift_labels[torch.isin(shift_labels, label_tokens_ids)]
            true_labels = torch.tensor([index_mapping[label.item()] for label in true_labels], device=true_labels.device)
            true_logits = shift_logits[torch.isin(shift_labels, label_tokens_ids)][:,label_tokens_ids]
            loss = loss_fct(true_logits, true_labels)

            return CausalLMOutputWithPast(
                loss=loss,
                logits=true_logits,
            )
    
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
            )
# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the value functions for the fine-grained RL
#  Referenced: https://github.com/liujch1998/rainier/
#  All Rights Reserved.
#  *********************************************************


from typing import Union, List, Dict
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from typing import Optional, List, Iterable, Dict, Any, Tuple
from .utils import logits_to_entropy, mask_pad


class MLP(torch.nn.Module):
    
    def __init__(self, d_model, d_out) -> None:
        super().__init__()
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, d_out),
        )
    
    def forward(self, x):
        return self.model(x)


class T5Value:

    def __init__(self,
                 model_ckpt: str,
                 model,
                 tokenizer,
                 accelerator,
                 freeze_model: bool = False,
                 vector_RL: int = 1
                ):
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.vector_RL = vector_RL
        assert self.vector_RL>=1, f"Vectorized RL only supported for >=1 reward functions, but rather got {self.vector_RL}"
        
        if model is not None:
            self.model = model
            return

        self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
            
        self.linear = MLP(self.model.config.d_model, self.vector_RL)
        
        # freeze all parameters except the last layer
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False


    def forward_pass(self,
                     prompts_input_ids: torch.Tensor, # (B, input_len)
                     prompts_attention_mask: torch.Tensor, # (B, input_len)
                     generated_input_ids: torch.Tensor, # (B, output_len)
                     generated_attention_mask: torch.Tensor, # (B, output_len)
                     reward_id = None,
                    ):

        outputs = self.model(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            labels=mask_pad(generated_input_ids, generated_attention_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        if self.vector_RL==1:
            logits = self.linear(outputs.decoder_hidden_states[-1]).squeeze(-1) # (B, output_len)
            results = {
                    'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, output_len)
            }

        elif self.vector_RL>1:
            logits = self.linear(outputs.decoder_hidden_states[-1]) # (B*vector_RL, output_len, vector_RL)
            bs,out_len = logits.shape[0], logits.shape[1]
            logits = logits.permute(2,0,1) #(vector_RL, B*vector_RL, output_len)

            if reward_id!=None: #passed batch B' contains samples only corresponding to particular reward type
                collected_logits = logits[reward_id] ##(B', output_len)
            else:
                #for each reward head, get logits only corresponding to its batch
                collected_logits = []
                for i in range(self.vector_RL):
                    start = (bs//self.vector_RL)*i
                    end = (bs//self.vector_RL)*(i+1)
                    collected_logits.append(logits[i,start:end]) #(B, output_len)
    
                # essentially, inflated the batch size to have as many samples as number of rewards
                collected_logits = torch.concat(collected_logits,dim=0) #(B*vector_RL, output_len, vocab)
            
            # logits = logits.reshape(-1,out_len) #(B*vector_RL, output_len)
            results = {
                'generated_value': mask_pad(collected_logits, generated_attention_mask, 0), # (B*vector_RL, output_len)
            }
        

        return results

# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the policy functions for the fine-grained RL
#  Referenced: https://github.com/liujch1998/rainier/
#  All Rights Reserved.
#  *********************************************************

from typing import Union, List, Dict
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from typing import Optional, List, Iterable, Dict, Any, Tuple
from .utils import logits_to_entropy, mask_pad


class T5Policy:

    def __init__(self,
                 model_ckpt: str,
                 tokenizer,
                 policy_value_sharing: bool,
                 accelerator,
                 vector_RL: int = 1
                ):
        self.tokenizer = tokenizer
        self.policy_value_sharing = policy_value_sharing
        self.accelerator = accelerator
        self.vector_RL = vector_RL
        assert self.vector_RL>=1, f"Vectorized RL only supported for >=1 reward functions, but rather got {self.vector_RL}"

        self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
        if self.vector_RL>1:
            lm_head_wts = self.model.lm_head.weight.detach() #(Vocab,Emb)
            self.model.lm_head = torch.nn.Linear(in_features=self.model.config.d_model,out_features=self.model.config.vocab_size*self.vector_RL, bias=self.model.lm_head.bias!=None,device=self.model.device)
            self.model.lm_head.load_state_dict({'weight':torch.concat([lm_head_wts]*self.vector_RL)}) #(Vocab*vector_RL,Emb) #for vector_RL no. of reward models
            
            #lm_head_wts.repeat(self.vector_RL,1)
        self.init_lm_head = self.model.lm_head.weight.clone().detach()
        # regression head for policy-value sharing
        self.linear = torch.nn.Linear(self.model.config.d_model, self.vector_RL)   
        
        self.model.eval()
        
    def sample(self,
               prompts_input_ids: torch.Tensor, # (B, input_len)
               prompts_attention_mask: torch.Tensor, # (B, input_len)
               do_sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
               num_beams: int = 1,
               num_return_sequences: int = 1,
              ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        
        prompts_text = self.tokenizer.batch_decode(prompts_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        # print(torch.all(unwrapped_model.lm_head.weight.cpu()==self.init_lm_head))
        # if unwrapped_model.lm_head.weight.grad!=None:
        #     print(torch.all(unwrapped_model.lm_head.weight.grad==0),unwrapped_model.lm_head.weight.grad.min(),unwrapped_model.lm_head.weight.grad.max())
        
        if self.vector_RL==1:
            if do_sample:
                generated_input_ids = unwrapped_model.generate(
                    input_ids=prompts_input_ids,
                    attention_mask=prompts_attention_mask,
                    max_length=self.tokenizer.max_generated_len + 1,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=num_return_sequences,
                    synced_gpus=True,
                ) # begins with 0 ([BOS]); ends with 1 ([EOS])
                
            else:
                generated_input_ids = unwrapped_model.generate(
                    input_ids=prompts_input_ids,
                    attention_mask=prompts_attention_mask,
                    max_length=self.tokenizer.max_generated_len + 1,
                    num_beams=num_beams,
                    do_sample=False,
                    num_return_sequences=num_return_sequences,
                    synced_gpus=True,
                )

            generated_input_ids = generated_input_ids[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])
    
            generated_input_ids = F.pad(generated_input_ids, (0, self.tokenizer.max_generated_len - generated_input_ids.size(1)), value=self.tokenizer.pad_token_id) # (B, output_len)
            generated_attention_mask = (generated_input_ids != self.tokenizer.pad_token_id).long()
            generated_text = self.tokenizer.batch_decode(generated_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
            # repeat input sequences for num_return_sequences times
            prompts_text = [elem for elem in prompts_text for _ in range(num_return_sequences)]
            
            return {
                'prompts_text': prompts_text,
                'prompts_input_ids': prompts_input_ids.repeat_interleave(num_return_sequences, dim=0), # (B, input_len)
                'prompts_attention_mask': prompts_attention_mask.repeat_interleave(num_return_sequences, dim=0), # (B, input_len)
                'generated_text': generated_text,
                'generated_input_ids': generated_input_ids, # (B, output_len)
                'generated_attention_mask': generated_attention_mask, # (B, output_len)
            }

        elif self.vector_RL>1:
            # import pdb; pdb.set_trace()
            unwrapped_model.eval()
            # unwrapped_model.encoder_cached = None #clear encoder cache if any
            with torch.no_grad():
                encoder_cache = unwrapped_model(input_ids=prompts_input_ids,
                                                attention_mask=prompts_attention_mask,
                                                return_encoder_cached=True,
                                                return_dict=True,
                                                output_attentions=False,
                                                output_hidden_states=True,
                                                )
                generation_kwargs = {'encoder_outputs':encoder_cache}
                # unwrapped_model.encoder_cached = encoder_cache #load current batch encoder outputs into cache
            
            # curr_lm_head_wts = unwrapped_model.lm_head.weight.clone().detach() #(vector_RL*vocab, Emb)
            # curr_lm_head_wts = curr_lm_head_wts.reshape(self.vector_RL, unwrapped_model.config.vocab_size, unwrapped_model.config.d_model)
            # print(torch.all(curr_lm_head_wts[0]==curr_lm_head_wts[1]))
            # unwrapped_model.lm_head = torch.nn.Linear(in_features=curr_lm_head_wts.shape[2], out_features=curr_lm_head_wts.shape[1], bias=False)
            # for rew_id in range(self.vector_RL):
            #     unwrapped_model.lm_head.weight = torch.nn.Parameter(curr_lm_head_wts[rew_id]) #(vector_RL, vocab, Emb)
            #     if do_sample:
            
            curr_device = unwrapped_model.device
            curr_lm_head = unwrapped_model.lm_head #(vector_RL*vocab, Emb)
            curr_lm_head_wts = unwrapped_model.lm_head.weight
            curr_lm_head_wts_reshaped = curr_lm_head_wts.reshape(self.vector_RL, unwrapped_model.config.vocab_size, unwrapped_model.config.d_model).detach()
            unwrapped_model.lm_head = torch.nn.Linear(in_features=unwrapped_model.config.d_model, out_features=unwrapped_model.config.vocab_size, bias=False,device=curr_device)

            
            for rew_id in range(self.vector_RL):
                unwrapped_model.lm_head.load_state_dict({'weight': curr_lm_head_wts_reshaped[rew_id]}) #(vector_RL, vocab, Emb)
                unwrapped_model.eval()
                with torch.no_grad():
                    if do_sample:
                        generated_input_ids_temp = unwrapped_model.generate(
                            input_ids=prompts_input_ids,
                            attention_mask=prompts_attention_mask,
                            max_length=self.tokenizer.max_generated_len + 1,
                            do_sample=True,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            num_return_sequences=num_return_sequences,
                            synced_gpus=True,
                            **generation_kwargs,
                        ) # begins with 0 ([BOS]); ends with 1 ([EOS])
                        
                    else:
                        generated_input_ids_temp = unwrapped_model.generate(
                            input_ids=prompts_input_ids,
                            attention_mask=prompts_attention_mask,
                            max_length=self.tokenizer.max_generated_len + 1,
                            num_beams=num_beams,
                            do_sample=False,
                            num_return_sequences=num_return_sequences,
                            synced_gpus=True,
                            **generation_kwargs,
                        )
                if rew_id==0:
                    generated_input_ids_temp = generated_input_ids_temp[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])
            
                    generated_input_ids = F.pad(generated_input_ids_temp, (0, self.tokenizer.max_generated_len - generated_input_ids_temp.size(1)), value=self.tokenizer.pad_token_id) # (B, output_len)
                
                else:
                    generated_input_ids_temp = generated_input_ids_temp[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])
            
                    generated_input_ids = torch.concat([generated_input_ids, 
                                                       F.pad(generated_input_ids_temp, (0, self.tokenizer.max_generated_len - generated_input_ids_temp.size(1)), value=self.tokenizer.pad_token_id)] # (B, output_len)
                                                      )
                generated_attention_mask = (generated_input_ids != self.tokenizer.pad_token_id).long()
                generated_text = self.tokenizer.batch_decode(generated_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
                    # repeat input sequences for num_return_sequences times
                prompts_text = [elem for elem in prompts_text for _ in range(num_return_sequences)]*self.vector_RL

            unwrapped_model.lm_head = curr_lm_head #torch.nn.Linear(in_features=unwrapped_model.config.d_model,out_features=unwrapped_model.config.vocab_size*self.vector_RL, bias=False, device=curr_device)
            # unwrapped_model.lm_head.load_state_dict({'weight': curr_lm_head_wts})#(Vocab*vector_RL,Emb) #for vector_RL no. of reward models
                        
            return {
                'prompts_text': prompts_text,
                'prompts_input_ids': prompts_input_ids.repeat_interleave(num_return_sequences, dim=0).repeat(self.vector_RL,1), # (B*vector_RL, input_len)
                'prompts_attention_mask': prompts_attention_mask.repeat_interleave(num_return_sequences, dim=0).repeat(self.vector_RL,1), # (B*vector_RL, input_len)
                'generated_text': generated_text,
                'generated_input_ids': generated_input_ids, # (B*vector_RL, output_len)
                'generated_attention_mask': generated_attention_mask, # (B*vector_RL, output_len)
            }
    

    def forward_pass(self,
                     prompts_input_ids: torch.Tensor, # (B, input_len) or (B*vector_RL, input_len)
                     prompts_attention_mask: torch.Tensor, # (B, input_len) or (B*vector_RL, input_len)
                     generated_input_ids: torch.Tensor, # (B, output_len) or (B*vector_RL, output_len)
                     generated_attention_mask: torch.Tensor, # (B, output_len) or (B*vector_RL, output_len)
                     reward_id = None,
                    ):
        # print(self.accelerator.unwrap_model(self.model).lm_head.weight.requires_grad)
        outputs = self.model(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            labels=mask_pad(generated_input_ids, generated_attention_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        if self.vector_RL==1:
            generated_logits = outputs.logits # (B, output_len, V)
            logprobs = F.log_softmax(generated_logits, dim=-1)
            generated_logprobs = torch.gather(logprobs, 2, generated_input_ids[:, :, None]).squeeze(2) # (B, output_len)
            generated_entropy = logits_to_entropy(generated_logits) # (B, output_len)
    
            results = {
                'generated_logits': generated_logits, # (B, output_len, V)
                'generated_logprobs': mask_pad(generated_logprobs, generated_attention_mask), # (B, output_len)
                'generated_entropy': mask_pad(generated_entropy, generated_attention_mask), # (B, output_len)
            }
    
            if self.policy_value_sharing:
                logits = self.linear(outputs.decoder_hidden_states[-1]).squeeze(-1) # (B, output_len)
                results.update({
                    'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, output_len)
                })

        elif self.vector_RL>1:
            # policy sample function already generated B*vector_RL number of samples, which we run through a vocab*vector_RL dimensional LM Head
            # the implementation is still bit redundant, we can send relevant samples to relevant heads
            generated_logits = outputs.logits # (B*vector_RL, output_len, Vocab*vector_RL)
            bs, out_len, vocab_size = generated_logits.shape
            generated_logits = generated_logits.reshape(bs,out_len,self.vector_RL,-1) #(B*vector_RL, output_len, vector_RL, Vocab)
            generated_logits = generated_logits.permute(2,0,1,3) #(vector_RL, B*vector_RL, output_len,  Vocab)

            if reward_id!=None: #passed batch B' correspond to only particular reward type
                collected_logits = generated_logits[reward_id] #(B', output_len, vocab)
                
            else: #assuming no rew_id information is given, collect assuming that batch has all reward samples
                #for each reward head, get logits only corresponding to its batch
                collected_logits = []
                for i in range(self.vector_RL):
                    start = (bs//self.vector_RL)*i
                    end = (bs//self.vector_RL)*(i+1)
                    collected_logits.append(generated_logits[i,start:end]) #(B, output_len, vocab)
    
                # essentially, inflated the batch size to have as many samples as number of rewards
                collected_logits = torch.concat(collected_logits,dim=0) #(B*vector_RL, output_len, vocab)

            logprobs = F.log_softmax(collected_logits, dim=-1) #softmax over Vocab
            generated_logprobs = torch.gather(logprobs, 2, generated_input_ids[:, :, None]).squeeze(2) # (B*vector_RL, output_len)
            generated_entropy = logits_to_entropy(generated_logits) # (B*vector_RL, output_len)
    
            results = {
                'generated_logits': generated_logits, # (B*vector_RL, output_len, V)
                'generated_logprobs': mask_pad(generated_logprobs, generated_attention_mask), # (B*vector_RL, output_len)
                'generated_entropy': mask_pad(generated_entropy, generated_attention_mask), # (B*vector_RL, output_len)
            }
    
            if self.policy_value_sharing:
                logits = self.linear(outputs.decoder_hidden_states[-1]) # (B*vector_RL, output_len, vector_RL)
                out_len = logits.shape[1]
                logits = logits.permute(2,0,1) #(vector_RL, B*vector_RL, output_len)

                if reward_id!=None: #passed batch B' contains samples only corresponding to particular reward type
                    collected_logits = logits[reward_id] #(B', output_len)
                else:
                    #for each reward head, get logits only corresponding to its batch
                    collected_logits = []
                    for i in range(self.vector_RL):
                        start = (bs//self.vector_RL)*i
                        end = (bs//self.vector_RL)*(i+1)
                        collected_logits.append(logits[i,start:end]) #(B, output_len)
        
                    # essentially, inflated the batch size to have as many samples as number of rewards
                    collected_logits = torch.concat(collected_logits,dim=0) #(B*vector_RL, output_len, vocab)
                
                results.update({
                    'generated_value': mask_pad(collected_logits, generated_attention_mask, 0), # (B*vector_RL, output_len)
                })
            # generated_logprobs = torch.gather(logprobs, 2, generated_input_ids.repeat(self.vector_RL,1,1)[:, :, :, None]).squeeze(-1) # (vector_RL, B, output_len)
            # generated_entropy = logits_to_entropy(generated_logits) # (vector_RL, B, output_len)
    
            # results = {
            #     'generated_logits': generated_logits, #(vector_RL, B, output_len,  Vocab)
            #     'generated_logprobs': mask_pad(generated_logprobs, generated_attention_mask.repeat(self.vector_RL,1,1)), # (vector_RL, B, output_len)
            #     'generated_entropy': mask_pad(generated_entropy, generated_attention_mask.repeat(self.vector_RL,1,1)), # (vector_RL, B, output_len)
            # }
    
            # if self.policy_value_sharing:
            #     logits = self.linear(outputs.decoder_hidden_states[-1]) # (B, output_len, vector_RL)
            #     logits = logits.permute(2,0,1) #(vector_RL, B, output_len)
            #     results.update({
            #         'generated_value': mask_pad(logits, generated_attention_mask.repeat(self.vector_RL,1,1), 0), # (vector_RL, B, output_len)
            #     })

        return results

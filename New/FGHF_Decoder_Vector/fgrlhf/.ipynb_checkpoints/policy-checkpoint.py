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
import copy


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
        
        self.model.decoder.vector_heads = torch.nn.ModuleList([copy.deepcopy(self.model.decoder.block[-1]) for _ in range(self.vector_RL)])
        del self.model.decoder.block[-1]

        # import pdb; pdb.set_trace()
        if self.vector_RL>1:
            self.dec_v0_wt = self.model.decoder.vector_heads[0].layer[0].SelfAttention.q.weight.detach().clone().to(self.accelerator.device)
            self.dec_v1_wt = self.model.decoder.vector_heads[1].layer[0].SelfAttention.q.weight.detach().clone().to(self.accelerator.device)
            self.dec_v2_wt = self.model.decoder.vector_heads[2].layer[0].SelfAttention.q.weight.detach().clone().to(self.accelerator.device)
            
        # regression head for policy-value sharing
        self.linear = torch.nn.Linear(self.model.config.d_model, 1)   

        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        if self.vector_RL==1:
            generation_kwargs = {'vector_head':0}
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
                    **generation_kwargs,
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
                    **generation_kwargs,
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
                generation_kwargs = {'encoder_outputs':encoder_cache,'vector_head':0}
                # unwrapped_model.encoder_cached = encoder_cache #load current batch encoder outputs into cache
            
            # curr_lm_head_wts = unwrapped_model.lm_head.weight.clone().detach() #(vector_RL*vocab, Emb)
            # curr_lm_head_wts = curr_lm_head_wts.reshape(self.vector_RL, unwrapped_model.config.vocab_size, unwrapped_model.config.d_model)
            # print(torch.all(curr_lm_head_wts[0]==curr_lm_head_wts[1]))
            # unwrapped_model.lm_head = torch.nn.Linear(in_features=curr_lm_head_wts.shape[2], out_features=curr_lm_head_wts.shape[1], bias=False)
            # for rew_id in range(self.vector_RL):
            #     unwrapped_model.lm_head.weight = torch.nn.Parameter(curr_lm_head_wts[rew_id]) #(vector_RL, vocab, Emb)
            #     if do_sample:

            for rew_id in range(self.vector_RL):
                with torch.no_grad():
                    generation_kwargs['vector_head'] = rew_id
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

            # unwrapped_model.lm_head = curr_lm_head #torch.nn.Linear(in_features=unwrapped_model.config.d_model,out_features=unwrapped_model.config.vocab_size*self.vector_RL, bias=False, device=curr_device)
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

        encoder_cache = self.model(input_ids=prompts_input_ids,
                                      attention_mask=prompts_attention_mask,
                                      labels=mask_pad(generated_input_ids, generated_attention_mask, -100),
                                      return_dict=True,
                                      output_attentions=False,
                                      output_hidden_states=True,
                                      return_encoder_cached=True
                                     )
        
        cached_decoder_before_vector = self.model(input_ids=prompts_input_ids,
                                                  attention_mask=prompts_attention_mask,
                                                  labels=mask_pad(generated_input_ids, generated_attention_mask, -100),
                                                  return_dict=True,
                                                  encoder_outputs=encoder_cache,
                                                  output_attentions=False,
                                                  output_hidden_states=True,
                                                  return_decoder_before_vector=True
                                                 )
        # import pdb; pdb.set_trace()
        if reward_id==None:
            orig_BS = int(generated_input_ids.shape[0]//self.vector_RL)
            for rew_id in range(self.vector_RL):
                outputs = self.model(
                    input_ids=prompts_input_ids,
                    attention_mask=prompts_attention_mask,
                    labels=mask_pad(generated_input_ids, generated_attention_mask, -100),
                    return_dict=True,
                    encoder_outputs=encoder_cache,
                    output_attentions=False,
                    output_hidden_states=True,
                    vector_head=rew_id,
                    cached_decoder_before_vector=cached_decoder_before_vector,
                )
        
                if rew_id==0:
                    output_logits = outputs.decoder_hidden_states[-1][rew_id*orig_BS:(rew_id+1)*orig_BS]
                    generated_logits = outputs.logits[rew_id*orig_BS:(rew_id+1)*orig_BS] # (B, output_len, V)
                    logprobs = F.log_softmax(generated_logits, dim=-1)
                    generated_logprobs = torch.gather(logprobs, 2, generated_input_ids[rew_id*orig_BS:(rew_id+1)*orig_BS, :, None]).squeeze(2) # (B, output_len)
                    generated_entropy = logits_to_entropy(generated_logits) # (B, output_len)
                else:
                    output_logits_temp = outputs.decoder_hidden_states[-1][rew_id*orig_BS:(rew_id+1)*orig_BS]
                    generated_logits_temp = outputs.logits[rew_id*orig_BS:(rew_id+1)*orig_BS] # (B, output_len, V)
                    logprobs = F.log_softmax(generated_logits_temp, dim=-1)
                    generated_logprobs_temp = torch.gather(logprobs, 2, generated_input_ids[rew_id*orig_BS:(rew_id+1)*orig_BS, :, None]).squeeze(2) # (B, output_len)
                    generated_entropy_temp = logits_to_entropy(generated_logits_temp) # (B, output_len)

                    output_logits = torch.concat([output_logits,output_logits_temp],dim=0)
                    generated_logits = torch.concat([generated_logits,generated_logits_temp],dim=0)
                    generated_logprobs = torch.concat([generated_logprobs,generated_logprobs_temp],dim=0)
                    generated_entropy = torch.concat([generated_entropy,generated_entropy_temp],dim=0)
            
        
        else:
            outputs = self.model(
                    input_ids=prompts_input_ids,
                    attention_mask=prompts_attention_mask,
                    labels=mask_pad(generated_input_ids, generated_attention_mask, -100),
                    return_dict=True,
                    encoder_outputs=encoder_cache,
                    output_attentions=False,
                    output_hidden_states=True,
                    vector_head=reward_id,
                    cached_decoder_before_vector=cached_decoder_before_vector,
                )
        
            output_logits = outputs.decoder_hidden_states[-1]
            generated_logits = outputs.logits # (B, output_len, V)
            logprobs = F.log_softmax(generated_logits, dim=-1)
            generated_logprobs = torch.gather(logprobs, 2, generated_input_ids[:, :, None]).squeeze(2) # (B, output_len)
            generated_entropy = logits_to_entropy(generated_logits) # (B, output_len)
            
        results = {
            'generated_logits': generated_logits, # (B*vector_RL, output_len, V)
            'generated_logprobs': mask_pad(generated_logprobs, generated_attention_mask), # (B*vector_RL, output_len)
            'generated_entropy': mask_pad(generated_entropy, generated_attention_mask), # (B*vector_RL, output_len)
        }
        
        if self.policy_value_sharing:
            logits = self.linear(output_logits).squeeze(-1) # (B, output_len)
            results.update({
                'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, output_len)
            })

    
        return results

import argparse
from collections import defaultdict
from itertools import chain
import json
import logging
import numpy as np
import os
import random
import shutil
from tqdm import tqdm
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
import accelerate
import wandb
import yaml
import nltk

from accelerate import DistributedType

from fgrlhf.ppo import PPOTrainer
from fgrlhf.policy import T5Policy
from fgrlhf.value import T5Value
from fgrlhf.utils import ensure_dir, set_seed, reduce_mean, reduce_sum, ceil_div, whiten, clamp

from reward import FineGrainedReward

logging.basicConfig(level=logging.ERROR)

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DEBUG'
# prepare accelerator and logger
from accelerate import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) #for allowing unused parameters in the model
accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])#
# accelerator = accelerate.Accelerator()
device = accelerator.device
log = accelerate.logging.get_logger(__name__, log_level='INFO')
def log_info(s):
    if accelerator.is_main_process:
        log.info(s)
        
# load parameters
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="path to config file")
args = parser.parse_args()
# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)


# prepare data
class TextGenDataset(Dataset):
    def __init__(self, split, tokenizer, accelerator=None, length_limit=None):
        super().__init__()
        
        self.split = split
        self.dataset_fns = {
            "train": "tasks/qa_feedback/data/train.json",
            "dev": "tasks/qa_feedback/data/dev.json",
            "test": "tasks/qa_feedback/data/test.json"
        }
        
        self.n_card = 1
        if accelerator is not None:
            self.n_card = accelerator.num_processes
        
        
        self.tokenizer = tokenizer

        self.instances = self.load_datasets()
        
        if length_limit is not None:
            self.instances = self.instances[:length_limit]

        if split == 'train':
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self): 
        instances = []
        
        task_data = None
        with open(self.dataset_fns[self.split], 'r') as f:
            task_data = json.load(f)
            
        for task_instance in task_data:
            instances.append({
                "prompt": "Answer the question based on the context that follows:\n" + task_instance['text'],
                "metadata": {
                    "prompt": "Answer the question based on the context that follows:\n" + task_instance['text'],
                    "references": task_instance['answer'],
                    "passages": task_instance['passages'],
                    "question": task_instance['question'],}
            })
        
        log_info(f'Loaded split {self.split} with {len(instances)} total instances')
        
        instances = instances[:len(instances)//self.n_card*self.n_card]  # or Trainer will stuck
        return instances

    # Make a collate function to fix dataloader weird list batching
    def collate_fn(self, batch):
        
        # process input prompts
        prompts = [item['prompt'] for item in batch]
        prompts_tok = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors='pt', 
            padding='max_length', 
            truncation=True,
            max_length=self.tokenizer.max_input_len,
            # padding_side=self.tokenizer.padding_side, # YUSHI: change later, now Ellen pad defaultly
            )
        
        prompts_input_ids = prompts_tok.input_ids
        prompts_attention_mask = prompts_tok.attention_mask
        
        # process metadata
        metadata = [item['metadata'] for item in batch]
        

        result = {
            'prompts_input_ids': prompts_input_ids,
            'prompts_attention_mask': prompts_attention_mask,
            'metadata': metadata
        }
        return result
    

def main():

    # set seed
    set_seed(args['train']['seed'], args['train']['cuda_deterministic'])
    
    # set saving directories
    log_info(f"Write to output directory: {args['logging']['save_dir']}")
    
    if accelerator.is_main_process:
        ensure_dir(args['logging']['save_dir'])
        # save the config file
        with open(os.path.join(args['logging']['save_dir'], 'args.json'), 'w') as f:
            json.dump(args, f, indent=2)

    
    # initialize policy and value model tokenizers
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model']['policy_model']['ckpt'], 
                                                           model_max_length=args['env']['max_input_len'])
    tokenizer.padding_side = args['model']['policy_model']['input_padding_side']
    tokenizer.max_input_len = args['env']['max_input_len']
    tokenizer.max_generated_len = args['env']['max_generated_len']
    
    
    # Load data
    log_info(f'Loading data ...')
    train_dataset = TextGenDataset( 'train', tokenizer, accelerator=accelerator)
    # train ds is shuffled in its constructor
    train_dataloader = DataLoader(train_dataset, batch_size=args['train']['sampling_batch_size_per_card'], 
                                  shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)

    eval_dataset = TextGenDataset( 'dev',  tokenizer, accelerator=accelerator, length_limit=None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args['train']['sampling_batch_size_per_card'], 
                                 shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)

    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)


    # Initialize models and optimizer
    log_info(f'Initializing models ...')

    if args['vector_RL']>1:
        total_batch_size = args['train']['sampling_batch_size_per_card'] * args['env']['train_num_samples_per_input']
        train_batch_size = args['train']['training_batch_size_per_card']
        assert total_batch_size % train_batch_size==0, f'Train batch size ({train_batch_size}) should be a factor of total batch size ({total_batch_size}) for vector RL training'

    ref_policy = T5Policy(
        model_ckpt=args['model']['policy_model']['ckpt'],
        tokenizer=tokenizer,
        policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
        accelerator=accelerator,
    )
    ref_policy.model, ref_policy.linear = accelerator.prepare(ref_policy.model, ref_policy.linear)
    policy = T5Policy(
        model_ckpt=args['model']['policy_model']['ckpt'],
        tokenizer=tokenizer,
        policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
        accelerator=accelerator,
        vector_RL = args['vector_RL'],
    )
    policy.model, policy.linear = accelerator.prepare(policy.model, policy.linear)
    
    value = T5Value(
        model_ckpt=args['model']['value_model']['ckpt'],
        model=policy.model if args['model']['value_model']['policy_value_sharing'] else None,
        tokenizer=tokenizer,
        accelerator=accelerator,
        freeze_model=False if args['model']['value_model']['policy_value_sharing'] else args['model']['value_model']['freeze_value_model'],
        vector_RL = args['vector_RL'],
        )
    if not args['model']['value_model']['policy_value_sharing']:
        value.model, value.linear = accelerator.prepare(value.model, value.linear)
    
    reward = FineGrainedReward(
        tokenizer=tokenizer,
        non_factual_model_ckpt=args['reward']['relevance_model']['ckpt'],
        factual_model_ckpt=args['reward']['factuality_model']['ckpt'],
        completeness_model_ckpt=args['reward']['completeness_model']['ckpt'],
        kl_coef=args['ppo']['kl_coef'],
        verbosity_positive_reward = args['reward']['relevance_model']['positive_reward'],
        verbosity_negative_reward = args['reward']['relevance_model']['negative_reward'],
        factuality_positive_reward = args['reward']['factuality_model']['positive_reward'],
        factuality_negative_reward = args['reward']['factuality_model']['negative_reward'],
        completeness_reward_mean = args['reward']['completeness_model']['mean'],
        completeness_reward_std = args['reward']['completeness_model']['std'],
        completeness_reward_bias = args['reward']['completeness_model']['bias'],
        completeness_reward_scale = args['reward']['completeness_model']['scale'],
        sep = "</s>",
        vector_RL = args['vector_RL'],
    )
    
    # prepare reward models
    reward.verbosity_reward.nf_reward_model = accelerator.prepare(reward.verbosity_reward.nf_reward_model)
    reward.factuality_reward.f_reward_model = accelerator.prepare(reward.factuality_reward.f_reward_model)
    reward.completeness_reward.model = accelerator.prepare(reward.completeness_reward.model)
    
    # prepare optimizers and schedulers
    if args['model']['value_model']['policy_value_sharing']:
        parameters = chain(policy.model.parameters(), policy.linear.parameters())
    else:
        parameters = chain(policy.model.parameters(), policy.linear.parameters(), value.model.parameters(), value.linear.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args['train']['lr'], eps=1e-5)
    if accelerator.distributed_type == DistributedType.MEGATRON_LM:
        tot_batch_size = accelerator.state.megatron_lm_plugin.global_batch_size
    else:
        tot_batch_size = args['train']['sampling_batch_size_per_card'] * accelerator.num_processes * args['env']['train_num_samples_per_input']
    total_steps = ceil_div(args['train']['total_episodes'], tot_batch_size)
    
    scheduler = transformers.get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=100*args['train']['n_ppo_epoch_per_rollout']*accelerator.num_processes,
        num_training_steps=total_steps*args['train']['n_ppo_epoch_per_rollout']*accelerator.num_processes,
    )
    from accelerate.utils import MegatronLMDummyScheduler

    if accelerator.distributed_type == DistributedType.MEGATRON_LM:
        lr_scheduler = MegatronLMDummyScheduler(
            optimizer=optimizer,
            total_num_steps=total_steps,
            warmup_num_steps=0,
        )
    # else:
    #     scheduler = transformers.get_constant_schedule(optimizer=optimizer)
    
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)


    # Set up trainer
    trainer = PPOTrainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        ref_policy_model=ref_policy,
        policy_model=policy,
        value_model=value,
        reward_model=reward,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        log_info=log_info,
        vector_RL=args['vector_RL'],
    )
    
    steps = list(range(total_steps + 1))
    steps = tqdm(steps) if accelerator.is_main_process else steps
    for step in steps:
        trainer.train(step)
        accelerator.wait_for_everyone()
        # early stopping because KL explodes
        if trainer.should_early_stop:
            if accelerator.is_local_main_process:
                print("Early stopping triggered. Terminating training.")
            break
            
if __name__ == '__main__':
    main()

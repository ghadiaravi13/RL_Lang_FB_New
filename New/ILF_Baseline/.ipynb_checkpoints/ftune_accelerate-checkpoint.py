import numpy as np
import pandas as pd
from datasets import load_dataset
import itertools

dataset = load_dataset("McGill-NLP/feedbackQA")

rating_class = {'Excellent':3 , 'Acceptable':2 , 'Could be Improved':1, 'Bad': 0}

def process_df(df):
    df['list_feedback'] = df['feedback'].apply(lambda x: [ r + "___" + e for r,e in zip(x['rating'],x['explanation']) ])
    df['sampled_feedback'] = df['list_feedback'].apply(lambda x: x[0].split("___") if (x[0].split("___")[0]!='Excellent' and x[0].split("___")[0]!='Acceptable') else (x[1].split("___") if (x[1].split("___")[0]!='Excellent' and x[1].split("___")[0]!='Acceptable') else np.random.choice(x).split("___")) )
    df['rating_class'] = df['sampled_feedback'].apply(lambda x: rating_class[x[0]])
    df['rating'] = df['sampled_feedback'].apply(lambda x: x[0])
    df['explanation'] = df['sampled_feedback'].apply(lambda x: x[1])
    return df

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from accelerate import Accelerator


import tqdm
from torch.utils.data import Dataset

class feedback_QA_dataset(Dataset):
    
    def __init__(self,df,tokenizer,max_length=1024):
        self.df = df
        self.max_len = max_length
        self.data = []
        self.tokenizer = tokenizer
        skipped = 0
        
        for i in range(len(self.df)):
            
            d = {}
            question = self.tokenizer(f"Question: {self.df.iloc[i]['question']}", add_special_tokens=True)
            answer = self.tokenizer(f"Answer: {self.df.iloc[i]['answer']}")
            padding = [0] * (self.max_len - len(question['input_ids']+answer['input_ids']) - 1)       
            
            input = question['input_ids'] + answer['input_ids'] + [tokenizer.eos_token_id] 
            labels = [-100]*len(question['input_ids']) + answer['input_ids'] + [tokenizer.eos_token_id] + [-100]*len(padding)
            attention_mask = [1]*len(question['input_ids']) + [1]*len(answer['input_ids']) + [1] + [0]*len(padding)

            
            if len(input) > self.max_len:
                skipped += 1
                continue
            
            input += padding
            
            d['input'] = input
            d['labels'] = labels
            d['attention_mask'] = attention_mask
            d['id'] = i

            for k in d.keys():
                d[k] = torch.tensor(d[k])

            self.data.append(d)
            
        print(f'skipped: {skipped}')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]

def validate(model,valid_DL,epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for num_batch,b in enumerate(valid_DL):
            out = model(input_ids=b['input'], labels=b['labels'])
            loss = out.loss
            total_loss += loss.item()
            # if num_batch==10:
            #     break
    
    return (total_loss/len(valid_DL))


def finetune(bert_chkpt,train_df, valid_df, TRAIN_BATCH_SIZE=3, VALID_BATCH_SIZE=4, EPOCHS=20, PATIENCE=5):
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    accelerator = Accelerator(gradient_accumulation_steps=4)
    device = accelerator.device
    
    tokenizer = AutoTokenizer.from_pretrained(bert_chkpt,cache_dir='/home/jupyter/Ravi_new/HF_cache')
    model = AutoModelForCausalLM.from_pretrained(bert_chkpt,cache_dir='/home/jupyter/Ravi_new/HF_cache')
    
    train_dataset = feedback_QA_dataset(train_df,tokenizer)
    train_DL = DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True)

    valid_dataset = feedback_QA_dataset(valid_df,tokenizer)
    valid_DL = DataLoader(valid_dataset,batch_size=VALID_BATCH_SIZE,shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-6)
    
    model, train_DL, valid_DL, optimizer = accelerator.prepare(model,train_DL,valid_DL,optimizer)

    def train(model,train_DL,optimizer,epoch,PRINT_EVERY=10):
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        for num_batch,b in enumerate(train_DL):
            out = model(input_ids=b['input'], labels=b['labels'])
            loss = out.loss
            total_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()

            if (num_batch+1)%PRINT_EVERY==0:
                accelerator.print(f'Epoch: {epoch}\t\t Num_Samples: {num_batch+1:04d}\t\t Train_Loss per sample: {total_loss/(num_batch+1)} ')
            # if num_batch==10:
            #     break
    
        return (total_loss/len(train_DL))
    
    best_val_loss = np.inf#validate(model,valid_DL,0,accelerator)
    for e in range(EPOCHS):
        
        train_loss = train(model,train_DL,optimizer,e)
        accelerator.print(f"Epoch: {e}\t\t---->\t\t Train Loss per batch: {train_loss} ")
        
        val_loss = validate(model,valid_DL,e)
        accelerator.print(f"Epoch: {e}\t\t---->\t\t Valid Loss per batch: {val_loss} ")
        
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            patience = PATIENCE
            
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                tokenizer.save_pretrained('Llama_Checkpoints/')
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = accelerator.get_state_dict(model)
            unwrapped_model.save_pretrained(
                'Llama_Checkpoints/', is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict)
            #accelerator.save_model(model,save_dir='Llama_Checkpoints')#({'model_state':model.state_dict(),
            #                   'optimizer':optimizer.state_dict(),
            #                   'epoch':e},
            #                   f"Llama_Checkpoints/best_model_chkpt.pth.tar")
        else:
            patience -= 1
            accelerator.print(f"REDUCING PATIENCE...{patience}")

def main():
    
    bert_chkpt = "meta-llama/Llama-2-7b-chat-hf"
    train_df = pd.read_csv('train_data.csv')
    valid_df = pd.read_csv('valid_data.csv')
    finetune(bert_chkpt, train_df, valid_df)

if __name__ == '__main__':
    main()
    
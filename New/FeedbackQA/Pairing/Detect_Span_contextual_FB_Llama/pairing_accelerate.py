import numpy as np
import pandas as pd
from datasets import load_dataset
import itertools

from torch import nn
from nltk import tokenize as nltk_tokenizer

dataset = load_dataset("McGill-NLP/feedbackQA")

rating_scores = {'Excellent':3 , 'Acceptable':2 , 'Could be Improved':1, 'Bad': -1}

def process_df(df):
    df['question'] = df['question'].apply(lambda x: x.replace('\n',' '))
    df['answer'] = df['answer'].apply(lambda x: x.replace('\n',' '))
    df['list_feedback'] = df['feedback'].apply(lambda x: [ r + "___" + e for r,e in zip(x['rating'],x['explanation']) ])
    df['sampled_feedback'] = df['list_feedback'].apply(lambda x: np.random.choice(x).split("___") )
    df['rating_score'] = df['sampled_feedback'].apply(lambda x: rating_scores[x[0]])
    df['rating'] = df['sampled_feedback'].apply(lambda x: x[0])
    df['explanation'] = df['sampled_feedback'].apply(lambda x: x[1])
    return df

train_df = process_df(pd.DataFrame(dataset['train']))
val_df = process_df(pd.DataFrame(dataset['validation']))
test_df = process_df(pd.DataFrame(dataset['test']))

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

import tqdm

class feedback_QA_dataset(Dataset):
    
    def __init__(self,df,max_length=400,tokenizer=None):
        self.df = df
        self.max_len = max_length
        self.data = []
        skipped = 0

        assert tokenizer!=None, 'Please pass a valid tokenizer'
        
        for i in tqdm.tqdm(range(len(self.df)),desc='vectorizing..'):
            
            d = {'id':i}
            
            tok_question = tokenizer('Question: ' + self.df.iloc[i]['question'] + ' Answer: ', add_special_tokens=False)
            tok_answer = tokenizer(self.df.iloc[i]['answer'].strip().replace('  ',' '), add_special_tokens=False)
            tok_feedback = tokenizer(self.df.iloc[i]['explanation'], add_special_tokens=False)

            d['question'] = tok_question['input_ids']
            d['answer'] = tok_answer['input_ids']
            d['feedback'] = tok_feedback['input_ids']
            
            if len(tok_question['input_ids']+tok_answer['input_ids']+tok_feedback['input_ids'])+4 > self.max_len:
                skipped +=1
                continue
            
            context = [tokenizer.bos_token_id] + tok_question['input_ids'] + tok_answer['input_ids']
            context_attn = [1] + tok_question['attention_mask'] + tok_answer['attention_mask']
            context_pool_mask = [0] + [0]*len(tok_question['input_ids']) + tok_answer['attention_mask']
            
            
            d['context_w_feedback'] = context + [tokenizer.eos_token_id] + tok_feedback['input_ids'] + [tokenizer.eos_token_id]
            
            PAD_LEN = self.max_len - len(d['context_w_feedback'])

            d['Input_len'] = len(d['context_w_feedback'])
            d['PAD_LEN'] = PAD_LEN
            
            d['context_w_feedback'] += [tokenizer.pad_token_id]*PAD_LEN
            d['context_w_feedback_attn'] = context_attn + [1] + tok_feedback['attention_mask'] + [1] + [0]*PAD_LEN            
            d['context'] = d['context_w_feedback']
            d['context_attn'] = context_attn + [1] + [0]*len(tok_feedback['attention_mask']) + [0] + [0]*PAD_LEN
            
            d['feedback_pool_mask'] = [0]*len(context_pool_mask) + [0] + tok_feedback['attention_mask'] + [0] + [0]*PAD_LEN
            d['answer_pool_mask'] = context_pool_mask + [0] + [0]*len(tok_feedback['attention_mask']) + [0] + [0]*PAD_LEN
            
            answer_phrases = nltk_tokenizer.sent_tokenize(self.df.iloc[i]['answer'].strip().replace('  ',' '))
            tok_phrases = tokenizer(answer_phrases,add_special_tokens=False,return_token_type_ids=True)

            d['tok_phrases'] = tok_phrases['input_ids']
            d['answer_phrases_pool_mask'] = []
            
            for j in range(len(answer_phrases)):
                answer_phrases_attn_mask = tok_phrases['token_type_ids'].copy()
                answer_phrases_attn_mask[j] = tok_phrases['attention_mask'][j].copy()
                answer_phrases_attn_mask = list(itertools.chain.from_iterable(answer_phrases_attn_mask))
                pad_len = len(tok_answer['attention_mask']) - len(answer_phrases_attn_mask)
                answer_phrases_attn_mask += [0]*pad_len
                
                answer_phrase_pool_mask = [0] + [0]*len(tok_question['input_ids']) + answer_phrases_attn_mask + [0] + [0]*len(tok_feedback['attention_mask']) + [0] + [0]*PAD_LEN
                
                d['answer_phrases_pool_mask'].append(answer_phrase_pool_mask)
            
            if len(d['answer_phrases_pool_mask'][0])>len(d['answer_pool_mask']):
                skipped +=1
                continue
                
            else:
                self.data.append(d)
                
        print('Skipped: ',skipped)

    def add_neg_samples(self):
        for i in tqdm.tqdm(range(self.__len__()),desc='adding neg samples...'):
            self.data[i]['feedback_set'] = [self.data[i]['context_w_feedback']]
            self.data[i]['feedback_attn_set'] = [self.data[i]['context_w_feedback_attn']]
            self.data[i]['feedback_pool_mask_set'] = [self.data[i]['feedback_pool_mask']]
            L = list(range(self.__len__()))
            L.remove(i)
            neg_samples_idx = np.random.choice(L,size=2)
            for n_id in neg_samples_idx:
                self.data[i]['feedback_set'].append(self.data[n_id]['context_w_feedback'])
                self.data[i]['feedback_attn_set'].append(self.data[n_id]['context_w_feedback_attn'])
                self.data[i]['feedback_pool_mask_set'].append(self.data[n_id]['feedback_pool_mask'])
            for k in self.data[i].keys():
                if k!='tok_phrases':
                    self.data[i][k] = torch.tensor(self.data[i][k])
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]

class discriminator(nn.Module):
    def __init__(self, model_chkpt):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(bert_chkpt,cache_dir='/home/jupyter/Ravi_new/HF_cache')
        # self.device = device
        
    def mean_pooling(self,model_output,attention_mask):
        token_embeddings = model_output #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        se = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(se, p=2, dim=1)
        
    def forward(self, b):
        sent_model_out = self.model(input_ids = b['context_w_feedback'],attention_mask=b['context_attn'])[0]
        feedback_model_out = self.model(input_ids = b['feedback_set'][0],attention_mask=b['feedback_attn_set'][0])[0]
        
        sent_emb = self.mean_pooling( sent_model_out, b['answer_pool_mask'])
        feedback_emb = self.mean_pooling( feedback_model_out, b['feedback_pool_mask_set'][0])
        
        # print(pmo[0].shape,b['answer_phrases_pool_mask'].shape)
        phrase_emb = self.mean_pooling( sent_model_out.repeat(b['answer_phrases_pool_mask'][0].shape[0],1,1), b['answer_phrases_pool_mask'][0] )
        # phrase_emb = torch.stack(phrase_emb).squeeze(1)
        cos_sim = F.cosine_similarity(sent_emb,feedback_emb,dim=1)
        cos_phrase_sim = torch.matmul(phrase_emb,feedback_emb.transpose(1,0))
        
        tgt_tensor = torch.zeros(b['feedback_set'].shape[1] , device=self.device)
        tgt_tensor[0] = 1.0 #the relevant feedback is always present at index 0
        
        return_dict = {'sent_ce_loss': F.cross_entropy(cos_sim,target=tgt_tensor),
                       'avg_phrase_ce_loss': F.cross_entropy(cos_phrase_sim.mean(0),target=tgt_tensor),
                       'sent_probs': F.softmax(cos_sim,dim=-1),
                       'phrase_probs': F.softmax(cos_phrase_sim,dim=-1)}
        
        return return_dict

from accelerate import Accelerator

def train(discriminator,train_dataset,valid_dataset,epochs,batch_size,optimizer,PATIENCE=20,save_dir=None):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    accelerator = Accelerator(gradient_accumulation_steps=batch_size)
    discriminator.device = accelerator.device
    
    train_DL = DataLoader(train_dataset,batch_size=1,shuffle=True)
    valid_DL = DataLoader(valid_dataset,batch_size=1,shuffle=True)
    
    discriminator, train_dl, valid_dl, optimizer = accelerator.prepare(discriminator, train_DL, valid_DL, optimizer)
    

    def validate(discriminator,valid_dl):
    
        discriminator.eval()
        valid_loss = 0
        p = 0
        with torch.no_grad():
            for b in valid_dl:
                y = discriminator(b)
                              # decoder_input_ids=b['feedback'].squeeze(1)[:,:-1].to(device),
                              # decoder_attention_mask=b['feedback_attn'].squeeze(1)[:,:-1].to(device))
                loss = y['sent_ce_loss'] + y['avg_phrase_ce_loss'] #F.cross_entropy(y.logits.permute(0,2,1), b['feedback'].squeeze(1)[:,1:].to(device), ignore_index=tokenizer.pad_token_id)
                valid_loss += loss.item()
                p+=1
                if p>10:
                    break
                
        accelerator.print("Validation Loss:",valid_loss)
        return valid_loss
    
    discriminator.train()
    
    loss_acc = 0
    num_batches = 0
    total_steps = 0
    
    patience = PATIENCE
    
    train_loss_arr,valid_loss_arr = [],[]
    
    optimizer.zero_grad()
    discriminator.zero_grad()
    
    valid_loss = validate(discriminator,valid_dl)
    valid_loss_arr.append(valid_loss/len(valid_dl))
    best_valid_loss = valid_loss
    
    for E in range(epochs):
                
        for b in train_dl:
            with accelerator.accumulate(discriminator):
                y = discriminator(b)
                              # decoder_input_ids=b['feedback'].squeeze(1)[:,:-1].to(device),
                              # decoder_attention_mask=b['feedback_attn'].squeeze(1)[:,:-1].to(device))
                loss = y['sent_ce_loss'] + y['avg_phrase_ce_loss'] #F.cross_entropy(y.logits.permute(0,2,1), b['feedback'].squeeze(1)[:,1:].to(device), ignore_index=tokenizer.pad_token_id)
                
                accelerator.backward(loss)
                loss_acc += loss.item()
                
                num_batches += 1
                total_steps += 1
            
                train_loss_arr.append(loss_acc/num_batches)
                
                optimizer.zero_grad()
            
                if total_steps%10==0 and total_steps!=0:
                    accelerator.print(f"Epoch: {E}\tSteps taken: {total_steps:04d}\tLoss: {loss_acc/num_batches}")
                    break
                
            #print("Epoch:",E,"\t","Steps taken:",total_steps,"\tLoss:",loss_acc/num_batches)
            
            # torch.save({'model_state':discriminator.state_dict(),
            #             'optimizer':optimizer.state_dict(),
            #             'epoch':E},
            #             f"{save_dir}/Epoch_{E}_model_chkpt.pth.tar")
            
        valid_loss = validate(discriminator,valid_dl)
        valid_loss_arr.append(valid_loss/len(valid_dl))
        
        if valid_loss<best_valid_loss:
            best_valid_loss = valid_loss
            patience = PATIENCE
            
            accelerator.wait_for_everyone()
            # if accelerator.is_main_process:
            #     tokenizer.save_pretrained('Span_Llama_Checkpoints/')
            # unwrapped_model = accelerator.unwrap_model(discriminator)
            state_dict = accelerator.get_state_dict(discriminator)
            torch.save({'model_dict':state_dict},f'{save_dir}/best_model_chkpt.pth.tar')
        else:
            patience -= 1
            accelerator.print(f"REDUCING PATIENCE...{patience}")

        if patience<=0:
            accelerator.print("RUNNING OUT OF PATIENCE... TERMINATING")
            break
    
    
    return train_loss_arr,valid_loss_arr

if __name__ == '__main__':
    
    from transformers import AutoModel
    # Load model from HuggingFace Hub
    bert_chkpt = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(bert_chkpt)
    
    tokenizer.pad_token_id = 0
    
    train_dataset = feedback_QA_dataset(train_df,tokenizer=tokenizer)
    train_dataset.add_neg_samples()
    valid_dataset = feedback_QA_dataset(val_df,tokenizer=tokenizer)
    valid_dataset.add_neg_samples()

    import os

    
    EPOCHS = 50
    BATCH_SIZE = 2
    LR = 1e-5
    
    # MPNet = AutoModel.from_pretrained(bert_chkpt).to(device)
    discriminator_model = discriminator(bert_chkpt)
    
    optimizer = torch.optim.AdamW(discriminator_model.parameters(),lr=LR)
    
    save_dir = 'Detect_Span_FB_Llama_chkpts_1'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    train_loss, valid_loss = train(discriminator_model,train_dataset,valid_dataset,EPOCHS,BATCH_SIZE,optimizer,5,save_dir)


        



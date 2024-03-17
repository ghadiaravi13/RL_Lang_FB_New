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
    
    def __init__(self,df,tokenizer,max_length=2048):
        self.df = df
        self.max_len = max_length
        self.data = []
        self.tokenizer = tokenizer
        skipped = 0
        
        for i in range(len(self.df)):
            
            d = {}
            if self.df.iloc[i]['rating_class']==3:
                skipped += 1
                continue
                
            prompt = "I will give you a question, an initial answer to the question, and feedback critiquing that answer. Based on the feedback, provide a refined answer. Do NOT generate anything other than the refined answer."
            question = self.df.iloc[i]['question']
            answer = self.df.iloc[i]['answer']
            feedback = self.df.iloc[i]['explanation']
            
            tok_input = self.tokenizer(f"{prompt}\nQuestion:{question}\nAnswer:{answer}\n\nFeedback:{feedback}\n\nRefined answer: ",
                                  add_special_tokens=True
                                 )
            if len(tok_input['input_ids']) > self.max_len:
                skipped += 1
                continue
                

            
            PAD_LEN = self.max_len - len(tok_input['input_ids'])

            d['input'] = tok_input['input_ids'] + [self.tokenizer.eos_token_id]*PAD_LEN
            d['attention_mask'] = tok_input['attention_mask'] + [0]*PAD_LEN
            d['id'] = i

            for k in d.keys():
                d[k] = torch.tensor(d[k])

            self.data.append(d)
        # print(f'skipped: {skipped}')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]


def refine(bert_chkpt,val_df):
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    accelerator = Accelerator()
    device = accelerator.device

    BS = 2
    NUM_RETURN_SEQUENCES = 3
    
    for i in range(NUM_RETURN_SEQUENCES):
        val_df[f'refined_answer_{i}'] = ['None']*len(val_df)
    
    tokenizer = AutoTokenizer.from_pretrained(bert_chkpt,cache_dir='/home/jupyter/Ravi_new/HF_cache')
    model = AutoModelForCausalLM.from_pretrained(bert_chkpt,cache_dir='/home/jupyter/Ravi_new/HF_cache')
    
    valid_dataset = feedback_QA_dataset(val_df,tokenizer)
    valid_DL = DataLoader(valid_dataset,batch_size=BS,shuffle=False)

    model,valid_DL = accelerator.prepare(model,valid_DL)
    
    model.eval()
    
    with torch.no_grad():
        for b in tqdm.tqdm(valid_DL):
            out = model.generate(inputs=b['input'],
                                 attention_mask=b['attention_mask'],
                                 max_new_tokens=500,
                                 num_return_sequences=NUM_RETURN_SEQUENCES,
                                 do_sample=True
                                )
            
            l = [a.split('Refined answer: ')[1].replace('</s>','') for a in tokenizer.batch_decode(out)]
            for i in range(NUM_RETURN_SEQUENCES):
                val_df[f'refined_answer_{i}'].loc[b['id'].tolist()] = l[i::NUM_RETURN_SEQUENCES]
            #val_df['refined_answer'].loc[b['id'].tolist()] = l[:BS]
            # accelerator.print(l)
            # break
    val_df.to_csv('train_refined.csv')
    return val_df

def main():
    
    bert_chkpt = "meta-llama/Llama-2-7b-chat-hf"
    val_df = process_df(pd.DataFrame(dataset['train']))
    res = refine(bert_chkpt,val_df)
    res.to_csv('train_refined.csv')

if __name__ == '__main__':
    main()
    
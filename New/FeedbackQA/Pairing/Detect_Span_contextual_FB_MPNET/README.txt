variant with feedback learnt in context of the question and answer

cleaning Q and A: replacing '\n' with ' '

SE = <Question: Q Answer: 'A'> <padding>
FE = <Question: Q Answer: A> <sep><sep> <'FB'> <padding>

('' denotes the embeddings that are mean_pooled)

for a sentence Si, its neg FB samples F-i are derived by using <S-i,F-i> into the model, not <Si,F-i>. This is because we wish to get contextualized representation of each Fi within its best possible context, ie, we want the best possible representation of Fi and then contrast it with neg samples.

baseline_preds: preds directly from the HF checkpoint without Finetuning
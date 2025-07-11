from transformers import AutoTokenizer, EsmModel
import torch
import numpy as np
import pandas as pd

train_path = './datasets/pdbbind/random/train.csv'
df_train = pd.read_csv(train_path)
val_path = './datasets/pdbbind/random/val.csv'
df_val = pd.read_csv(val_path)

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D").cuda()

Protein_list = df_train['Protein'].tolist() + df_val['Protein'].tolist()
Protein_set = set(Protein_list)

protein_esm_dict = dict()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for sequence in Protein_set:
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state.squeeze(0)[1:-1]
        chunks = torch.split(last_hidden_states, 9, dim=0)
        mer_9 = []
        for i, chunk in enumerate(chunks):
            mer_9.append(torch.mean(chunk, dim=0).to("cpu").tolist())

        protein_esm_dict[sequence] = mer_9


np.save('./datasets/pdbbind/protein_esm3b_mer9_dict.npy', protein_esm_dict)

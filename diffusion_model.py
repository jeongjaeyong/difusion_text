import matplotlib.pyplot as plt
from transformers import AutoTokenizer, BertLMHeadModel
import torch
from torch import nn
from vector_quantize_pytorch import VectorQuantize
from diffusion_data import *
import torch.optim as optim
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class My_Diffusion(torch.nn.Module):
    def __init__(self, bert_model):
        super(My_Diffusion, self).__init__()
        self.encoder =bert_model.bert
        self.vq = VectorQuantize(
            dim=1024,
            codebook_size=1024,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.  # the weight on the commitment loss
        )
        self.decoder = bert_model.cls

    def forward(self, x):
        o1 = self.encoder(x)
        latent = self.vq(o1.last_hidden_state[0])
        output = self.decoder(latent[0].unsqueeze(0))
        return output, latent

tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
model = BertLMHeadModel.from_pretrained("bert-large-uncased")
# model.pooler.dense = nn.Linear(in_features=1024, out_features=tokenizer.vocab_size, bias=True)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')


my_diffusion = My_Diffusion(model)
my_diffusion.encoder.requires_grad = False
my_diffusion.to('cuda')
train_dataloader, valid_dataloader = get_data_load()
optimizer = optim.Adam(my_diffusion.parameters(), lr=0.00001)

train_loss = []
valid_loss = []
avg_train_loss = []
avg_valid_loss = []



epochs = 10
for epoch in range(epochs):
    count = 0
    for data in train_dataloader:
        ask_list, respoone_list, ask_mask, respoone_mask= data['ask_list'], data['respoone_list'], data['ask_mask'], data['respoone_mask']
        output, latent = my_diffusion.forward(ask_list.to('cuda'))
        loss = nn.CrossEntropyLoss()
        output_loss = loss(output[0], respoone_list.to('cuda')[0])
        optimizer.zero_grad()
        output_loss.backward()
        optimizer.step()

        train_loss.append(output_loss.item())
        torch.cuda.empty_cache()
        count += 1
        if (count+1)%10000==0:
            best_loss = np.mean(train_loss)
            torch.save(my_diffusion.state_dict(), "diffusion.pt")
            avg_train_loss.append(np.mean(train_loss))
            plt.clf()
            plt.plot(avg_train_loss)
            plt.savefig("check.png")
        train_loss = []

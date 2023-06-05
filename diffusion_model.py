import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from vector_quantize_pytorch import VectorQuantize
from diffusion_data import *
import torch.optim as optim
import numpy as np

class My_Diffusion(torch.nn.Module):
    def __init__(self, bert_model):
        super(My_Diffusion, self).__init__()
        self.embeddings = bert_model.embeddings
        self.encoder =bert_model.encoder
        self.encoder.layer = self.encoder.layer[:12]
        self.vq = VectorQuantize(
            dim=1024,
            codebook_size=1024,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.  # the weight on the commitment loss
        )
        self.decoder = bert_model.encoder
        self.decoder.layer = self.decoder.layer[12:]
        self.pooler = bert_model.pooler

    def forward(self, x):
        o1 = self.embeddings(x)
        o1 = self.encoder(o1)
        latent = self.vq(o1[0])
        output = self.decoder(latent[0].unsqueeze(0))
        output = self.pooler(output.last_hidden_state)
        return output, latent

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained("bert-large-uncased")
model.pooler.dense = nn.Linear(in_features=1024, out_features=tokenizer.vocab_size, bias=True)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')


my_diffusion = My_Diffusion(model)
my_diffusion.embeddings.requires_grad = False
my_diffusion.encoder.requires_grad = False
my_diffusion.to('cuda')
train_dataloader, valid_dataloader = get_data_load()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_loss = []
valid_loss = []
avg_train_loss = []
avg_valid_loss = []

epochs = 10
best_loss = 100
for epoch in range(epochs):
    for data in train_dataloader:
        ask_list, respoone_list, ask_mask, respoone_mask= data['ask_list'], data['respoone_list'], data['ask_mask'], data['respoone_mask']
        output, latent = my_diffusion.forward(ask_list.to('cuda'))
        output_loss = loss(output[0], respoone_list[0].to('cuda'))
        optimizer.zero_grad()
        output_loss.backward()
        optimizer.step()

        train_loss.append(output_loss.item())

    for data in valid_dataloader:
        ask_list, respoone_list, ask_mask, respoone_mask = data['ask_list'], data['respoone_list'], data['ask_mask'], \
        data['respoone_mask']
        output, latent = my_diffusion.forward(ask_list.to('cuda'))
        output_loss = loss(output[0], respoone_list[0].to('cuda'))
        valid_loss.append(output_loss.item())


    avg_train_loss.append(np.mean(train_loss))
    avg_valid_loss.append(np.mean(valid_loss))
    if np.mean(valid_loss)<best_loss:
        best_loss = np.mean(valid_loss)
        torch.save(my_diffusion.state_dict(), "diffusion.pt")

    train_loss = []
    valid_loss = []

    plt.clf()
    plt.plot(avg_train_loss, color='r')
    plt.plot(avg_valid_loss, color='r')
    plt.show()

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer
class Dataset(Dataset):
    def __init__(self, ask, response, tokenizer):
        self.ask = ask
        self.response = response
        self.tokenizer = tokenizer
        self.lenth = 32
    def __len__(self):
        return len(self.ask)

    def __getitem__(self, idx):
        ask = self.ask[idx]
        response = self.response[idx]

        ask = self.tokenizer(ask, return_tensors='pt')
        ask_mask = ask.attention_mask[0]
        ask = ask.input_ids[0]
        ask = ask[:self.lenth]
        ask_mask = ask_mask[:self.lenth]

        ask[-1] = 102
        response = self.tokenizer(response, return_tensors='pt')
        response_mask = response.attention_mask[0]
        response = response.input_ids[0]
        response = response[:self.lenth]
        response_mask = response_mask[:self.lenth]
        response[-1] = 102

        return ask, response, ask_mask, response_mask

def collate_fn(batch):
    ask_list = []
    respoone_list = []
    ask_mask = []
    respoone_mask = []
    for item in batch:
        ask_list.append(item[0])
        respoone_list.append(item[1])
        ask_mask.append(item[2])
        respoone_mask.append(item[3])

    ask_list = torch.nn.utils.rnn.pad_sequence(ask_list, batch_first=True)
    respoone_list = torch.nn.utils.rnn.pad_sequence(respoone_list, batch_first=True)
    ask_mask = torch.nn.utils.rnn.pad_sequence(ask_mask, batch_first=True)
    respoone_mask = torch.nn.utils.rnn.pad_sequence(respoone_mask, batch_first=True)

    data = {'ask_list':ask_list, 'respoone_list':respoone_list, 'ask_mask':ask_mask, 'respoone_mask':respoone_mask}
    return data

def get_data_load(is_pre_train=True):
    train = load_dataset("allenai/soda", split="train")
    valid = load_dataset("allenai/soda", split="validation")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    train_ask = []
    train_response = []

    for i in range(len(train)):
        dialogue = train[i]['dialogue']
        for index in range(len(dialogue)-1):
            train_ask.append([dialogue[index]])
            if is_pre_train:
                train_response.append([dialogue[index]])
            else:
                train_response.append([dialogue[index+1]])


    valid_ask = []
    valid_response = []
    for i in range(len(valid)):
        dialogue = train[i]['dialogue']
        for index in range(len(dialogue)-1):
            valid_ask.append([dialogue[index]])
            if is_pre_train:
                valid_response.append([dialogue[index]])
            else:
                valid_response.append([dialogue[index+1]])


    train_dataset = Dataset(train_ask, train_response, tokenizer)
    valid_dataset = Dataset(valid_ask, valid_response, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    return train_dataloader, valid_dataloader
train_dataloader, valid_dataloader = get_data_load()


import numpy as np # linear algebra
import pandas as pd
import os
import platform

from config import DATASET_DIR

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig, AutoTokenizer

from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

MODEL_OUT_DIR = f"{DATASET_DIR}/bert_regressor"
## Model Configurations
MAX_LEN_TRAIN = 205
MAX_LEN_VALID = 205
MAX_LEN_TEST = 205
BATCH_SIZE = 64
LR = 1e-3
NUM_EPOCHS = 10
NUM_THREADS = 1  ## Number of threads for collecting dataset
MODEL_NAME = 'bert-base-uncased'

if not os.path.isdir(MODEL_OUT_DIR):
    os.makedirs(MODEL_OUT_DIR)

class Text_Dataset(Dataset):
    def __init__(self, data, maxlen, tokenizer):
        self.df = data.reset_index()
        self.tokenizer = tokenizer
        self.maxlen = maxlen
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        text = self.df.loc[index, 'text']
        try:
            target = self.df.loc[index, 'target']
        except:
            target = 0.0

        #Preprocess the text to be suitable for the transformer
        tokens = self.tokenizer.tokenize(text) 
        tokens = ['[CLS]'] + tokens + ['[SEP]'] 
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] 
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] 
        #Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        input_ids = torch.tensor(input_ids) 
        #Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        
        target = torch.tensor(target, dtype=torch.float32)
        
        return input_ids, attention_mask, target
    
class BertRegressor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        #The output layer that takes the [CLS] representation and gives an output
        self.cls_layer1 = nn.Linear(config.hidden_size,128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128,128)
        self.tanh1 = nn.Tanh()
        self.ff2 = nn.Linear(128,1)

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        output = self.tanh1(output)
        output = self.ff2(output)
        return output

def get_rmse(output, target):
    err = torch.sqrt(metrics.mean_squared_error(target, output))
    return err

def evaluate(model, criterion, dataloader, device):
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0

    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):
            
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)
            
            mean_loss += criterion(output, target.type_as(output)).item()
#             mean_err += get_rmse(output, target)
            count += 1
            
    return mean_loss/count

def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    best_acc = 0
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        train_loss = 0
        for i, (input_ids, attention_mask, target) in enumerate(iterable=train_loader):
            optimizer.zero_grad()  
            
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(output, target.type_as(output))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"Training loss is {train_loss/len(train_loader)}")
        val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
        print("Epoch {} complete! Validation Loss : {}".format(epoch, val_loss))

def predict(model, dataloader, device):
    model.eval()
    predicted_labels = []
    actual_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, target in dataloader:
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)
            predicted_labels.extend(output.cpu().numpy())
            actual_labels.extend(target.cpu().numpy())

    return np.array(predicted_labels), np.array(actual_labels)

if __name__ == "__main__":

    system = platform.system()
    if system == "Darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load data
    df = pd.read_csv(f"{DATASET_DIR}/dataset_bert.csv")
    df.rename(
        columns={
            "Housing-Unit-Risk": "target"
        }
    )
    df_train, df_test = train_test_split(df, test_size=0.05, random_state=42)

    train_data, validation = train_test_split(df_train, test_size=0.1, random_state=21)

    word_count_train = train_data['text'].apply(lambda x: len(x.split()))
    word_count_valid = validation['text'].apply(lambda x: len(x.split()))
    word_count_test = df_test['text'].apply(lambda x: len(x.split()))

    MAX_LEN_TRAIN = word_count_train.max()
    MAX_LEN_VALID = word_count_valid.max()
    MAX_LEN_TEST = word_count_test.max()

    config = AutoConfig.from_pretrained(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = BertRegressor.from_pretrained(MODEL_NAME, config=config)
    
    ## Putting model to device
    model = model.to(device)
    ## Takes as the input the logits of the positive class and computes the binary cross-entropy 
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    ## Optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=LR)

    ## Training Dataset
    train_set = Text_Dataset(data=train_data, maxlen=MAX_LEN_TRAIN, tokenizer=tokenizer)
    valid_set = Text_Dataset(data=validation, maxlen=MAX_LEN_VALID, tokenizer=tokenizer)
    test_set = Text_Dataset(data=df_test, maxlen=MAX_LEN_TEST, tokenizer=tokenizer)

    ## Data Loaders
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)

    train(
        model=model, 
        criterion=criterion,
        optimizer=optimizer, 
        train_loader=train_loader,
        val_loader=valid_loader,
        epochs = 10,
        device = device)

    predicted_labels, actual_labels = predict(model, test_loader, device)

    mse = metrics.mean_squared_error(actual_labels, predicted_labels)
    rmse = np.sqrt(mse)
    print(f'MSE: {mse}, RMSE: {rmse}')
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import torch.utils.data as data
from torch import tensor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import json
from sklearn.model_selection import train_test_split

with open('BERT_classification/shakespeare_text_forBERT.txt','r', encoding='utf-8') as shk_txt:
    shk_data = shk_txt.readlines()
with open('BERT_classification/formal_text_forBERT.txt','r', encoding='utf-8') as frm_txt:
    frm_data = frm_txt.readlines()
with open('BERT_classification/informal_text_forBERT.txt','r', encoding='utf-8') as inf_txt:
    inf_data = inf_txt.readlines()

def gen_labels(label_len, style_tag):
    label_list = []
    for i in range(label_len):
        label_list.append(style_tag)
    return label_list

shk_labels = gen_labels(len(shk_data),'[SHK]')
frm_labels = gen_labels(len(frm_data),'[FRM]')
inf_labels = gen_labels(len(inf_data),'[INF]')


full_text = np.concatenate([shk_data,frm_data,inf_data])
full_labels = np.concatenate([shk_labels,frm_labels,inf_labels])

#shuffle data set
indices = np.arange(full_text.shape[0])
np.random.shuffle(indices)
full_text = full_text[indices]
full_labels = full_labels[indices]

X_train, X_test, y_train, y_test = train_test_split(full_text, full_labels, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125)

dataset = {
    'train':{'text':X_train,'labels':y_train},
    'validation':{'text':X_val,'labels':y_val},
    'test':{'text':X_test,'labels':y_test}
}
LABELS = ['[SHK]','[FRM]','[INF]']
try:
    llm_path = '/opt/models/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
except:
    llm_path = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

NUM_EPOCHS = 12 #15 epochs of training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prep_data(dataset):
   text = [sample['text'] for sample in dataset]
   labels = [sample['labels'][0] for sample in dataset]
   return text, labels

def get_tensor_dataset(tokens, labels):
   data_seq = tensor(tokens['input_ids'])
   data_mask = tensor(tokens['attention_mask'])
   data_label = tensor(labels)

   return TensorDataset(data_seq, data_mask, data_label)
      

def preproccess(dataset):    
    train_text, train_labels = prep_data(dataset['train'])
    val_text, val_labels = prep_data(dataset['validation'])
    test_text, test_labels = prep_data(dataset['test'])

    train_tokenized = tokenizer(train_text, padding='max_length', truncation=True, return_tensors='pt')
    val_tokenized = tokenizer(val_text, padding='max_length', truncation=True, return_tensors='pt')
    test_tokenized = tokenizer(test_text, padding='max_length', truncation=True, return_tensors='pt')

    train_data = get_tensor_dataset(train_tokenized, train_labels)
    val_data = get_tensor_dataset(val_tokenized, val_labels)
    test_data = get_tensor_dataset(test_tokenized, test_labels)

    return train_data, val_data, test_data

def train_model(model,optimizer,dataloader,loss_func):
    model.train()
    total_loss = 0
    for batch in dataloader:
       batch = [sample.to(device) for sample in batch]
       id, mask, labels = batch
       
       model.zero_grad()
       output = model(id,mask)
       preds = output.logits
       loss = loss_func(preds, labels)
       total_loss += loss.item()
       loss.backward()

       optimizer.step()

    avg_loss = total_loss/len(dataloader)
    return avg_loss

def eval_model(model,dataloader,loss_func):
    model.eval()
    total_loss = 0
    for batch in dataloader:
       batch = [sample.to(device) for sample in batch]
       id, mask, labels = batch
       with torch.no_grad():
            output = model(id,mask)
            preds = output.logits
            loss = loss_func(preds, labels)
            total_loss += loss.item()

    avg_loss = total_loss/len(dataloader)
    return avg_loss

def test_model(model, dataloader, loss_func):
    model.eval()
    total_loss = 0

    true_labels = []
    pred_labels = []
    for batch in dataloader:
       batch = [sample.to(device) for sample in batch]
       id, mask, labels = batch
       with torch.no_grad():
            output = model(id,mask)
            preds = output.logits
            loss = loss_func(preds, labels)
            total_loss += loss.item()

            pred_labels.extend(torch.argmax(preds,dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    task_report = classification_report(true_labels, pred_labels, target_names=LABELS)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, task_report

def fine_tune(model, lr, train_dataloader, val_dataloader): #trains the model

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    train_history = {}
    val_history = {}

    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, optimizer, train_dataloader, loss_func)
        val_loss = eval_model(model, val_dataloader, loss_func)
        best_loss = min(val_loss,best_loss)
            
        train_history.update({epoch:train_loss})
        val_history.update({epoch:val_loss})
    return best_loss, train_history, val_history

def make_plot(train_hist, val_hist,name):
    train_epochs = [res for res in train_hist.keys()]
    train_results = [res for res in train_hist.values()]

    val_epochs = [res for res in val_hist.keys()]
    val_results = [res for res in val_hist.values()]
    plt.plot(train_epochs,train_results,color='g')
    plt.plot(val_epochs,val_results,color='m')
    plt.legend(["Train cost", "Validation cost"])
    plt.ylabel("Loss")
    plt.xlabel("Epochs of Training")
    plt.title("Train and Validation Loss per Epoch")
    plt.savefig(name)
    return

def run_tests():
    train, val, test = preproccess(dataset)

    train_dataloader = DataLoader(train, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=16, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        llm_path,
        num_labels=3
        )
    model.to(device)
    val_loss, train_history, val_history = fine_tune(model,5e-5, train_dataloader, val_dataloader)
    make_plot(train_history,val_history,f"mod_data_llm.png")

    model_loss, model_results = test_model(model,test_dataloader, nn.CrossEntropyLoss())

    #save for model for later use
    model.save_pretrained('./model_gen_eval')

    with open("model_results.txt","w") as out_file1:
        out_file1.write(f"Validation Loss: {val_loss}\n")
        out_file1.write(f"Test Loss: {model_loss}\n")
        out_file1.writelines(model_results)

try:
    run_tests()
except Exception as e:
    with open("errors.txt","w") as errors:
        errors.write(str(e))
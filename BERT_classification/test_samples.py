import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import torch.utils.data as data
from torch import tensor
from transformers import BertTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import json
from sklearn.model_selection import train_test_split

#RNN
#'/content/generated_shakes.txt'
#'/content/generated_formals.txt'
#'/content/generated_informals.txt'

#MCM
#'/content/MCM_shakes.txt'
#'/content/MCM_formal.txt'
#'/content/MCM_informal.txt'

#Baseline
#'/content/200_shakes.txt'
#'/content/200_formal.txt'
#'/content/200_informal.txt'

with open('/content/200_shakes.txt','r', encoding='utf-8') as shk_txt:
    shk_data = shk_txt.readlines()
with open('/content/200_formal.txt','r', encoding='utf-8') as frm_txt:
    frm_data = frm_txt.readlines()
with open('/content/200_informal.txt','r', encoding='utf-8') as inf_txt:
    inf_data = inf_txt.readlines()

def gen_labels(label_len, style_tag):
    label_list = []
    for i in range(label_len):
        label_list.append(style_tag)
    return label_list

shk_labels = gen_labels(len(shk_data),0)
frm_labels = gen_labels(len(frm_data),1)
inf_labels = gen_labels(len(inf_data),2)


full_text = np.concatenate([shk_data,frm_data,inf_data])
full_labels = np.concatenate([shk_labels,frm_labels,inf_labels])

#shuffle data set
indices = np.arange(full_text.shape[0])
np.random.shuffle(indices)
full_text = full_text[indices]
full_labels = full_labels[indices]

dataset = {
    'test':[{'text':text,'labels':label} for text, label in zip(full_text,full_labels)]
}
LABELS = ['[SHK]','[FRM]','[INF]']
try:
    llm_path = '/opt/models/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(llm_path)
except:
    llm_path = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(llm_path)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prep_data(dataset):
   text = [sample['text'] for sample in dataset]
   labels = [sample['labels'] for sample in dataset]
   return text, labels

def get_tensor_dataset(tokens, labels):
   data_seq = tensor(tokens['input_ids'])
   data_mask = tensor(tokens['attention_mask'])
   data_label = tensor(labels)

   return TensorDataset(data_seq, data_mask, data_label)


def preproccess(dataset):
    test_text, test_labels = prep_data(dataset['test'])

    test_tokenized = tokenizer(test_text, padding='max_length', truncation=True, return_tensors='pt')

    test_data = get_tensor_dataset(test_tokenized, test_labels)

    return test_data

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

def run_tests():
    test = preproccess(dataset)

    test_dataloader = DataLoader(test, batch_size=16, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        '/content/drive/My Drive/550_final_proj/model_gen_better',
        num_labels=3
        )
    model.to(device)
    model_loss, model_results = test_model(model,test_dataloader, nn.CrossEntropyLoss())

    with open("/content/drive/My Drive/550_final_proj/RNN_results.txt","w") as out_file1:
        out_file1.write(f"Test Loss: {model_loss}\n")
        out_file1.writelines(model_results)

run_tests()
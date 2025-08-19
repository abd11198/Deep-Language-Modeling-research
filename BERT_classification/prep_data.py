import numpy as np
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
print(full_text.shape)
full_labels = np.concatenate([shk_labels,frm_labels,inf_labels])

#shuffle data set
indices = np.arange(full_text.shape[0])
np.random.shuffle(indices)
full_text = full_text[indices]
full_labels = full_labels[indices]

X_train, X_test, y_train, y_test = train_test_split(full_text, full_labels, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125)

print(len(X_train),len(X_val),len(X_test))
dataset = {
    'train':{'text':X_train,'labels':y_train},
    'validation':{'text':X_val,'labels':y_val},
    'test':{'text':X_test,'labels':y_test}
}
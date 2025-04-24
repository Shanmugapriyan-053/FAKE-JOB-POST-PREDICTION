#Load Dataset

import pandas as pd
df = pd.read_excel("/content/preprocessed_fake_job_postings.xlsx")
print(df.columns)  

#Drop Missing/Null Values

import pandas as pd
df = pd.read_excel("/content/preprocessed_fake_job_postings.xlsx")
print(df.columns)  
df.rename(columns={'description': 'description'}, inplace=True)  
df.dropna(subset=['description' ], inplace=True)

#Clean Text

import re
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"[^A-Za-z0-9\s]", "", text) 
    text = text.lower()  
    return text
df['description'] = df['description'].apply(clean_text)

#Label Encode

import pandas as pd
df = pd.read_excel("/content/preprocessed_fake_job_postings.xlsx")
print(df.columns)  # Check what columns are present
if 'label' not in df.columns:
    df['label'] = 0  
df['label'] = df['label'].map({'fraudulent': 1, 'real': 0})  

#Split Data

from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['description'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)


#Conver to Tensor Dataset

import torch
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels)
)
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    torch.tensor(test_labels)
)



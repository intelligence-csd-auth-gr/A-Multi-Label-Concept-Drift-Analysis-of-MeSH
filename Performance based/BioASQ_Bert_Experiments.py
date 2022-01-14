# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:52:00 2021

@author: room5
"""
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
#from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from torchmetrics.functional import auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm
import numpy as np


RANDOM_SEED=42
MAX_TOKEN_COUNT=128
pl.seed_everything(RANDOM_SEED)

#train_data = pd.read_csv(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\mesh_2013.csv')
#test_data = pd.read_csv(r'C:\Users\room5\PycharmProjects\Concept Drift for MeSH\mesh_2014.csv')
train_data = pd.read_csv('//home//myloniko//bioasq_bert//mesh_2013.csv')
test_data = pd.read_csv('//home//myloniko//bioasq_bert//mesh_2019.csv')


string_labels = train_data.target.values.tolist()
string_labels_test = test_data.target.values.tolist()

labels=list()
labels_test=list()
for label in string_labels:
    labels.append(label.split("#"))
for label_test in string_labels_test:
    labels_test.append(label_test.split("#"))    
del string_labels,string_labels_test
mlb = MultiLabelBinarizer()
mlb.fit(labels)



############################################### Transform without batches #######################################
labels=mlb.transform(labels)
labels_test = mlb.transform(labels_test)
train_data.target = labels.tolist()
test_data.target = labels_test.tolist()
print(train_data.target)
print(test_data.target)

del labels,labels_test



BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

class BioASQ_Dataset(Dataset):
  def __init__(
    self,
    data: pd.DataFrame,
    tokenizer: BertTokenizer,
    max_token_len: int = 512
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
  def __len__(self):
    return len(self.data)
  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]
    text = data_row.text
    labels=data_row.target
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return dict(
      text=text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor(labels)
    )




#%%
"""
######################################## Testing Cell ##############################################################

train_dataset = BioASQ_Dataset(
  train_data,
  tokenizer,
  max_token_len=512
)

#sample_item = train_dataset[0]
#print(sample_item['text'])
#print(sample_item['labels'])

bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
sample_batch = next(iter(DataLoader(train_dataset, batch_size=8, num_workers=0)))
print(sample_batch["input_ids"].shape)
print(sample_batch["attention_mask"].shape)

output = bert_model(sample_batch["input_ids"], sample_batch["attention_mask"])

print(output.last_hidden_state.shape) 
print(output.pooler_output.shape)
"""
#%%

class BioASQ_DataModule(pl.LightningDataModule):
  def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len
  def setup(self, stage=None):
    self.train_dataset = BioASQ_Dataset(
      self.train_df,
      self.tokenizer,
      self.max_token_len
    )
    self.test_dataset = BioASQ_Dataset(
      self.test_df,
      self.tokenizer,
      self.max_token_len
    )
  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=0
    )
  def val_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      num_workers=0
    )
  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=0
    )


N_EPOCHS = 10
BATCH_SIZE = 16
data_module = BioASQ_DataModule(
  train_data,
  test_data,
  tokenizer,
  batch_size=BATCH_SIZE,
  max_token_len=MAX_TOKEN_COUNT
)

#%%

class BioASQ_BertModel(pl.LightningModule):
  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCELoss()
  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output
  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}
  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss
  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss
  def training_epoch_end(self, outputs):
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)
    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)
    for i in range(0,len(mlb.classes_)):
      class_roc_auc = auroc(predictions[:,i], labels[:,i])
      name = mlb.classes_[i]
      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )
    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )



steps_per_epoch=len(train_data) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5

model = BioASQ_BertModel(
  n_classes=len(mlb.classes_),
  n_warmup_steps=warmup_steps,
  n_training_steps=total_training_steps
)

#%%

checkpoint_callbac = ModelCheckpoint(
  dirpath="checkpoints",
  filename="best-checkpoint",
  save_top_k=1,
  verbose=True,
  monitor="val_loss",
  mode="min"
)


logger = TensorBoardLogger("lightning_logs", name="BioASQ")
#early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)


trainer = pl.Trainer(
  logger=logger,
  checkpoint_callback=checkpoint_callbac,
  callbacks=[checkpoint_callbac],
  max_epochs=N_EPOCHS,
  gpus=1,
  progress_bar_refresh_rate=30
)

#%%
trainer.fit(model, data_module)
trainer.save_checkpoint('model.ckpt')

#%%

trained_model = BioASQ_BertModel.load_from_checkpoint(
  trainer.checkpoint_callback.best_model_path,
  n_classes=len(mlb.classes_)
)
trained_model.eval()
trained_model.freeze()

#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model = trained_model.to(device)
test_dataset = BioASQ_Dataset(
  test_data,
  tokenizer,
  max_token_len=MAX_TOKEN_COUNT
)

predictions = []
labels = []
for item in tqdm(test_dataset):
  _, prediction = trained_model(
    item["input_ids"].unsqueeze(dim=0).to(device),
    item["attention_mask"].unsqueeze(dim=0).to(device)
  )
  predictions.append(prediction.flatten())
  labels.append(item["labels"].int())
predictions = torch.stack(predictions).detach().cpu()
labels = torch.stack(labels).detach().cpu()
print(predictions)
print(labels)
pickle.dump(predictions, open('model_predictions.pkl', 'wb'))
pickle.dump(labels, open('test_set_true.pkl', 'wb'))
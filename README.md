# BERT_with_keras
This is a implementation of **BERT**(**B**idirectional **E**ncoder **R**epresentation of **T**ransformer) with **Keras**.

The backend of Keras must be **tensorflow**.

## Usage

Here is a quick-start example to preprocess raw data for pretraining and fine-tuning for text classification.
For more details, see [Predicting Movie Review Sentriment with BERT](https://github.com/miroozyx/BERT_with_keras/blob/master/movie_reviews_classification.ipynb)

### Data
Let's use Standord's Large Movie Review Dataset for **BERT** pretraining and fine-tuning, the code below, which downloads,extracts and imports the dateset, is 
borrowed from this [tensorflow tutorial](https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub). The
dataset consists of IMDB movie reviews labeled by positivity from 1 to 10.
```python
import os
import re
import tensorflow as tf
import pandas as pd

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz", 
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
        extract=True)
  
    train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                         "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                          "aclImdb", "test"))
    return train_df, test_df
 
train, test = download_and_load_datasets()
```

### pre-training

let's train a bert pre-training model.
```python
import os
import spacy
from const import bert_data_path,bert_model_path
from preprocess import create_pretraining_data_from_docs
from pretraining import bert_pretraining
nlp = spacy.load('en')

# use IMDB movie review as pretraining data
texts = train['sentence'].tolist() + test['sentence'].tolist()

sentences_texts=[]
for text in texts:
    doc = nlp(text)
    sentences_texts.append([s.text for s in doc.sents])

vocab_path = os.path.join(bert_data_path, 'vocab.txt')

create_pretraining_data_from_docs(sentences_texts,
                                  vocab_path=vocab_path,
                                  save_path=os.path.join(bert_data_path,'pretraining_data.npz'),
                                  token_method='wordpiece',
                                  language='en',
                                  dupe_factor=10)

bert_pretraining(train_data_path=os.path.join(bert_data_path,'pretraining_data.npz'),
                 bert_config_file=os.path.join(bert_data_path, 'bert_config.json'),
                 save_path=bert_model_path,
                 batch_size=32,
                 seq_length=128,
                 max_predictions_per_seq=20,
                 val_batch_size=32,
                 multi_gpu=0,
                 num_warmup_steps=1000,
                 checkpoints_interval_steps=1000,
                 pretraining_model_name='bert_pretraining.h5',
                 encoder_model_name='bert_encoder.h5')
```
Then, pertraining data would be found in save_dir. 
### Fine-tuning
You can use the pre-training model as the initial point for your NLP model. 
For example, you can use the pre-training model to init a classfier model. 
```python
import os
import keras
import numpy as np
from const import bert_data_path, bert_model_path
from modeling import BertConfig
from classifier import SingleSeqDataProcessor, convert_examples_to_features, Text_Classifier, save_features, TextSequence
from tokenization import FullTokenizer
from optimization import AdamWeightDecayOpt
from checkpoint import StepModelCheckpoint

# data preprossing
train_examples = SingleSeqDataProcessor.get_train_examples(train_data=train['sentence'].tolist(),labels=train['polarity'].tolist())
dev_exmaples = SingleSeqDataProcessor.get_dev_examples(dev_data=test['sentence'].tolist(), labels=test['polarity'].tolist())

# `word piece tokenizer` need to a prepared vocabulary.
vocab_path = os.path.join(bert_data_path, 'vocab.txt')

# load vocab to tokenizer
tokenizer = FullTokenizer(vocab_path, do_lower_case=True)

# convert the train and dev examples to features
train_features = convert_examples_to_features(train_examples, 
                                              label_list=[0,1], 
                                              max_seq_length=128, 
                                              tokenizer= tokenizer)
dev_features = convert_examples_to_features(dev_exmaples, label_list=[0,1], max_seq_length=128, tokenizer=tokenizer)

# convert features to a dictionary of numpy arrays.
train_features_array_dict = save_features(features=train_features)
dev_features_array_dict = save_features(features=dev_features)

# get train and validation data
train_x = [train_features_array_dict['input_ids'], train_features_array_dict['input_mask'], train_features_array_dict['segment_ids']]
train_y = keras.utils.to_categorical(train_features_array_dict['label_ids'], 2)
val_x = [dev_features_array_dict['input_ids'], dev_features_array_dict['input_mask'], dev_features_array_dict['segment_ids']]
val_y = keras.utils.to_categorical(dev_features_array_dict['label_ids'],2)

# load bert configuration file
config = BertConfig.from_json_file(os.path.join(bert_data_path, 'bert_config.json'))
epochs = 3
num_gpus = None
# if you come across a OOM problem, reduce the batch size.
batch_size = 16

# calculation the number of training steps by epoch size.
num_train_samples = len(train_features_array_dict['input_ids'])
num_train_steps = int(np.ceil(num_train_samples / batch_size)) * epochs
print("number of train steps: {}".format(num_train_steps))

# Use weight decay adam optimizer. this optimizer is sightly different with Keras's Standard Adam optimizer. 
# For more details, view source code of AdamWeightDecayOpt.
adam = AdamWeightDecayOpt(
        lr=5e-5,
        num_train_steps=num_train_steps,
        num_warmup_steps=100,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        weight_decay_rate=0.01,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
    )

# This checkpoint evaluate the bert model performance on batch end.
checkpoint = StepModelCheckpoint(filepath="%s/%s" % (bert_model_path, 'imdb_classifer_model.h5'),
                                 verbose=1, monitor='val_acc',
                                 save_best_only=True,
                                 xlen=3,
                                 period=100,
                                 start_step=100,
                                 val_batch_size=128)
# create a model
classifier = Text_Classifier(bert_config=config,
                             pretrain_model_path=os.path.join(bert_model_path, 'bert_movie_reviews_encoder.h5'),
                             batch_size=batch_size,
                             seq_length=128,
                             optimizer=adam,
                             num_classes=2,
                             metrics=['acc'],
                             multi_gpu= num_gpus
                             )

# when using multi-gpus, the parallel model of bert cann't be used to evaluate/predict.
# You can only use the cpu_build model to evalate and predict.
if num_gpus is not None:
    checkpoint.single_gpu_model = classifier.model

# train model
generator= TextSequence(x=train_x,y=train_y,batch_size=batch_size)
history = classifier.fit_generator(generator=generator,
                                   epochs=epochs,
                                   shuffle=True,
                                   callbacks=[checkpoint],
                                   validation_data=(val_x,val_y)
                                   )                                
```



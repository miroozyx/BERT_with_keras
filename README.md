# BERT_with_keras
This is a implementation of **BERT**(**B**idirectional **E**ncoder **R**epresentation of **T**ransformer) with **Keras**.

## Usage

Here is a quick-start example to preprocess raw data for pretraining and fine-tuning.

### pre-training

let's train a bert pre-training model.
```python
import os
import spacy
from const import bert_data_path,bert_model_path
from preprocess import create_pretraining_data_from_docs
from pretraining import bert_pretraining
nlp = spacy.load('en')
texts =[
'The history of natural language processing generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published an article titled "Intelligence" which proposed what is now called the Turing test as a criterion of intelligence.',
'The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors claimed that within three or five years, machine translation would be a solved problem.[2] However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced. Little further research in machine translation was conducted until the late 1980s, when the first statistical machine translation systems were developed',
'Some notably successful natural language processing systems developed in the 1960s were SHRDLU, a natural language system working in restricted "blocks worlds" with restricted vocabularies, and ELIZA, a simulation of a Rogerian psychotherapist, written by Joseph Weizenbaum between 1964 and 1966. Using almost no information about human thought or emotion, ELIZA sometimes provided a startlingly human-like interaction. When the "patient" exceeded the very small knowledge base, ELIZA might provide a generic response, for example, responding to "My head hurts" with "Why do you say your head hurts?"',
'During the 1970s, many programmers began to write "conceptual ontologies", which structured real-world information into computer-understandable data. Examples are MARGIE (Schank, 1975), SAM (Cullingford, 1978), PAM (Wilensky, 1978), TaleSpin (Meehan, 1976), QUALM (Lehnert, 1977), Politics (Carbonell, 1979), and Plot Units (Lehnert 1981). During this time, many chatterbots were written including PARRY, Racter, and Jabberwacky'
]
sentences_texts=[]
for text in texts:
    doc = nlp(text)
    sentences_texts.append([s.text for s in doc.sents])

vocab_path = os.path.join(bert_data_path, 'vocab.txt')

create_pretraining_data_from_docs(sentences_texts,
                                  vocab_path=vocab_path,
                                  save_dir=bert_data_path,
                                  token_method='wordpiece',
                                  language='en',
                                  dupe_factor=10)

bert_pretraining(train_data_path=bert_data_path,
                 bert_config_file=os.path.join(bert_data_path, 'bert_config.json'),
                 save_path=bert_model_path,
                 batch_size=32,
                 seq_length=128,
                 max_predictions_per_seq=20,
                 val_batch_size=32,
                 multi_gpu=0,
                 num_warmup_steps=1,
                 checkpoints_interval_steps=1,
                 pretraining_model_name='bert_pretraining.h5',
                 encoder_model_name='bert_encoder.h5')
```
Then, pertraining data would be found in save_dir. 
### Fine-tuning
You can use the pre-training model as the initial point for your NLP model. 
For example, you can use the pre-training model to init a classfier model. 
```python
import os
from keras.layers import Input, Dropout, Dense
from keras.models import Model
from const import bert_data_path, bert_model_path
from modeling import BertConfig, BertModel

config = BertConfig.from_json_file(os.path.join(bert_data_path, 'bert_config.json'))
bert = BertModel(config,
                     batch_size=32,
                     seq_length=128,
                     max_predictions_per_seq=20,
                     use_token_type=True,
                     embeddings_matrix=None,
                     mask=True
                     )
bert_encoder = bert.get_bert_encoder()
bert_encoder.load_weights(os.path.join(bert_model_path, 'bert_encoder.h5'))
input_ids = Input(shape=(128,))
input_mask = Input(shape=(128,))
input_token_type_ids = Input(shape=(128,))
pooled_output = bert_encoder([input_ids, input_mask, input_token_type_ids])
bert_encoder = Dropout(0.1)(pooled_output)
pred = Dense(units=2,
             activation='softmax')(bert_encoder)
model = Model(inputs=[input_ids, input_mask, input_token_type_ids], outputs=pred)
model.compile(...)
model.fit(...)
```



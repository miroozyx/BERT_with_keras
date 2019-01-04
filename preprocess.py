# author: Kris Zhang
import os
import spacy
import numpy as np
from keras.utils import to_categorical
from tokenization import FullTokenizer,SpacyTokenizer
from keras_preprocessing.sequence import pad_sequences
from create_pretraining_data import create_training_instances

def create_pretraining_data_from_docs(docs, save_path, vocab_path, token_method='wordpiece',language='en',
                                    max_seq_length=128, dupe_factor=10, short_seq_prob=0.1, masked_lm_prob=0.15,
                                    max_predictons_per_seq=20):
    """docs: sequence of sequence of sentences.

    Args:
        docs: Sequence of sequence. Docs is a sequence of documents.
            A document is a sequence of sentences.
        save_path: path to save pretraining data.
        vocab_path: The vocabulary file that the BERT model was trained on.
            only enable when token_method='wordpiece'.
        token_method: string. 'wordpiece' or 'spacy'
        language: string. 'en' or 'chn'
        max_seq_length: integer. Maximum sequence length.
        dupe_factor: integer. Number of times to duplicate the input data (with different masks).
        short_seq_prob: float. Probability of creating sequences which are shorter than the maximum length.
        masked_lm_prob: float. Masked LM probability.
        max_predictons_per_seq: integer. Maximum number of masked LM predictions per sequence.
    """

    if not hasattr(docs,'__len__'):
        raise ValueError("`docs` should be sequence of sequence.")
    else:
        if not hasattr(docs[0], '__len__'):
            raise ValueError("`docs` should be sequence of sequence.")
    if token_method not in ['wordpiece', 'spacy']:
        raise ValueError("`token_method` must be one of `wordpiece` and `spacy`.")
    if language not in ['en','chn']:
        raise ValueError("`language` should be one of `en` and `chn`.")

    if token_method == "spacy" and language == "chn":
        raise ValueError("spacy tokenizer only enable when `language` is `en`.")

    if token_method == "wordpiece":
        tokenizer = FullTokenizer(vocab_path, do_lower_case=True)
    else:
        tokenizer = SpacyTokenizer(vocab_path, do_lower_case=True)

    instances = create_training_instances(docs,
                                          tokenizer=tokenizer,
                                          max_seq_length=max_seq_length,
                                          dupe_factor=dupe_factor,
                                          short_seq_prob=short_seq_prob,
                                          masked_lm_prob=masked_lm_prob,
                                          max_predictions_per_seq=max_predictons_per_seq)

    pretraining_data = dict(tokens=[],
                            segment_ids=[],
                            is_random_next=[],
                            masked_lm_positions=[],
                            masked_lm_labels=[])

    for i, instance in enumerate(instances):
        if i < 10:
            print("num-{}: {}".format(i, instance))
        pretraining_data['tokens'].append(instance.tokens)
        pretraining_data['segment_ids'].append(instance.segment_ids)
        pretraining_data['is_random_next'].append(int(instance.is_random_next))
        pretraining_data['masked_lm_positions'].append(instance.masked_lm_positions)
        pretraining_data['masked_lm_labels'].append(instance.masked_lm_labels)

    tokens_ids = []
    tokens_mask = []
    for tokens in pretraining_data['tokens']:
        sub_ids = tokenizer.convert_tokens_to_ids(tokens)
        sub_mask = [1] * len(sub_ids)
        tokens_ids.append(sub_ids)
        tokens_mask.append(sub_mask)

    masked_lm_ids = []
    for mask_labels in pretraining_data['masked_lm_labels']:
        sub_masked_lm_ids = tokenizer.convert_tokens_to_ids(mask_labels)
        masked_lm_ids.append(sub_masked_lm_ids)

    # input
    tokens_ids = pad_sequences(tokens_ids, maxlen=128, padding='post', truncating='post')
    tokens_mask = pad_sequences(tokens_mask, maxlen=128, padding='post', truncating='post')
    segment_ids = pad_sequences(pretraining_data['segment_ids'], maxlen=128, padding='post', truncating='post')
    masked_lm_positions = pad_sequences(pretraining_data['masked_lm_positions'], maxlen=20, padding='post',
                                        truncating='post')
    # label
    is_random_next = to_categorical(pretraining_data['is_random_next'], num_classes=2)
    masked_lm_labels = pad_sequences(masked_lm_ids, maxlen=20, padding='post', truncating='post')

    # save
    np.savez(file=save_path,
             tokens_ids = tokens_ids,
             tokens_mask = tokens_mask,
             segment_ids = segment_ids,
             is_random_next = is_random_next,
             masked_lm_positions = masked_lm_positions,
             masked_lm_labels = masked_lm_labels)
    # np.save(os.path.join(save_dir, 'tokens_ids.npy'), tokens_ids)
    # np.save(os.path.join(save_dir, 'tokens_mask.npy'), tokens_mask)
    # np.save(os.path.join(save_dir, 'segment_ids.npy'), segment_ids)
    # np.save(os.path.join(save_dir, 'is_random_next.npy'), is_random_next)
    # np.save(os.path.join(save_dir, 'masked_lm_positions.npy'), masked_lm_positions)
    # np.save(os.path.join(save_dir, 'masked_lm_labels.npy'), masked_lm_labels)
    print("[INFO] number of train data:",len(tokens_ids))
    print("[INFO] is_random_next ratio:",np.sum(pretraining_data['is_random_next'])/len(is_random_next))


if __name__ == "__main__":
    from .const import bert_data_path

    vocab_path = os.path.join(bert_data_path, 'vocab.txt')
    text_list = [[],[]]
    create_pretraining_data_from_docs(text_list,
                                    vocab_path=vocab_path,
                                    save_path=os.path.join(bert_data_path, 'bert_pertraining_data.npz'),
                                    token_method='wordpiece',
                                    language='en',
                                    dupe_factor=10)

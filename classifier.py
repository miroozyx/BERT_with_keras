# author: Kris Zhang
import os
import numpy as np
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras import initializers, losses
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from modeling import BertConfig, BertModel


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class SingleSeqDataProcessor(object):
    """data converter for single sequence classification data sets."""

    @classmethod
    def get_train_examples(self, train_data, labels):
        if not isinstance(train_data, list):
            raise ValueError("`train_data` should be a list.")
        if not isinstance(labels, list):
            raise ValueError("`label` should be a list.")
        if len(train_data) != len(labels):
            raise ValueError("`train_data` and `labels` should have the same length.")
        examples = []
        for i, sequence in enumerate(train_data):
            guid = "train-%d" % (i)
            text_a = sequence
            label = labels[i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label)
            )
        return examples

    @classmethod
    def get_dev_examples(self, dev_data, labels):
        if not isinstance(dev_data, list):
            raise ValueError("`dev_data` should be a list.")
        if not isinstance(labels, list):
            raise ValueError("`label` should be a list.")
        if len(dev_data) != len(labels):
            raise ValueError("`dev_data` and `labels` should have the same length.")
        examples = []
        for i, sequence in enumerate(dev_data):
            guid = "dev-%d" % (i)
            text_a = sequence
            label = labels[i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label)
            )
        return examples

    @classmethod
    def get_test_examples(self, test_data):
        if not isinstance(test_data, list):
            raise ValueError("`dev_data` should be a list.")
        examples = []
        for i, sequence in enumerate(test_data):
            guid = "test-%d" % (i)
            text_a = sequence
            examples.append(
                InputExample(guid=guid, text_a=text_a)
            )
        return examples


class SeqPairDataProcessor(object):
    """data converter for single sequence classification data sets."""

    @classmethod
    def get_train_examples(self, train_data_a, train_data_b, labels):
        for data in [train_data_a, train_data_b]:
            if not isinstance(data, list):
                raise ValueError("`%s` should be a list." % (data))
        if not isinstance(labels, list):
            raise ValueError("`label` should be a list.")
        if len(train_data_a) != len(train_data_b) != len(labels):
            raise ValueError("`train_data_a`, `train_data_b` and `labels` should have the same length.")
        examples = []
        for i, sequence in enumerate(train_data_a):
            guid = "train-%d" % (i)
            text_a = sequence
            text_b = train_data_b[i]
            label = labels[i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples

    @classmethod
    def get_dev_examples(self, dev_data_a, dev_data_b, labels):
        for data in [dev_data_a, dev_data_b]:
            if not isinstance(data, list):
                raise ValueError("`%s` should be a list." % (data))
        if not isinstance(labels, list):
            raise ValueError("`label` should be a list.")
        if len(dev_data_a) != len(dev_data_b) != len(labels):
            raise ValueError("`dev_data_a`, `dev_data_b` and `labels` should have the same length.")
        examples = []
        for i, sequence in enumerate(dev_data_a):
            guid = "dev-%d" % (i)
            text_a = sequence
            text_b = dev_data_b[i]
            label = labels[i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples

    @classmethod
    def get_test_examples(self, test_data_a, test_data_b):
        for data in [test_data_a, test_data_b]:
            if not isinstance(data, list):
                raise ValueError("`%s` should be a list." % (data))
        examples = []
        for i, sequence in enumerate(test_data_a):
            guid = "test-%d" % (i)
            text_a = sequence
            text_b = test_data_b[i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b)
            )
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join(tokens))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id)
    features.append(feature)
    return features


def save_features(features, save_dir=None, input_ids_file='input_ids.npy',
                  input_mask_file='input_mask.npy', segment_ids_file='segment_ids.npy',
                  label_ids_file='label_ids.npy'):
    input_ids = []
    input_mask = []
    segment_ids = []
    label_ids = []
    for feature in features:
        input_ids.append(feature.input_ids)
        input_mask.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)
        label_ids.append(feature.label_id)

    if save_dir is not None:
        np.save(os.path.join(save_dir, input_ids_file), input_ids)
        np.save(os.path.join(save_dir, input_mask_file), input_mask)
        np.save(os.path.join(save_dir, segment_ids_file), segment_ids)
        np.save(os.path.join(save_dir, label_ids_file), label_ids)
    else:
        features_array_dict = dict(input_ids = np.asarray(input_ids),
                                   input_mask = np.asarray(input_mask),
                                   segment_ids = np.asarray(segment_ids),
                                   label_ids = np.asarray(label_ids))
        return features_array_dict


class text_classifier(object):
    def __init__(self, bert_config, pretrain_model_path, batch_size, seq_length, optimizer, num_classes,
                 use_token_type=True, mask=True, max_predictions_per_seq=20, multi_gpu=None):
        if not isinstance(bert_config, BertConfig):
            raise ValueError("`bert_config` must be a instance of `BertConfig`")
        if multi_gpu:
            if not tf.test.is_gpu_available:
                raise ValueError("GPU is not available.")

        self.config = bert_config
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_token_type = use_token_type
        self.max_predictions_per_seq = max_predictions_per_seq
        self.mask = mask
        self.num_classes = num_classes

        if multi_gpu:
            with tf.device('/cpu:0'):
                model = self._build_model(pretrain_model_path)
                model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy)
            parallel_model = multi_gpu_model(model=model, gpus=multi_gpu)
            parallel_model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy)
        else:
            model = self._build_model(pretrain_model_path)
            model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy)

        self.estimator = model
        if multi_gpu:
            self.estimator = parallel_model

    def fit(self, x, y, epochs, shuffle=True, callbacks=None, validation_split=0., validation_data=None,
            class_weight=None, sample_weight=None, **kwargs):
        self.estimator.fit(x=x,
                           y=y,
                           batch_size=self.batch_size,
                           epochs=epochs,
                           shuffle=shuffle,
                           callbacks=callbacks,
                           validation_split=validation_split,
                           validation_data=validation_data,
                           class_weight=class_weight,
                           sample_weight=sample_weight,
                           **kwargs)

    def predict(self, x,
                batch_size=None,
                verbose=0,
                steps=None):
        result = self.estimator.predict(x=x, batch_size=batch_size, verbose=verbose, steps=steps)
        return result

    def _build_model(self, pretrain_model):
        input_ids = Input(shape=(self.seq_length,))
        input_mask = Input(shape=(self.seq_length,))
        inputs = [input_ids, input_mask]
        if self.use_token_type:
            input_token_type_ids = Input(shape=(self.seq_length,))
            inputs.append(input_token_type_ids)

        self.bert = BertModel(self.config,
                              batch_size=self.batch_size,
                              seq_length=self.seq_length,
                              max_predictions_per_seq=self.max_predictions_per_seq,
                              use_token_type=self.use_token_type,
                              mask=self.mask)
        self.bert_encoder = self.bert.get_bert_encoder()
        self.bert_encoder.load_weights(pretrain_model)
        pooled_output = self.bert_encoder(inputs)
        pooled_output = Dropout(self.config.hidden_dropout_prob)(pooled_output)
        pred = Dense(units=self.num_classes,
                     activation='softmax',
                     kernel_initializer=initializers.truncated_normal(stddev=self.config.initializer_range)
                     )(pooled_output)
        model = Model(inputs=inputs, outputs=pred)
        return model



# author: Kris Zhang
import sys
sys.path.append('/mnt/kris/competition')
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical, Sequence, multi_gpu_model
from keras import losses
from bert.modeling import BertConfig, BertModel
from bert.optimization import AdamWeightDecayOpt, StepPreTrainModelCheckpoint


class SampleSequence(Sequence):
    def __init__(self, x, y, batch_size, vocab_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        assert len(x) == 4
        assert len(y) == 2

    def __len__(self):
        return int(np.ceil((len(self.x[0]) / self.batch_size)))

    def __getitem__(self, idx):
        batch_x = [
            self.x[0][idx * self.batch_size:(idx + 1) * self.batch_size],
            self.x[1][idx * self.batch_size:(idx + 1) * self.batch_size],
            self.x[2][idx * self.batch_size:(idx + 1) * self.batch_size],
            self.x[3][idx * self.batch_size:(idx + 1) * self.batch_size]
        ]
        batch_y0 = self.y[0][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y0 = batch_y0.reshape((-1))
        batch_y0 = to_categorical(batch_y0, num_classes=self.vocab_size)
        batch_y0 = batch_y0.reshape((-1, 20, self.vocab_size))
        batch_y = [batch_y0, self.y[1][idx * self.batch_size:(idx + 1) * self.batch_size]]
        return (batch_x, batch_y)


def bert_pretraining(train_data_path, bert_config_file, save_path, batch_size=32, epochs=2, seq_length=128,
                     max_predictions_per_seq=20, lr=5e-5,num_warmup_steps=10000, save_checkpoints_steps=1000,
                     weight_decay_rate=0.01, validation_ratio=0.1, max_num_val=10000, multi_gpu=0, val_batch_size=None):
    tokens_ids = np.load(os.path.join(train_data_path, 'tokens_ids.npy'))
    tokens_mask = np.load(os.path.join(train_data_path, 'tokens_mask.npy'))
    segment_ids = np.load(os.path.join(train_data_path, 'segment_ids.npy'))
    is_random_next = np.load(os.path.join(train_data_path, 'is_random_next.npy'))
    masked_lm_positions = np.load(os.path.join(train_data_path, 'masked_lm_positions.npy'))
    masked_lm_label = np.load(os.path.join(train_data_path, 'masked_lm_labels.npy'))

    num_train_samples = int(len(tokens_ids) * (1 - validation_ratio))
    num_train_steps = int(np.ceil(num_train_samples / batch_size)) * epochs
    print('[INFO] train steps:', num_train_steps)
    print("[INFO] train samples:", len(tokens_ids))

    config = BertConfig.from_json_file(bert_config_file)

    num_val = int(len(tokens_ids) * validation_ratio)
    if num_val > max_num_val:
        validation_ratio = max_num_val / len(tokens_ids)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=12345)
    for train_index, test_index in sss.split(tokens_ids, is_random_next):
        train_tokens_ids, test_tokens_ids = tokens_ids[train_index], tokens_ids[test_index]
        train_tokens_mask, test_tokens_mask = tokens_mask[train_index], tokens_mask[test_index]
        train_segment_ids, test_segment_ids = segment_ids[train_index], segment_ids[test_index]
        train_is_random_next, test_is_random_next = is_random_next[train_index], is_random_next[test_index]
        train_masked_lm_positions, test_masked_lm_positions = masked_lm_positions[train_index], masked_lm_positions[
            test_index]
        train_masked_lm_label, test_masked_lm_label = masked_lm_label[train_index], masked_lm_label[test_index]
        test_masked_lm_label = test_masked_lm_label.reshape((-1))
        test_masked_lm_label = to_categorical(test_masked_lm_label, num_classes=config.vocab_size)
        test_masked_lm_label = test_masked_lm_label.reshape((-1, 20, config.vocab_size))

    print("[INFO] 构建预训练神经网络...")
    adam = AdamWeightDecayOpt(
        lr=lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        weight_decay_rate=weight_decay_rate,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
    )
    bert = BertModel(config,
                     batch_size=batch_size,
                     seq_length=seq_length,
                     max_predictions_per_seq=max_predictions_per_seq,
                     use_token_type=True,
                     embeddings_matrix=None,
                     mask=True
                     )
    if multi_gpu:
        # To avoid OOM errors, this model could have been built on CPU
        with tf.device('/cpu:0'):
            pretraining_model = bert.get_pretraining_model()
            pretraining_model.compile(optimizer=adam, loss=losses.categorical_crossentropy, metrics=['acc'],
                                      loss_weights=[0.5, 0.5])
        parallel_pretraining_model = multi_gpu_model(model=pretraining_model, gpus=multi_gpu)
        parallel_pretraining_model.compile(optimizer=adam, loss=losses.categorical_crossentropy, metrics=['acc'],
                                           loss_weights=[0.5, 0.5])
    else:
        pretraining_model = bert.get_pretraining_model()
        pretraining_model.compile(optimizer=adam, loss=losses.categorical_crossentropy, metrics=['acc'],
                                  loss_weights=[0.5, 0.5])

    print("[INFO] 训练神经网络 for {} epochs".format(epochs))
    train_sample_generator = SampleSequence(
        x=[train_tokens_ids, train_tokens_mask, train_segment_ids, train_masked_lm_positions],
        y=[train_masked_lm_label, train_is_random_next],
        batch_size=batch_size,
        vocab_size=config.vocab_size
    )

    checkpoint_model = None
    if multi_gpu:
        checkpoint_model = pretraining_model
    checkpoint = StepPreTrainModelCheckpoint(
        filepath="%s/best.hdf5" % (save_path),
        monitor='val_acc',
        start_step=num_warmup_steps,
        period=save_checkpoints_steps,
        save_best_only=True,
        verbose=1,
        val_batch_size=val_batch_size,
        model=checkpoint_model  #when use multi_gpu_model, set model to the original model
    )

    estimator = pretraining_model
    if multi_gpu:
        estimator = parallel_pretraining_model

    estimator.fit_generator(
        generator=train_sample_generator,
        epochs=epochs,
        callbacks=[checkpoint],
        shuffle=False,
        validation_data=([test_tokens_ids, test_tokens_mask, test_segment_ids, test_masked_lm_positions],
                         [test_masked_lm_label, test_is_random_next]),
    )

    pretraining_model.load_weights("%s/best.h5" % (save_path))
    bert_model = bert.get_bert_encoder()
    bert_model.save_weights("%s/bert_encoder_model.h5" % (save_path))


if __name__ == "__main__":
    from quora_insincere_questions.const import data_path, model_path
    from bert.const import bert_data_path

    vocab_path = os.path.join(bert_data_path, 'vocab.txt')
    bert_config_file = os.path.join(bert_data_path, 'bert_config.json')
    bert_pretraining(
        train_data_path=data_path,
        bert_config_file=bert_config_file,
        save_path=model_path,
        val_batch_size=512
    )




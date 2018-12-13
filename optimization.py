# author: Kris Zhang
import re
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Optimizer
from keras.legacy import interfaces
import numpy as np
import warnings
from keras.callbacks import Callback
from keras.utils import Sequence
from sklearn.metrics import f1_score, classification_report, accuracy_score

class AdamWeightDecayOpt(Optimizer):
    """Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        num_train_steps: total number train batches in N epoches. np.ceil(train_samples/batch_size)*epoches.
        num_warmup_steps: number steps of warmup
        weight_decay_rate: apply weight decay to weights
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        bias_corrected: boolean. whether or not to use bias_corrected to adam optimizer.
        exclude_from_weight_decay: list of str. weights is excluded from weight deacy.

    # References
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr, num_train_steps, num_warmup_steps, weight_decay_rate=0.0, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-6, bias_corrected=False, exclude_from_weight_decay=None, **kwargs):
        super(AdamWeightDecayOpt, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
        self.epsilon = epsilon
        self.weight_decay_rate = weight_decay_rate
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.bias_corrected = bias_corrected

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = tf.train.polynomial_decay(
            self.lr,
            self.iterations,
            self.num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False
        )

        # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
        # learning rate will be `global_step/num_warmup_steps * init_lr`.
        t = K.cast(self.iterations, K.floatx()) + 1
        warmup_percent_done = K.cast(t / self.num_warmup_steps, dtype=K.floatx())
        warmup_lr = self.lr * warmup_percent_done
        is_warmup = K.cast(t < self.num_warmup_steps, dtype=K.floatx())
        lr = ((1.0 - is_warmup) * lr) + is_warmup * warmup_lr

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.bias_corrected:
                m_t /= 1 - K.pow(self.beta_1, t)
                v_t /= 1 - K.pow(self.beta_2, t)

            update = m_t / (K.sqrt(v_t) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            param_name = self._get_variable_name(p.name)
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * p

            p_t = p - lr * update

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon}
        base_config = super(AdamWeightDecayOpt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


class StepF1ModelCheckpoint(Callback):
    """Save the model after every epoch and scale the learning rate.

    `filepath` can contain named formatting options,
    which will be filled the value of `step` and
    keys in `logs` (passed in `on_batch_end`).

    For example: if `filepath` is `weights.{step:07d}-{val_f1:.3f}-{threshold:.3f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        start_step: the started step of model to eval f1 score.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of steps) between checkpoints.
        binary: classification mode, binary or not.
        threshold_binary: threshold value to recognize 0 or 1 class.
        pred_batch_size: int. Control the batch size of prediction.
        model.  keras model. model is applied to predict.
    """

    def __init__(self, filepath,
                 start_step=10000,verbose=1,
                 save_best_only=True, save_weights_only=True, xlen=1,
                 period=1000, binary=True,
                 threshold_binary=np.arange(0.005,1.0,0.005),
                 threshold_verbose=0,
                 pred_batch_size=512,
                 model=None
                 ):
        super(StepF1ModelCheckpoint, self).__init__()
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.steps_since_last_save = 0
        self.xlen = xlen

        self.binary = binary
        self.threshold_binary = threshold_binary
        self.threshold_verbose = threshold_verbose
        self.pred_batch_size = pred_batch_size
        self.single_gpu_model = model

        self.monitor_op = np.greater
        self.best = -np.Inf
        self.start_step = start_step
        self.cur_step = 0

        if not isinstance(xlen, (int, np.int)):
            raise ValueError("The arg: 'xlen' should be integer.")
        if xlen < 1:
            raise ValueError("The arg: 'xlen' should be >= 1.")

    def set_model(self, model):
        if self.single_gpu_model is not None:
            self.model = self.single_gpu_model
        else:
            self.model = model

    def on_batch_begin(self, batch, logs=None):
        self.cur_step += 1

    def on_batch_end(self, batch, logs=None):
        if self.cur_step > self.start_step:
            self.steps_since_last_save += 1
            if self.steps_since_last_save >= self.period:
                self.steps_since_last_save = 0

                if self.xlen == 1:
                    x = self.validation_data[0]
                    y = self.validation_data[1]
                else:
                    x = self.validation_data[0: self.xlen]
                    y = self.validation_data[self.xlen]

                if self.binary:
                    val_prob = np.asarray(self.model.predict(x, batch_size=self.pred_batch_size))
                    if np.shape(val_prob)[1] == 1:
                        val_prob = val_prob[:, 0]
                        val_true = y
                    else:
                        val_prob = val_prob[:, 1]
                        val_true = np.argmax(y, axis=1)

                    val_f1s = []
                    for threshold in self.threshold_binary:
                        val_pred = np.asarray(val_prob > threshold, dtype=np.int)
                        val_f1 = f1_score(val_true, val_pred)
                        val_f1s.append(val_f1)
                        if self.threshold_verbose > 0:
                            print("\n-------  threshold_value:{:.2f}  -------".format(threshold))
                            print(classification_report(val_true, val_pred))
                    val_f1s = np.array(val_f1s)
                    val_f1 = val_f1s.max()
                    threshold_value = self.threshold_binary[val_f1s.argmax()]

                else:
                    val_pred = np.argmax(np.asarray(self.model.predict(x)), axis=1)
                    val_true = np.argmax(y, axis=1)
                    val_f1 = f1_score(val_true, val_pred, average='macro')
                    threshold_value = 0.

                filepath = self.filepath.format(step=self.cur_step, val_f1=val_f1, threshold=threshold_value)
                if self.save_best_only:
                    current = val_f1
                    if current is None:
                        warnings.warn('Can save best model only with val_f1 available, '
                                      'skipping.', RuntimeWarning)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                if not self.binary:
                                    print('\nStep %07d: val_f1 improved from %0.5f to %0.5f,'
                                          ' saving model to %s'
                                          % (self.cur_step, self.best, current, filepath))
                                else:
                                    print('\nStep %07d: val_f1 improved from %0.5f to %0.5f at threshold_value = %0.3f,'
                                          ' saving model to %s'
                                          % (self.cur_step, self.best, current, threshold_value, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('\nStep %07d: val_f1 = %0.5f did not improve from %0.5f' %
                                      (self.cur_step, current ,self.best))
                else:
                    if self.verbose > 0:
                        print('\nStep %07d: saving model to %s, threshold_value = %0.2f'
                              % (self.cur_step, filepath, threshold_value))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)


class StepPreTrainModelCheckpoint(Callback):
    """Save the model after every epoch.

        `filepath` can contain named formatting options,
        which will be filled the value of `epoch` and
        keys in `logs` (passed in `on_epoch_end`).

        For example: if `filepath` is `weights.{batch:02d}-{val_loss:.2f}.hdf5`,
        then the model checkpoints will be saved with the epoch number and
        the validation loss in the filename.

        # Arguments
            filepath: string, path to save the model file.
            monitor: quantity to monitor.
            verbose: verbosity mode, 0 or 1.
            save_best_only: if `save_best_only=True`,
                the latest best model according to
                the quantity monitored will not be overwritten.
            mode: one of {auto, min, max}.
                If `save_best_only=True`, the decision
                to overwrite the current save file is made
                based on either the maximization or the
                minimization of the monitored quantity. For `val_acc`,
                this should be `max`, for `val_loss` this should
                be `min`, etc. In `auto` mode, the direction is
                automatically inferred from the name of the monitored quantity.
            save_weights_only: if True, then only the model's weights will be
                saved (`model.save_weights(filepath)`), else the full model
                is saved (`model.save(filepath)`).
            period: Interval (number of epochs) between checkpoints.
            start_step: steps to start checkpoint.
            val_batch_size: batch size used in validation.
            model: model used in validation.
        """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, start_step=10000, val_batch_size=None, model=None):
        super(StepPreTrainModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.start_step = start_step
        self.steps_since_last_save = 0
        self.cur_step = 0
        self.val_batch_size = val_batch_size
        self.single_gpu_model = model

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def set_model(self, model):
        if self.single_gpu_model is not None:
            self.model = self.single_gpu_model
        else:
            self.model = model

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.cur_step += 1
        if self.cur_step > self.start_step:
            self.steps_since_last_save += 1
            if self.steps_since_last_save >= self.period:
                self.steps_since_last_save = 0
                filepath = self.filepath.format(step=self.cur_step, **logs)
                if self.save_best_only:
                    if self.validation_data is None:
                        raise ValueError("`validation_data is None.`")
                    val_data = self.validation_data
                    val_x = val_data[0: 4]
                    val_y = val_data[4: 6]
                    val_sample_weights = val_data[6:8]
                    if self.val_batch_size is not None:
                        batch_size = self.val_batch_size
                    else:
                        batch_size = logs.get('size')
                    val_outs = self.model.evaluate(
                            val_x, val_y,
                            batch_size=batch_size,
                            sample_weight=val_sample_weights,
                            verbose=0
                    )
                    val_lm_model_acc  = val_outs[3]
                    val_sentence_model_acc = val_outs[4]
                    print("\nstep %07d: cur_lm_acc is %0.5f, cur_is_random_nex_acc is %0.5f"
                          % (self.cur_step, val_lm_model_acc, val_sentence_model_acc))
                    current = (val_lm_model_acc + val_sentence_model_acc) / 2
                    # current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn('Can save best model only with %s available, '
                                      'skipping.' % (self.monitor), RuntimeWarning)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nStep %07d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s'
                                      % (self.cur_step, self.monitor, self.best,
                                         current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('\nStep %07d: %s did not improve from %0.5f, current is %0.5f' %
                                      (self.cur_step, self.monitor, self.best, current))
                else:
                    if self.verbose > 0:
                        print('\nStep %07d: saving model to %s' % (self.cur_step, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)

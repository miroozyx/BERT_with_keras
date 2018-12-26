# author: Kris Zhang
import warnings
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, classification_report


class StepPreTrainModelCheckpoint(Callback):
    """Save the model after specified interval of steps with 'val_acc' monitor.

        `filepath` can contain named formatting options,
        which will be filled the value of `step` and
        keys in `logs` (passed in `on_epoch_end`).

        For example: if `filepath` is `weights.{step:02d}-{val_loss:.2f}.hdf5`,
        then the model checkpoints will be saved with the epoch number and
        the validation loss in the filename.

        # Arguments
            filepath: string, path to save the model file.
            verbose: verbosity mode, 0 or 1.
            save_best_only: if `save_best_only=True`,
                the latest best model according to
                the quantity monitored will not be overwritten.
            save_weights_only: if True, then only the model's weights will be
                saved (`model.save_weights(filepath)`), else the full model
                is saved (`model.save(filepath)`).
            period: Interval (number of epochs) between checkpoints.
            start_step: steps to start checkpoint.
            val_batch_size: batch size used in validation.
            model: model used in validation.
        """

    def __init__(self, filepath,verbose=0,
                 save_best_only=False, save_weights_only=False,
                 period=1, start_step=10000, val_batch_size=None, model=None):
        super(StepPreTrainModelCheckpoint, self).__init__()
        self.monitor = 'val_acc'
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

        self.monitor_op = np.greater
        self.best = -np.Inf

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
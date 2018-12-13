# author: Kris Zhang
import warnings
import numpy as np
from keras.callbacks import Callback


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

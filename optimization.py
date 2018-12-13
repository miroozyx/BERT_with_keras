# author: Kris Zhang
import re
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Optimizer
from keras.legacy import interfaces


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

# author: Kris Zhang
import six
import json
import copy
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer, Input, Dropout, Lambda, BatchNormalization, Dense, Embedding
from keras import activations, initializers, regularizers, constraints
from keras.models import Model
from keras.legacy import interfaces


class EmbeddingLookup(Embedding):
    """
    EmbeddingLookup is slightly different from Embedding Class. you need input the mask by yourself followed
    by the embedding input.
    The outputs are two tensors: the first one: is embedding matrix. the last one is embeddings table of all indexes.
    """
    @interfaces.legacy_embedding_support
    def __init__(self, input_dim, output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask=True,
                 input_length=None,
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(EmbeddingLookup, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=False,
            input_length=input_length,
            **kwargs
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask = mask
        self.supports_masking = mask
        self.input_length = input_length

    def compute_mask(self, inputs, mask=None):
        if self.mask:
            assert len(inputs) == 2
        else:
            assert len(inputs) == 1

        if not self.mask:
            return [None, None]
        output_mask = inputs[1]
        return [output_mask, None]

    def compute_output_shape(self, input_shape):
        if self.mask:
            output_shape0 = super(EmbeddingLookup, self).compute_output_shape(input_shape[0])
        else:
            output_shape0 = super(EmbeddingLookup, self).compute_output_shape(input_shape)
        output_shape1 = (self.input_dim, self.output_dim)
        return [output_shape0, output_shape1]

    def call(self, inputs, **kwargs):
        if not self.mask:
            if K.dtype(inputs) != 'int32':
                inputs = K.cast(inputs, 'int32')
            out0 = K.gather(self.embeddings, inputs)
            out1 = tf.convert_to_tensor(self.embeddings)
            return [out0, out1]
        else:
            inputs = inputs[0]
            if K.dtype(inputs) != 'int32':
                inputs = K.cast(inputs, 'int32')
            out0 = K.gather(self.embeddings, inputs)
            out1 = tf.convert_to_tensor(self.embeddings)
            return [out0, out1]


def gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


class TensorAdd(Layer):
    '''ops: A+B'''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(TensorAdd, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            mask = mask[0]
        return mask

    def call(self, inputs, mask=None):
        assert len(inputs) == 2
        x1, x2 = inputs
        assert K.int_shape(x1) == K.int_shape(x2)
        output = x1 + x2
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Embedding_Postprocessor(Layer):
    def __init__(self, use_token_type=False,
                 # token_type_ids=None,
                 token_type_vocab_size=16,
                 use_position_embeddings=True,
                 initializer_range=0.02,
                 max_position_embeddings=512,
                 **kwargs):
        self.use_token_type = use_token_type
        # self.token_type_ids = token_type_ids
        self.token_type_vocab_size = token_type_vocab_size
        self.use_position_embeddings = use_position_embeddings
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.supports_masking = True
        super(Embedding_Postprocessor, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_token_type:
            _, seq_length, input_width = input_shape[0]
            self.token_type_table = self.add_weight(
                shape=(self.token_type_vocab_size, input_width),
                initializer=initializers.truncated_normal(stddev=self.initializer_range),
                name='token_type_embeddings'
            )
        else:
            _, seq_length, input_width = input_shape

        if self.use_position_embeddings:
            assert seq_length <= self.max_position_embeddings
            self.full_position_embeddings = self.add_weight(
                shape=(self.max_position_embeddings, input_width),
                initializer=initializers.truncated_normal(stddev=self.initializer_range),
                name='position_embeddings'
            )
        super(Embedding_Postprocessor, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if self.use_token_type:
            mask = mask[0]
        return mask

    def call(self, inputs, mask=None):
        if self.use_token_type:
            assert  len(inputs) == 2, "`token_type_ids` must be specified if `use_token_type` is True."
            output = inputs[0]
            _, seq_length, input_width = K.int_shape(output)
            # print(inputs)
            assert seq_length == K.int_shape(inputs[1])[1], "width of `token_type_ids` must be equal to `seq_length`"
            token_type_ids = inputs[1]
            # assert K.int_shape(token_type_ids)[1] <= self.token_type_vocab_size
            flat_token_type_ids = K.reshape(token_type_ids, [-1])
            flat_token_type_ids = K.cast(flat_token_type_ids, dtype='int32')
            token_type_one_hot_ids = K.one_hot(flat_token_type_ids, num_classes=self.token_type_vocab_size)
            token_type_embeddings = K.dot(token_type_one_hot_ids, self.token_type_table)
            token_type_embeddings = K.reshape(token_type_embeddings, shape=[-1, seq_length, input_width])
            # print(token_type_embeddings)
            output += token_type_embeddings
        else:
            output = inputs
            seq_length = K.int_shape(inputs)[1]

        if self.use_position_embeddings:
            position_embeddings = K.slice(self.full_position_embeddings, [0, 0], [seq_length, -1])
            output += position_embeddings

        return output

    def compute_output_shape(self, input_shape):
        if self.use_token_type:
            output_shape = input_shape[0]
        else:
            output_shape = input_shape
        return output_shape


class MultiHeadAttentionLayer(Layer):
    def __init__(self,
                 num_attention_heads=8,
                 size_per_head=64,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 **kwargs):
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.query_act = activations.get(query_act)
        self.key_act = activations.get(key_act)
        self.value_act = activations.get(value_act)
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.supports_masking = True
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert  len(input_shape) == 2
        self.W_query = self.add_weight(shape=(input_shape[0][-1], self.num_attention_heads*self.size_per_head),
                                       name='Wq',
                                       initializer=initializers.truncated_normal(stddev=self.initializer_range))
        self.bias_query = self.add_weight(shape=(self.num_attention_heads*self.size_per_head,),
                                          name='bq',
                                          initializer=initializers.get('zeros'))
        self.W_key = self.add_weight(shape=(input_shape[1][-1], self.num_attention_heads*self.size_per_head),
                                     name='Wk',
                                     initializer=initializers.truncated_normal(stddev=self.initializer_range))
        self.bias_key = self.add_weight(shape=(self.num_attention_heads*self.size_per_head,),
                                        name='bk',
                                        initializer=initializers.get('zeros'))
        self.W_value= self.add_weight(shape=(input_shape[1][-1], self.num_attention_heads*self.size_per_head),
                                      name='Wv',
                                      initializer=initializers.truncated_normal(stddev=self.initializer_range))
        self.bias_value = self.add_weight(shape=(self.num_attention_heads*self.size_per_head,),
                                        name='bv',
                                        initializer=initializers.get('zeros'))
        super(MultiHeadAttentionLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            mask = mask[0]
        return mask

    def call(self, inputs, mask=None):
        assert len(inputs) == 2
        query = K.bias_add(K.dot(inputs[0], self.W_query), self.bias_query)
        if self.query_act is not None:
            query = self.query_act(query)
        key = K.bias_add(K.dot(inputs[1], self.W_key), self.bias_key)
        if self.key_act is not None:
            key = self.key_act(key)
        value = K.bias_add(K.dot(inputs[1], self.W_value), self.bias_value)
        if self.value_act is not None:
            value = self.value_act(value)

        query = K.reshape(query, shape=(-1, K.int_shape(inputs[0])[1], self.num_attention_heads, self.size_per_head))
        query = K.permute_dimensions(query, pattern=(0,2,1,3))
        key = K.reshape(key, shape=(-1, K.int_shape(inputs[1])[1], self.num_attention_heads, self.size_per_head))
        key = K.permute_dimensions(key, pattern=(0,2,1,3))
        value = K.reshape(value, shape=(-1, K.int_shape(inputs[1])[1], self.num_attention_heads, self.size_per_head))
        value = K.permute_dimensions(value, pattern=(0,2,1,3))

        attention_scores = K.batch_dot(query, key, axes=(3,3))
        attention_scores /= np.sqrt(self.size_per_head)

        if mask is not None and mask != [None, None]:
            mask_q, mask_k = mask
            mask_q = K.cast(mask_q, K.floatx())
            mask_k = K.cast(mask_k, K.floatx())
            mask_q = K.expand_dims(mask_q)
            mask_k = K.expand_dims(mask_k)
            attention_mask = K.batch_dot(mask_q, mask_k, axes=(-1,-1))
            attention_mask = K.expand_dims(attention_mask, axis=1)
            adder = (1 - attention_mask) * (-10000.0)
            attention_scores += adder

        attention_probs = K.softmax(attention_scores, axis=-1)
        attention_probs = K.dropout(attention_probs, self.attention_probs_dropout_prob)

        context = K.batch_dot(attention_probs, value, axes=(3,2))
        context = K.permute_dimensions(context, pattern=(0,2,1,3))
        context = K.reshape(context, shape=(-1, K.int_shape(inputs[0])[1], self.num_attention_heads*self.size_per_head))

        return context

    def compute_output_shape(self, input_shape):
        return input_shape[0][0],input_shape[0][1], self.num_attention_heads*self.size_per_head


class BertConfig(object):
    """Configuration for 'BerModel'
    Args:
        vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
        hidden_size: Size of the encoder layers and the pooler layer.
        num_hidden_layers: Number of hidden layers in the Transformer encoder.
        num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
        intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
        hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
        max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
        initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
    """
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob=hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size=type_vocab_size
        self.initializer_range=initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        with tf.gfile.GFile(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
    """Build bert model.

    Example usage:
    ```python
    # Already been converted into token ids
    input_ids = np.array([[31, 51, 99], [15, 5, 0]])
    input_mask = np.array([[1, 1, 1], [1, 1, 0]])
    token_type_ids = [[0, 0, 1], [0, 2, 0]]
    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
    bert = BertModel(config=config, batch_size=32, seq_length=128,use_token_type=True, mask=True)
    model = bert.get_bert_model
    model.compile(...)
    model.fit(x=[input_ids, input_mask, token_type_ids],
              y=...)
    ```

    """
    def __init__(self,
                 config,
                 batch_size,
                 seq_length,
                 max_predictions_per_seq=20,
                 use_token_type=False,
                 embeddings_matrix=None,
                 mask=False):
        """ Constructor for BertModel

        Args:
            config: instance of BertConfig.
            batch_size: Integer. Number of samples per gradient update.
            seq_length: Integer. The maximum total input sequence length after tokenization.
                Sequences longer than this will be truncated, and sequences shorter
                than this will be padded. Must match data generation.
            max_predictions_pre_seq: Integer. Maximum number of masked LM predictions per sequence.
            use_token_type: Boolean. Whether of not use segmented id in model.
            embeddings_matrix: initial embeddings weights.
            mask: boolean. When mask is True. input of mask must be added to model inputs.
        """
        if not isinstance(config, BertConfig):
            raise ValueError("`config` must be a instance of `BertConfig`.")
        config = copy.deepcopy(config)
        self.config = config
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.max_predictions_per_seq = max_predictions_per_seq
        self.embedding_matrix = embeddings_matrix
        self.inputs =[]
        input_ids = Input(shape=(seq_length,), dtype='int32', name='input_ids')
        self.inputs.append(input_ids)
        input_mask = Input(shape=(seq_length,), dtype='int32', name='input_mask')
        self.inputs.append(input_mask)
        # perform embeddings layer
        if embeddings_matrix is not None:
            embeddings_layer = EmbeddingLookup(
                input_dim=embeddings_matrix.shape[0],
                output_dim=embeddings_matrix.shape[1],
                weights=[embeddings_matrix],
                mask=mask,
                input_length=seq_length,
                trainable=True
            )
        else:
            embeddings_layer = EmbeddingLookup(
                input_dim=config.vocab_size,
                output_dim=config.hidden_size,
                mask=mask,
                input_length=seq_length,
                trainable=True,
                embeddings_initializer=initializers.TruncatedNormal(stddev=config.initializer_range),
            )
        self.embedding_output, self.embedding_table = embeddings_layer([input_ids, input_mask])

        # Generally, embeddings_matrix came from pre-trained model of Word2vec, Glove, etc. embedding_size usually
        # to be 300, so we need to transform the dim to what we need. for example: 300->768
        if embeddings_matrix is not None and embeddings_matrix.shape[1] < config.hidden_size:
            self.embedding_output = Dense(
                units=config.hidden_size,
                kernel_initializer=initializers.TruncatedNormal(stddev=config.initializer_range),
            )(self.embedding_output)

        # add postional embeddings and token type embeddings, then layer normalize and perform dropout
        if use_token_type:
            input_token_type_ids = Input(shape=(seq_length,),dtype='int32', name='input_segment_ids')
            self.inputs.append(input_token_type_ids)
            self.embedding_output = Embedding_Postprocessor(
                use_token_type=use_token_type,
                token_type_vocab_size=config.type_vocab_size,
                use_position_embeddings=True,
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings
            )([self.embedding_output, input_token_type_ids])
        else:
            self.embedding_output = Embedding_Postprocessor(
                use_token_type=False,
                token_type_vocab_size=config.type_vocab_size,
                use_position_embeddings=True,
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
            )(self.embedding_output)


        self.embedding_output = BatchNormalization(name="layer_norm_embeddings")(self.embedding_output)
        self.embedding_output = Dropout(config.hidden_dropout_prob)(self.embedding_output)

        # Run the stacked transformer.
        attention_head_size = int(config.hidden_size / config.num_attention_heads)
        prev_output = self.embedding_output

        all_layer_outputs = []
        for layer_idx in range(config.num_hidden_layers):
            layer_input = prev_output
            # self-attention
            attention_ouput = MultiHeadAttentionLayer(
                num_attention_heads=config.num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=config.attention_dropout_prob,
                initializer_range=config.initializer_range
            )([layer_input, layer_input])
            attention_ouput = Dense(
                units=config.hidden_size,
                kernel_initializer=initializers.TruncatedNormal(stddev=config.initializer_range)
            )(attention_ouput)
            attention_ouput = Dropout(config.hidden_dropout_prob)(attention_ouput)
            attention_ouput = TensorAdd()([layer_input, attention_ouput])
            attention_ouput = BatchNormalization(name="layer_norm" + "_" + str(layer_idx))(attention_ouput)

            # FFN
            intermediate_ouput = Dense(
                units=config.intermediate_size,
                activation=get_activation(config.hidden_act),
                kernel_initializer=initializers.TruncatedNormal(stddev=config.initializer_range),
            )(attention_ouput)
            layer_output = Dense(
                units=config.hidden_size,
                kernel_initializer=initializers.TruncatedNormal(stddev=config.initializer_range)
            )(intermediate_ouput)
            layer_output = Dropout(config.hidden_dropout_prob)(layer_output)
            layer_output = TensorAdd()([attention_ouput, layer_output])
            layer_output = BatchNormalization(name="layer_norm" + "-" + str(layer_idx))(layer_output)
            prev_output = layer_output
            all_layer_outputs.append(layer_output)

        self.all_encoder_layers = all_layer_outputs

        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.sequence_output = self.all_encoder_layers[-1]
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = Lambda(
            lambda x: K.squeeze(x[:,0:1,:], axis=1),
            output_shape=lambda x: (x[0],x[2]),
        )(self.sequence_output)

        self.pooled_output = Dense(
            units=config.hidden_size,
            activation='tanh',
            kernel_initializer=initializers.TruncatedNormal(stddev=config.initializer_range)
        )(first_token_tensor)
        # construct text encoder model
        self.bert_model = Model(inputs=self.inputs, outputs=self.pooled_output, name='bert_model')

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.
        float Tensor of shape [batch_size, seq_length, hidden_size]
        """
        return self.sequence_output

    def get_all_encoder_layer(self):
        """Gets all encoder layer output.
        list of float Tensor of shape [batch_size, seq_length, hidden_size]"""
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of embedding layer.
        float Tensor of shape [batch_size, seq_length, hidden_size"""
        return self.embedding_output

    def get_embedding_table(self):
        """Gets embeddings of vocabulary.
        float Tensor of shape [vocab_size, hidden_size]
        """
        return self.embedding_table

    def get_bert_encoder(self):
        """Gets bert encoder, which can encoder a sequence to a vector"""
        return self.bert_model

    def get_lm_model(self):
        """construct language model for pretraining"""
        config = self.config
        positions_input = Input(shape=(self.max_predictions_per_seq,), dtype='int32', name='masked_lm_positions')
        cur_inputs = self.inputs + [positions_input]

        sequence_output = Lambda(
            function=lambda x: gather_indexes(x[0], x[1]),
            output_shape=lambda x: (x[0][0], x[1][1], x[0][2])
        )([self.sequence_output, positions_input])

        sequence_output = Dense(
            units=config.hidden_size,
            activation=get_activation(config.hidden_act),
            kernel_initializer=initializers.truncated_normal(stddev=config.initializer_range),
        )(sequence_output)
        sequence_output = BatchNormalization(name='layer_norm_lm')(sequence_output)

        sequence_att = Lambda(
            function=lambda x: K.dot(x[0], K.permute_dimensions(x[1], pattern=(1,0))),
            output_shape=lambda x: (x[0][0], x[0][1] ,x[1][0]),
        )([sequence_output, self.embedding_table])

        class AddBiasSoftmax(Layer):
            def __init__(self, **kwargs):
                self.supports_masking=True
                super(AddBiasSoftmax, self).__init__(**kwargs)

            def build(self, input_shape):
                self.bias = self.add_weight(shape=(input_shape[-1],),
                                                  name='output_bias',
                                                  initializer=initializers.get('zeros'))
                super(AddBiasSoftmax, self).build(input_shape)

            def call(self, inputs, **kwargs):
                output = K.bias_add(inputs, self.bias)
                output = K.softmax(output, axis=-1)
                return output

            def compute_output_shape(self, input_shape):
                return input_shape

        sequence_softmax = AddBiasSoftmax()(sequence_att)

        self.lm_model = Model(inputs=cur_inputs, outputs=sequence_softmax, name='lm_model')
        return self.lm_model

    def get_next_sentence_model(self):
        """construct next sentence model for pretraining"""
        pooled_output = self.bert_model(self.inputs)
        pred = Dense(units=2,
                     activation='softmax',
                     kernel_initializer=initializers.truncated_normal(stddev=self.config.initializer_range)
                     )(pooled_output)
        self.next_sentence_model = Model(inputs=self.inputs, outputs=pred, name='next_sentence_model')
        return self.next_sentence_model

    def get_pretraining_model(self):
        """construct model for pretraining"""
        positions_maxlen = self.max_predictions_per_seq
        positions_input = Input(shape=(positions_maxlen,), dtype='int32',name='masked_lm')
        lm_inputs = self.inputs + [positions_input]
        next_sentence_encoder = self.get_next_sentence_model()
        lm_encoder = self.get_lm_model()
        lm_pred = lm_encoder(lm_inputs)
        next_sentence_pred = next_sentence_encoder(self.inputs)
        predtraining_model = Model(inputs=lm_inputs, outputs=[lm_pred, next_sentence_pred])
        return predtraining_model

    def get_classifer_model(self, num_classes):
        """construct model for classify """
        bert_encoder = Dropout(self.config.hidden_dropout_prob)(self.pooled_output)
        pred = Dense(units=num_classes,
                     activation='softmax',
                     kernel_initializer=initializers.truncated_normal(stddev=self.config.initializer_range),
                     )(bert_encoder)
        self.classifer_model = Model(inputs=self.inputs, outputs=pred)
        return self.classifer_model


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch.
    sequence_tensor shape: [batch_size, seq_length, width]
    positions shape: [batch_size, positions_maxlen]
    """
    positions = K.cast(positions, dtype='int32')
    sequence_shape = K.int_shape(sequence_tensor)
    seq_length = sequence_shape[1]
    positions_length = K.int_shape(positions)[1]

    flat_positions = K.reshape(positions,[-1])
    flat_positions = K.one_hot(flat_positions, seq_length)
    flat_positions = K.reshape(flat_positions, [-1, positions_length, seq_length])
    output_tensor = K.batch_dot(flat_positions, sequence_tensor, axes=(2,1))
    return output_tensor

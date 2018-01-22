from .hmlstm_cell import HMLSTMCell, HMLSTMState
from .multi_hmlstm_cell import MultiHMLSTMCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
import numpy as np


class HMLSTMNetwork(object):
    def __init__(self,
                 max_seq_length,
                 input_size=1,
                 output_size=1,
                 num_layers=3,
                 hidden_state_sizes=50,
                 num_hidden_layers=2,
                 out_hidden_size=100,
                 embed_size=100,
                 task='regression',
                 layer_norm=False,
                 recursion_depth=2,
                 batch_size=128,
                 variable_path='./vars',
                 residual=False,
                 last_layer_residual=False,
                 lr_start=0.01,
                 lr_end=0.00001,
                 lr_steps=1000,
                 grad_clip=1.0,
                 iterator=None
                 ):
        """
        HMLSTMNetwork is a class representing hierarchical multiscale
        long short-term memory network.

        params:
        ---
        input_size: integer, the size of an input at one timestep
        output_size: integer, the size of an output at one timestep
        num_layers: integer, the number of layers in the hmlstm
        hidden_state_size: integer or list of integers. If it is an integer,
            it is the size of the hidden state for each layer of the hmlstm.
            If it is a list, it must have length equal to the number of layers,
            and each integer of the list is the size of the hidden state for
            the layer correspodning to its index.
        num_hidden_layers: the number of hidden layers in the output network
        out_hidden_size: integer, the size of the hidden layers in the
            output network.
        embed_size: integer, the size of the embedding in the output network.
        task: string, one of 'regression' and 'classification'.
        """

        self._last_layer_residual = last_layer_residual
        self._residual = residual
        self._max_seq_length = max_seq_length
        self._out_hidden_size = out_hidden_size
        self._embed_size = embed_size
        self._num_layers = num_layers
        self._num_hidden_layers = num_hidden_layers
        self._input_size = input_size
        self._session = None
        self._graph = None
        self._task = task
        self._output_size = output_size
        self._layer_norm = layer_norm
        self._recursion_depth = recursion_depth
        self._batch_size = batch_size
        self._variable_path = variable_path
        self._grad_clip=grad_clip
        self._last_states = None

        if type(hidden_state_sizes) is list \
                and len(hidden_state_sizes) != num_layers:
            raise ValueError('The number of hidden states provided must be the'
                             + ' same as the nubmer of layers.')

        if type(hidden_state_sizes) == int:
            self._hidden_state_sizes = [hidden_state_sizes] * self._num_layers
        else:
            self._hidden_state_sizes = hidden_state_sizes

        if task == 'classification':
            self._loss_function = tf.nn.softmax_cross_entropy_with_logits
        elif task == 'regression':
            self._loss_function = lambda logits, labels: tf.square((logits - labels))

        if iterator is None:
            batch_in_shape = (None, None, self._input_size)
            batch_out_shape = (None, None, self._output_size)
            self.batch_in = tf.placeholder(
                tf.float32, shape=batch_in_shape, name='batch_in')
            self.batch_out = tf.placeholder(
                tf.float32, shape=batch_out_shape, name='batch_out')
            self.lengths = tf.placeholder(tf.int32, shape=(None,))
            self._initial_states = tf.placeholder(tf.float32,
                                                  (batch_size, (sum(self._hidden_state_sizes) * 2) + self._num_layers),
                                                  name='initial_states')
        else:
            self.batch_in, self.batch_out, self.lengths, self._initial_states = iterator.get_next()
            self.batch_in = tf.transpose(self.batch_in, (1, 0, 2))
            self.batch_out = tf.transpose(self.batch_out, (1, 0, 2))
            self._iterator = iterator

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        learning_rate = tf.train.polynomial_decay(lr_start, self.global_step, lr_steps, lr_end, power=0.5)

        self._optimizer = tf.train.AdamOptimizer(learning_rate)

        # self._optimizer = tf.train.AdamOptimizer(1e-3)
        self._initialize_output_variables()
        self._initialize_gate_variables()
        self._initialize_embedding_variables()

        self.init()

    def _initialize_gate_variables(self):
        with vs.variable_scope('gates_vars'):
            for l in range(self._num_layers):
                vs.get_variable(
                    'gate_%s' % l, [sum(self._hidden_state_sizes), 1],
                    dtype=tf.float32)

    def _initialize_embedding_variables(self):
        with vs.variable_scope('embedding_vars'):
            embed_shape = [sum(self._hidden_state_sizes), self._embed_size]
            vs.get_variable('embed_weights', embed_shape, dtype=tf.float32)

    def _initialize_output_variables(self):
        with vs.variable_scope('output_module_vars'):
            for i in range(0, self._num_hidden_layers):
                b_var_name = 'b' + str(i + 1)
                b_size = self._out_hidden_size if i != self._num_hidden_layers - 1 else self._output_size
                vs.get_variable(b_var_name, [1, b_size], dtype=tf.float32)

                w_var_name = 'w' + str(i + 1)
                w_in_size = self._embed_size if i == 0 else self._out_hidden_size
                w_out_size = self._output_size if i == self._num_hidden_layers - 1 else self._out_hidden_size
                vs.get_variable(w_var_name, [w_in_size, w_out_size], dtype=tf.float32)

    def load_variables(self):
        saver = tf.train.Saver()
        print('loading variables...')
        saver.restore(self._session, self._variable_path)

    def save_variables(self, path='./hmlstm_ckpt'):
        saver = tf.train.Saver()
        print('saving variables...')
        saver.save(self._session, path)

    def gate_input(self, hidden_states):
        '''
        gate the incoming hidden states
        hidden_states: [B, sum(h_l)]

        gated_input: [B, sum(h_l)]
        '''
        with vs.variable_scope('gates_vars', reuse=True):
            gates = []  # [[B, 1] for l in range(L)]
            for l in range(self._num_layers):
                weights = vs.get_variable('gate_%d' % l, dtype=tf.float32)
                gates.append(tf.sigmoid(tf.matmul(hidden_states, weights)))

            split = array_ops.split(
                value=hidden_states,
                num_or_size_splits=self._hidden_state_sizes,
                axis=1)

            gated_list = []  # [[B, h_l] for l in range(L)]
            for gate, hidden_state in zip(gates, split):
                gated_list.append(tf.multiply(gate, hidden_state))

            gated_input = tf.concat(gated_list, axis=1)  # [B, sum(h_l)]
        return gated_input

    def embed_input(self, gated_input):
        '''
        gated_input: [B, sum(h_l)]

        embedding: [B, E], i.e. [B, embed_size]
        '''
        with vs.variable_scope('embedding_vars', reuse=True):
            embed_weights = vs.get_variable('embed_weights', dtype=tf.float32)

            prod = tf.matmul(gated_input, embed_weights)
            embedding = tf.nn.relu(prod)

        return embedding

    def output_module(self, embedding, outcome):
        '''
        embedding: [B, E]
        outcome: [B, output_size]

        loss: [B, output_size] or [B, 1]
        prediction: [B, output_size]
        '''
        with vs.variable_scope('output_module_vars', reuse=True):
            # feed forward network
            _layers = []
            for i in range(0, self._num_hidden_layers):
                inputs = embedding if i == 0 else _layers[i - 1]
                w = vs.get_variable('w' + str(i + 1))
                b = vs.get_variable('b' + str(i + 1))
                _l = tf.matmul(inputs, w) + b
                if i != self._num_hidden_layers - 1:
                    _l = tf.nn.tanh(_l)
                _layers.append(_l)

            prediction = tf.identity(_layers[self._num_hidden_layers - 1], name='prediction')

            # first layer

            # the loss function used below
            # softmax_cross_entropy_with_logits

            loss_args = {'logits': prediction, 'labels': outcome}
            loss = self._loss_function(**loss_args)

            if self._task == 'classification':
                # due to nature of classification loss function
                loss = tf.expand_dims(loss, -1)

        return loss, prediction

    def create_multicell(self, batch_size, reuse):
        def hmlstm_cell(layer):
            residual = self._residual
            if layer == 0:
                h_below_size = self._input_size
            else:
                h_below_size = self._hidden_state_sizes[layer - 1]

            if layer == self._num_layers - 1:
                # doesn't matter, all zeros, but for convenience with summing
                # so the sum of ha sizes is just sum of hidden states
                h_above_size = self._hidden_state_sizes[0]
                residual = self._last_layer_residual
            else:
                h_above_size = self._hidden_state_sizes[layer + 1]

            return HMLSTMCell(self._hidden_state_sizes[layer], batch_size,
                              h_below_size, h_above_size, reuse, layer_norm=self._layer_norm,
                              recursion_depth=self._recursion_depth, residual=residual)

        hmlstm = MultiHMLSTMCell(
            [hmlstm_cell(l) for l in range(self._num_layers)], reuse)

        return hmlstm

    def split_out_cell_states(self, accum):
        '''
        accum: [B, H], i.e. [B, sum(h_l) * 2 + num_layers]


        cell_states: a list of ([B, h_l], [B, h_l], [B, 1]), with length L
        '''
        splits = []
        for size in self._hidden_state_sizes:
            splits += [size, size, 1]

        split_states = array_ops.split(value=accum,
                                       num_or_size_splits=splits, axis=1)

        cell_states = []
        for l in range(self._num_layers):
            c = split_states[(l * 3)]
            h = split_states[(l * 3) + 1]
            z = split_states[(l * 3) + 2]
            cell_states.append(HMLSTMState(c=c, h=h, z=z))

        return cell_states

    def get_h_aboves(self, hidden_states, batch_size, hmlstm):
        '''
        hidden_states: [[B, h_l] for l in range(L)]

        h_aboves: [B, sum(ha_l)], ha denotes h_above
        '''
        concated_hs = array_ops.concat(hidden_states[1:], axis=1)

        h_above_for_last_layer = tf.zeros(
            [batch_size, hmlstm._cells[-1]._h_above_size], dtype=tf.float32)

        h_aboves = array_ops.concat(
            [concated_hs, h_above_for_last_layer], axis=1)

        return h_aboves

    def network(self, reuse):
        batch_size = tf.shape(self.batch_in)[1]
        hmlstm = self.create_multicell(batch_size, reuse)

        def scan_rnn(accum, elem):
            # each element is the set of all hidden states from the previous
            # time step
            cell_states = self.split_out_cell_states(accum)

            h_aboves = self.get_h_aboves([cs.h for cs in cell_states],
                                         batch_size, hmlstm)  # [B, sum(ha_l)]
            # [B, I] + [B, sum(ha_l)] -> [B, I + sum(ha_l)]
            hmlstm_in = array_ops.concat((elem, h_aboves), axis=1)
            _, state = hmlstm(hmlstm_in, cell_states)
            # a list of (c=[B, h_l], h=[B, h_l], z=[B, 1]) ->
            # a list of [B, h_l + h_l + 1]
            concated_states = [array_ops.concat(tuple(s), axis=1) for s in state]
            return array_ops.concat(concated_states, axis=1)  # [B, H]

        # denote 'elem_len' as 'H'
        elem_len = (sum(self._hidden_state_sizes) * 2) + self._num_layers

        states = tf.scan(scan_rnn, self.batch_in, self._initial_states)  # [T, B, H]

        def map_indicators(elem):
            state = self.split_out_cell_states(elem)
            return tf.concat([l.z for l in state], axis=1)

        raw_indicators = tf.map_fn(map_indicators, states)  # [T, B, L]
        indicators = tf.transpose(raw_indicators, [1, 2, 0])  # [B, L, T]
        to_map = tf.concat((states, self.batch_out), axis=2)  # [T, B, H + O]

        def map_output(elem):
            splits = tf.constant([elem_len, self._output_size])
            cell_states, outcome = array_ops.split(value=elem,
                                                   num_or_size_splits=splits,
                                                   axis=1)

            hs = [s.h for s in self.split_out_cell_states(cell_states)]
            gated = self.gate_input(tf.concat(hs, axis=1))  # [B, sum(h_l)]
            embeded = self.embed_input(gated)  # [B, E]
            loss, prediction = self.output_module(embeded, outcome)
            # [B, embeded_size + output_size * 2] or [B, embeded_size + 1 + output_size]
            return tf.concat((embeded, loss, prediction), axis=1)

        def map_extract_layer_h(elem):
            hs = [s.h for s in self.split_out_cell_states(elem)]
            hs = tf.concat(hs, axis=1)  # [B,  sum(h_l)]
            return hs

        hs = tf.map_fn(map_extract_layer_h, states)  # [T, B, sum(h_l)]
        hs = tf.transpose(hs, [1, 0, 2])
        hs = tf.split(hs, self._hidden_state_sizes, axis=2)  # List([T, B, h_l])

        mapped = tf.map_fn(map_output, to_map)  # [T, B, _]

        # mapped has diffenent shape for task 'regression' and 'classification'
        embeded = mapped[:, :, :self._embed_size]
        loss = mapped[:, :, self._embed_size:-self._output_size]  # [T, B, 1]
        loss = tf.transpose(loss, [1, 0, 2])  # [B, T, 1]
        loss = tf.reshape(loss, [-1, self._max_seq_length])  # [B, T]
        sequence_mask = tf.sequence_mask(tf.to_int32(self.lengths), maxlen=self._max_seq_length,
                                         dtype=tf.float32)  # [B, T]
        loss = tf.multiply(loss, sequence_mask)  # [B, T]
        loss = tf.reduce_mean(loss)  # [1]
        regularization = 0  # 0.0000000001*tf.reduce_mean(indicators)
        loss = loss + regularization
        predictions = mapped[:, :, -self._output_size:]  #  [T, B, output_size]
        predictions = tf.nn.softmax(predictions)

        gvs = self._optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -self._grad_clip, self._grad_clip), var) for grad, var in gvs]
        train = self._optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        accuracy = tf.metrics.accuracy(tf.argmax(self.batch_out, axis=2), tf.argmax(predictions, axis=2))

        return train, loss, indicators, predictions, embeded, hs, states, accuracy

    def init(self):
        if self._session is None:
            self._get_graph()
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self._session = tf.train.MonitoredTrainingSession()
            self._session.run(init)

    def get_initial_states(self, TBPTT=False):
        if not TBPTT or self._last_states is None:
            return np.zeros((self._batch_size, (sum(self._hidden_state_sizes) * 2) + self._num_layers))
        else:
            return self._last_states

    def reset_states(self):
        self._last_states = None

    def train_iterator(self, initializer, epochs=1, verbose=False):
        losses = []

        optim, loss, _, _, _, _, states, _ = self._get_graph()

        for epoch in range(epochs):
            self._session.run(initializer)
            if verbose: print('Epoch %d' % epoch)
            ops = [optim, loss, states]
            while True:
                try:
                    _, _loss, _states = self._session.run(ops)
                    losses.append(_loss)
                    self._last_states = _states[-1, :, :]
                except tf.errors.OutOfRangeError:
                    break
            if _loss < 0:
                raise Exception('Negative loss')
            if verbose: print('loss:', _loss)
        return losses

    def train(self,
              batches_in,
              batches_out,
              batches_seq_lengths,
              epochs=3,
              verbose=False,
              TBPTT=False):
        """
        Train the network.

        params:
        ---
        batches_in: a 4 dimensional numpy array. The dimensions should be
            [num_batches, batch_size, num_timesteps, input_size]
            These represent the input at each time step for each batch.
        batches_out: a 4 dimensional numpy array. The dimensions should be
            [num_batches, batch_size, num_timesteps, output_size]
            These represent the output at each time step for each batch.
        epochs: integer, number of epochs
        """

        optim, loss, _, _, _, _, states, _ = self._get_graph()

        losses = []

        for epoch in range(epochs):
            epoch_losses = []
            if verbose: print('Epoch %d' % epoch)
            for batch_in, batch_out, seq_lengths in zip(batches_in, batches_out, batches_seq_lengths):
                ops = [optim, loss, states]
                feed_dict = {
                    self.batch_in: np.swapaxes(batch_in, 0, 1),
                    self.batch_out: np.swapaxes(batch_out, 0, 1),
                    self.lengths: seq_lengths,
                    self._initial_states: self.get_initial_states(TBPTT)
                }
                _, _loss, _states = self._session.run(ops, feed_dict)
                self._last_states = _states[-1, :, :]
                if _loss < 0:
                    raise Exception('Negative loss')
                if verbose: print('loss:', _loss)
                epoch_losses.append(_loss)
            losses.append(sum(epoch_losses))
        return losses

    def predict_iterator(self, initializer):
        self._session.run(initializer)

        _, _, _, predictions, embeddings, hs, states, accuracy = self._get_graph()

        accuracies = []

        while True:
            try:
                inputs, _predictions, _embeddings, _hs, _states, _accuracy = self._session.run([self.batch_in, predictions, embeddings, hs, states, accuracy])
                accuracies.append(_accuracy)
                self._last_states = _states[-1, :, :]
            except tf.errors.OutOfRangeError:
                break
        return np.mean(accuracies)


    def predict(self, batch, return_gradients=False, TBPTT=False):
        """
        Make predictions.

        If there is no active session in the network
        object (i.e. it has not yet been used to train or predict, or the
        tensorflow session has been manually closed), variables will be
        loaded from the provided path. Otherwise variables already present
        in the session will be used.

        params:
        ---
        batch: batch for which to make predictions. should have dimensions
            [batch_size, num_timesteps, output_size]


        returns:
        ---
        predictions for the batch
        embeddings of the last timesteps for the batch
        """

        batch = np.array(batch)
        _, _, _, predictions, embeddings, hs, states, _ = self._get_graph()

        # batch_out is not used for prediction, but needs to be fed in
        batch_out_size = (batch.shape[1], batch.shape[0], self._output_size)
        gradients = tf.gradients(predictions[-1:, :], self.batch_in)
        _predictions, _gradients, _embeddings, _hs, _states = self._session.run([predictions, gradients, embeddings, hs, states],
                                                                           {
                                                                               self.batch_in: np.swapaxes(batch, 0, 1),
                                                                               self.batch_out: np.zeros(batch_out_size),
                                                                               self._initial_states: self.get_initial_states(TBPTT)
                                                                           })
        self._last_states = _states[-1, :, :]

        if return_gradients:
            return tuple(np.swapaxes(r, 0, 1) for
                         r in (_predictions, _gradients[0]))

        return np.swapaxes(_predictions, 0, 1), np.swapaxes(_embeddings, 0, 1), _hs

    def predict_boundaries(self, batch, TBPTT=False):
        """
        Find indicator values for every layer at every timestep.

        If there is no active session in the network
        object (i.e. it has not yet been used to train or predict, or the
        tensorflow session has been manually closed), variables will be
        loaded from the provided path. Otherwise variables already present
        in the session will be used.

        params:
        ---
        batch: batch for which to make predictions. should have dimensions
            [batch_size, num_timesteps, output_size]

        returns:
        ---
        indicator values for ever layer at every timestep
        """

        batch = np.array(batch)
        _, _, indicators, _, _, _, _, _ = self._get_graph()

        # batch_out is not used for prediction, but needs to be fed in
        batch_out_size = (batch.shape[1], batch.shape[0], self._output_size)
        _indicators = self._session.run(indicators, {
            self.batch_in: np.swapaxes(batch, 0, 1),
            self.batch_out: np.zeros(batch_out_size),
            self._initial_states: self.get_initial_states(TBPTT)
        })

        return np.array(_indicators)

    def _get_graph(self):
        if self._graph is None:
            self._graph = self.network(reuse=False)
        return self._graph

    def print_nb_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Number of parameters", total_parameters)

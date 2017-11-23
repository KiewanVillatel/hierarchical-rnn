from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import layers
import collections

HMLSTMState = collections.namedtuple('HMLSTMCellState', ['c', 'h', 'z'])


class HMLSTMCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, batch_size, h_below_size, h_above_size,
                 reuse, layer_norm=True, layer_norm_gain=1.0, layer_norm_shift=0.0, recursion_depth=2):
        super(HMLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._h_below_size = h_below_size
        self._h_above_size = h_above_size
        self._batch_size = batch_size
        self._layer_norm = layer_norm
        self._layer_norm_g = layer_norm_gain
        self._layer_norm_b = layer_norm_shift
        self._recursion_depth = recursion_depth

    @property
    def state_size(self):
        # the state is c, h, and z
        return self._num_units, self._num_units, 1

    @property
    def output_size(self):
        # outputs h and z
        return self._num_units + 1

    def zero_state(self, batch_size, dtype):
        c = tf.zeros([batch_size, self._num_units])
        h = tf.zeros([batch_size, self._num_units])
        z = tf.zeros([batch_size])
        return HMLSTMState(c=c, h=h, z=z)

    def recurrent_call(self, inputs, state, outputs, nb_call):

        with vs.variable_scope("recursion"):

            if nb_call == 0:
                return outputs, state

            c = state.c  # [B, h_l]
            h = state.h  # [B, h_l]
            z = state.z  # [B, 1]

            in_splits = tf.constant([self._h_below_size, 1, self._h_above_size])

            hb, zb, ha = array_ops.split(
                value=inputs,
                num_or_size_splits=in_splits,
                axis=1,
                name='split')  # [B, hb_l], [B, 1], [B, ha_l]

            s_recurrent = h  # [B, h_l]

            expanded_z = z  # [B, 1]
            s_above = tf.multiply(expanded_z, ha)  # [B, ha_l]
            s_below = tf.multiply(zb, hb)  # [B, hb_l]

            length = 4 * self._num_units + 1
            states = [s_recurrent, s_above, s_below]

            bias_init = tf.constant_initializer(0, dtype=tf.float32)
            # [B, 4 * h_l + 1]

            concat = rnn_cell_impl._linear(states, length, bias=True,
                                           bias_initializer=bias_init)

            gate_splits = tf.constant(
                ([self._num_units] * 4) + [1], dtype=tf.int32)

            i, g, f, o, z_tilde = array_ops.split(
                value=concat, num_or_size_splits=gate_splits, axis=1)

            if self._layer_norm:
                i = self._norm(i, 'i')  # [B, h_l]
                g = self._norm(g, 'g')  # [B, h_l]
                f = self._norm(f, 'f')  # [B, h_l]
                o = self._norm(o, 'o')  # [B, h_l]

            i = tf.sigmoid(i)  # [B, h_l]
            g = tf.tanh(g)  # [B, h_l]
            f = tf.sigmoid(f)  # [B, h_l]
            o = tf.sigmoid(o)  # [B, h_l]

            new_c = self.calculate_new_cell_state(c, g, i, f, z, zb)
            new_h = self.calculate_new_hidden_state(h, o, new_c, z, zb)
            new_z = tf.expand_dims(self.calculate_new_indicator(z_tilde), -1)

            output = array_ops.concat((new_h, new_z), axis=1)  # [B, h_l + 1]
            new_state = HMLSTMState(c=new_c, h=new_h, z=new_z)

            return self.recurrent_call(inputs, new_state, output, nb_call - 1)

    def call(self, inputs, state):
        """
        Hierarchical multi-scale long short-term memory cell (HMLSTM)

        inputs: [B, hb_l + 1 + ha_l]
        state: (c=[B, h_l], h=[B, h_l], z=[B, 1])

        output: [B, h_l + 1]
        new_state: (c=[B, h_l], h=[B, h_l], z=[B, 1])
        """
        return self.recurrent_call(inputs, state, None, self._recursion_depth)

    def _norm(self, inp, scope):
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._layer_norm_g)
        beta_init = init_ops.constant_initializer(self._layer_norm_b)
        with vs.variable_scope(scope):
            # Initialize beta and gamma for use by layer_norm.
            vs.get_variable("gamma", shape=shape, initializer=gamma_init)
            vs.get_variable("beta", shape=shape, initializer=beta_init)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def calculate_new_cell_state(self, c, g, i, f, z, zb):
        '''
        update c and h according to correct operations

        c, g, i, f: [B, h_l]
        z, zb: [B, 1]

        new_c: [B, h_l]
        '''
        z = tf.squeeze(z, axis=[1])  # [B]
        zb = tf.squeeze(zb, axis=[1])  # [B]
        new_c = tf.where(
            tf.equal(z, tf.constant(1., dtype=tf.float32)),  # [B]
            tf.multiply(i, g, name='c'),  # [B, h_l], flush
            tf.where(
                tf.equal(zb, tf.constant(0., dtype=tf.float32)),  # [B]
                tf.identity(c),  # [B, h_l], copy
                tf.add(tf.multiply(f, c), tf.multiply(i, g))  # [B, h_l], update
            )
        )
        return new_c  # [B, h_l]

    def calculate_new_hidden_state(self, h, o, new_c, z, zb):
        '''
        h, o, new_c: [B, h_l]
        z, zb: [B, 1]

        new_h: [B, h_l]
        '''
        z = tf.squeeze(z, axis=[1])  # [B]
        zb = tf.squeeze(zb, axis=[1])  # [B]
        new_h = tf.where(
            tf.logical_and(
                tf.equal(z, tf.constant(0., dtype=tf.float32)),
                tf.equal(zb, tf.constant(0., dtype=tf.float32))
            ),  # [B]
            tf.identity(h),  # [B, h_l], if copy
            tf.multiply(o, tf.tanh(self._norm(new_c, 'new_c') if self._layer_norm else new_c))  # [B, h_l], otherwise
        )
        return new_h  # [B, h_l]

    def calculate_new_indicator(self, z_tilde):
        # use slope annealing trick
        slope_multiplier = 1  # NOTE: Change this for some tasks
        sigmoided = tf.sigmoid(z_tilde * slope_multiplier)

        # replace gradient calculation - use straight-through estimator
        # see: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        graph = tf.get_default_graph()
        with ops.name_scope('BinaryRound') as name:
            with graph.gradient_override_map({'Round': 'Identity'}):
                new_z = tf.round(sigmoided, name=name)

        return tf.squeeze(new_z, axis=1)

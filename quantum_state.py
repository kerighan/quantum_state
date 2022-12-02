import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Layer


def sparse_dot(a, b):
    return tf.ragged.map_flat_values(tf.matmul, a, b)


class QuantumState(Layer):
    def __init__(self, n_states=10, units=None, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.units = units
        self.activation = Activation(activation)

    def build(self, input_shape):
        in_dim = input_shape[-1]
        in_len = input_shape[-2]
        if self.units is None:
            units = in_dim
        else:
            units = self.units

        self.PSI_r = self.add_weight(name="PSI_r",
                                     shape=(in_dim, 1),
                                     initializer="glorot_normal")
        self.PSI_theta = self.add_weight(name="PSI_theta",
                                         shape=(in_dim, 1),
                                         initializer="glorot_uniform")
        self.PSI_theta_bias = self.add_weight(name="PSI_theta_bias",
                                              shape=(1,),
                                              initializer="zeros")

        self.PHI_r = self.add_weight(name="PHI_r",
                                     shape=(in_dim, self.n_states),
                                     initializer="glorot_normal")
        self.PHI_theta = self.add_weight(name="PHI_theta",
                                         shape=(in_dim, self.n_states),
                                         initializer="glorot_uniform")
        self.PHI_theta_bias = self.add_weight(name="PHI_theta_bias",
                                              shape=(1,),
                                              initializer="zeros")

        self.S = self.add_weight(name="S",
                                 shape=(self.n_states, units),
                                 initializer="glorot_uniform")
        self.S_bias = self.add_weight(name="S_bias",
                                      shape=(1,),
                                      initializer="zeros")

    def get_psi(self, input, mask):
        # compute wave function psi coefficients
        logits = tf.matmul(input, self.PSI_r)
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)
        ai = tf.exp(logits)
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            ai *= mask
        psi_probabilities = ai / tf.reduce_sum(ai, axis=1, keepdims=True)
        psi_modulus_sqrt = tf.math.sqrt(psi_probabilities)

        psi_argument = tf.matmul(input, self.PSI_theta)
        psi_argument += self.PSI_theta_bias
        psi_argument = np.pi*(1 + tf.nn.tanh(psi_argument))
        return psi_modulus_sqrt, psi_argument

    def get_phi(self, input, mask):
        # compute wave function phi coefficients
        logits = tf.matmul(input, self.PHI_r)
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)
        ai = tf.exp(logits)
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            ai *= mask
        phi_probabilities = ai / tf.reduce_sum(ai, axis=1, keepdims=True)
        phi_modulus_sqrt = tf.math.sqrt(phi_probabilities)

        phi_argument = tf.matmul(input, self.PHI_theta)
        phi_argument += self.PHI_theta_bias
        phi_argument = -np.pi*(1 + np.pi*tf.nn.tanh(phi_argument))
        return phi_modulus_sqrt, phi_argument

    def get_psi_ragged(self, input):
        # compute wave function psi coefficients
        logits = sparse_dot(input, self.PSI_r)
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)
        ai = tf.exp(logits)
        psi_probabilities = ai / tf.reduce_sum(ai, axis=1, keepdims=True)
        psi_modulus_sqrt = tf.math.sqrt(psi_probabilities)

        psi_argument = tf.ragged.map_flat_values(
            tf.matmul, input, self.PSI_theta)
        psi_argument += self.PSI_theta_bias
        psi_argument = np.pi*(1 + tf.nn.tanh(psi_argument))
        return psi_modulus_sqrt, psi_argument

    def get_phi_ragged(self, input):
        # compute wave function phi coefficients
        logits = sparse_dot(input, self.PHI_r)
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)
        ai = tf.exp(logits)
        phi_probabilities = ai / tf.reduce_sum(ai, axis=1, keepdims=True)
        phi_modulus_sqrt = tf.math.sqrt(phi_probabilities)

        phi_argument = tf.ragged.map_flat_values(
            tf.matmul, input, self.PHI_theta)
        phi_argument += self.PHI_theta_bias
        phi_argument = -np.pi*(1 + np.pi*tf.nn.tanh(phi_argument))
        return phi_modulus_sqrt, phi_argument

    def call(self, input, mask=None):
        if isinstance(input, tf.RaggedTensor):
            psi_modulus_sqrt, psi_argument = self.get_psi_ragged(input)
            phi_modulus_sqrt, phi_argument = self.get_phi_ragged(input)
        else:
            psi_modulus_sqrt, psi_argument = self.get_psi(input, mask)
            phi_modulus_sqrt, phi_argument = self.get_phi(input, mask)

        modulus = psi_modulus_sqrt * phi_modulus_sqrt
        argument = psi_argument - phi_argument
        real = tf.reduce_sum(modulus * tf.cos(argument), axis=1)
        imag = tf.reduce_sum(modulus * tf.sin(argument), axis=1)
        collapse = tf.square(real) + tf.square(imag)

        value = tf.matmul(collapse, self.S)
        return self.activation(value + self.S_bias)


class MultiHeadQuantumState(Layer):
    def __init__(
        self,
        n_heads=4, n_states=10,
        units=None, activation="tanh",
        hidden_dim=None, hidden_activation="tanh",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_states = n_states
        self.units = units
        self.hidden_dim = hidden_dim
        if hidden_dim is not None:
            self.dense_layers = [
                tf.keras.layers.Dense(hidden_dim, activation=hidden_activation)
                for _ in range(n_heads)
            ]
        self.quantum_layers = [
            QuantumState(n_states=n_states, units=units, activation=activation)
            for _ in range(n_heads)
        ]

    def call(self, input):
        if self.hidden_dim is not None:
            res = [quantum(dense(input))
                   for quantum, dense in zip(self.quantum_layers,
                                             self.dense_layers)]
        else:
            res = [quantum(input) for quantum in self.quantum_layers]
        return tf.concat(res, axis=-1)

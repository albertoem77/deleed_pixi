import os
import numpy as np
import keras
from keras import ops, regularizers
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
from isrobust_TFM.layers import InformedConstraint,InformedBiasConstraint

class Sampling(Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.seed_generator = keras.random.SeedGenerator(42)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), mean=0.0, stddev=0.1)#, seed=self.seed_generator
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class LossVAE(Layer):
    """
    Layer that adds VAE total loss to the model.
    """

    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim

    def call(self, inputs):
        inputs, outputs, z_mean, z_log_sigma = inputs
        reconstruction_loss =K.mean(K.square(inputs - outputs))
        reconstruction_loss *= self.input_dim
        kl_loss = 1 + z_log_sigma - ops.square(z_mean) - ops.exp(z_log_sigma)
        kl_loss = ops.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = ops.mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return outputs


class InformedVAE():
    def __init__(self, adjacency_matrices,act="tanh", learning_rate=1e-5, seed=42): #adjacency_names,adjacency_activation
        self.adjacency_matrices = adjacency_matrices if isinstance(adjacency_matrices, list) else [adjacency_matrices]
        self.latent_dim = self.adjacency_matrices[-1].shape[1] // 2
        self.act = act
        self.learning_rate = learning_rate
        set_all_seeds(seed)
        self.input_dim = self.adjacency_matrices[0].shape[0]


    def build_informed_layer(self, adj):
        return Dense(
            adj.shape[1],
            activation=self.act,
            activity_regularizer=regularizers.L2(1e-5),
            kernel_constraint=InformedConstraint(adj),
            bias_constraint=InformedBiasConstraint(adj),
            name=f"informed_layer_{len(self.layers)}"
        )

    def build_vae(layers, seed, learning_rate):
        latent_dim = layers[-1].kernel_constraint.adj.shape[1] // 2
        input_dim = layers[0].kernel_constraint.adj.shape[0]
       
    
        inputs = Input(shape=(input_dim,))
    
        # build recursevely the hidden layers of the encoder
        for i, layer in enumerate(layers):
            if i == 0:
                inner_encoder = layer(inputs)
            else:
                inner_encoder = layer(inner_encoder)
    
        z_mean = Dense(latent_dim)(inner_encoder)
        z_log_sigma = Dense(latent_dim)(inner_encoder)
    
        z = Sampling()([z_mean, z_log_sigma])
    
        # Create encoder
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name="encoder")
    
        # Create decoder
        latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
    
        # build recursevely the hidden layers of the decoder
        for i, layer in enumerate(layers[::-1]):
            if i == 0:
                inner_decoder = Dense(
                    layer.kernel_constraint.adj.shape[1], activation="tanh"
                )(latent_inputs)
            else:
                inner_decoder = Dense(
                    layer.kernel_constraint.adj.shape[1], activation="tanh"
                )(inner_decoder)
    
        outputs = Dense(input_dim, activation="linear")(inner_decoder)
        decoder = Model(latent_inputs, outputs, name="decoder")
    
        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae_outputs = VAE_Loss(input_dim)([inputs, outputs, z_mean, z_log_sigma])
        vae = Model(inputs, vae_outputs, name="vae_mlp")
    
        vae.compile(optimizer=Adam(learning_rate=learning_rate), metrics=["mse"])
        self.decoder = decoder
        self.encoder = encoder
        self.vae = vae

        

    def fit(self, *args, **kwargs):
        return self.vae.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.vae.predict(*args, **kwargs)







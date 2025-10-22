import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import warnings
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.layers import concatenate as concat
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import random as python_random

warnings.filterwarnings('ignore')
# Removed tf.compat.v1.disable_eager_execution() for compatibility with newer TensorFlow versions

def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(K.shape(mu)[0], K.shape(mu)[1]), mean=0., stddev=1., seed=42)
    return mu + K.exp(l_sigma / 2) * eps

def define_cvae(X_train, y_train, latent_space_size, num_hidden_layers, hidden_layer_dims, activ, random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    python_random.seed(random_seed)
    tf.random.set_seed(random_seed)

    os.environ['TF_DETERMINISTIC_OPS'] = str(1)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    n_x = X_train.shape[1]
    n_y = y_train.shape[1]

    X = Input(shape=(n_x,), name='x')
    label = Input(shape=(n_y,), name='labels')
    inputs = concat([X, label], name='cvae_inputs')

    encoder_layers = []
    for i in range(num_hidden_layers):
        encoder_layers.append(Dense(hidden_layer_dims[i], activation=activ, name='encoder_layer_' + str(i)))

    encoder_layers[0] = encoder_layers[0](inputs)
    for i in range(1, num_hidden_layers):
        encoder_layers[i] = encoder_layers[i](encoder_layers[i - 1])

    mu = Dense(latent_space_size, activation='linear', name='mu')(encoder_layers[-1])
    l_sigma = Dense(latent_space_size, activation='linear', name='sigma')(encoder_layers[-1])

    z = Lambda(sample_z, output_shape=(latent_space_size,), name='z')([mu, l_sigma])
    zc = concat([z, label], name='zc')

    decoder_layers = []
    for i in range(num_hidden_layers):
        decoder_layers.append(Dense(hidden_layer_dims[num_hidden_layers - 1 - i], activation=activ, name='decoder_layer_' + str(i)))
    decoder_out = Dense(n_x, activation='sigmoid', name='output')

    cvae_trained_decoder = [decoder_layers[0](zc)]
    for i in range(1, num_hidden_layers):
        cvae_trained_decoder.append(decoder_layers[i](cvae_trained_decoder[i - 1]))

    outputs = decoder_out(cvae_trained_decoder[-1])

    # Create a model that outputs both reconstruction and latent variables
    cvae_to_train = Model([X, label], [outputs, mu, l_sigma])

    encoder_to_train = Model([X, label], mu)

    d_in = Input(shape=(latent_space_size + n_y,), name='decoder_input')
    decoder_trained = [decoder_layers[0](d_in)]
    for i in range(1, num_hidden_layers):
        decoder_trained.append(decoder_layers[i](decoder_trained[i - 1]))
    d_out = decoder_out(decoder_trained[-1])

    decoder_to_train = Model(d_in, d_out)

    return [cvae_to_train, encoder_to_train, decoder_to_train]

# Custom loss function that works with the new model output
def vae_loss_wrapper(y_true, y_pred_tuple):
    """
    y_pred_tuple contains: [reconstruction, mu, l_sigma]
    But Keras will pass them separately, so we need to handle this differently
    """
    # Only use the reconstruction part for the main loss
    return K.sum(K.binary_crossentropy(y_true, y_pred_tuple), axis=-1)

def train_cvae(cvae_to_train, X_train, y_train, X_test, y_test, batchsize, n_epoch, optim, pat):
    # Convert pandas DataFrames to numpy arrays if needed
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    # Convert to float32 for TensorFlow
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Custom training loop to handle VAE loss with KL divergence
    @tf.function
    def train_step(x_batch, y_batch, x_true):
        with tf.GradientTape() as tape:
            # Forward pass
            reconstruction, mu, l_sigma = cvae_to_train([x_batch, y_batch], training=True)
            
            # Reconstruction loss
            recon_loss = tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(x_true, reconstruction),
                axis=-1
            )
            
            # KL divergence loss
            kl_loss = 0.5 * tf.reduce_sum(
                tf.exp(l_sigma) + tf.square(mu) - 1.0 - l_sigma,
                axis=-1
            )
            
            # Total loss
            total_loss = tf.reduce_mean(recon_loss + kl_loss)
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, cvae_to_train.trainable_variables)
        optim.apply_gradients(zip(gradients, cvae_to_train.trainable_variables))
        
        return total_loss, tf.reduce_mean(kl_loss), tf.reduce_mean(recon_loss)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'loss': [], 'val_loss': [], 'KL_loss': [], 'recon_loss': []}
    
    for epoch in range(n_epoch):
        print(f'Epoch {epoch + 1}/{n_epoch}')
        
        # Training
        train_losses = []
        train_kl_losses = []
        train_recon_losses = []
        
        for i in range(0, len(X_train), batchsize):
            batch_end = min(i + batchsize, len(X_train))
            x_batch = X_train[i:batch_end]
            y_batch = y_train[i:batch_end]
            
            loss, kl, recon = train_step(x_batch, y_batch, x_batch)
            train_losses.append(loss.numpy())
            train_kl_losses.append(kl.numpy())
            train_recon_losses.append(recon.numpy())
        
        # Validation
        val_reconstruction, val_mu, val_l_sigma = cvae_to_train([X_test, y_test], training=False)
        val_recon_loss = tf.reduce_mean(tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(X_test, val_reconstruction),
            axis=-1
        ))
        val_kl_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(
            tf.exp(val_l_sigma) + tf.square(val_mu) - 1.0 - val_l_sigma,
            axis=-1
        ))
        val_loss = val_recon_loss + val_kl_loss
        
        epoch_loss = np.mean(train_losses)
        epoch_kl = np.mean(train_kl_losses)
        epoch_recon = np.mean(train_recon_losses)
        
        history['loss'].append(epoch_loss)
        history['val_loss'].append(val_loss.numpy())
        history['KL_loss'].append(epoch_kl)
        history['recon_loss'].append(epoch_recon)
        
        print(f'loss: {epoch_loss:.4f} - KL_loss: {epoch_kl:.4f} - recon_loss: {epoch_recon:.4f} - val_loss: {val_loss.numpy():.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= pat:
                print(f'Early stopping after {epoch + 1} epochs')
                break
    
    # Create a simple object to return that mimics Keras History
    class SimpleHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    return SimpleHistory(history)

def single_encoded_sample(x_row, y_row, trained_encoder):
    return trained_encoder.predict([x_row.reshape(1, x_row.shape[0]), y_row.reshape(1, y_row.shape[0])])

def single_decoded_sample(encoded_sample, trained_decoder):
    return trained_decoder.predict(encoded_sample)

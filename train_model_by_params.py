import cvae_general
import pandas as pd
from tensorflow.keras.optimizers import Adam
import helper
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Model Training Script", description="Trains a CVAE model based on inputted hyperparameters and saves the model to a specified location")
    parser.add_argument('training_data', help="Path to .pickle, .csv, or .tsv for individual training samples", type=str)
    parser.add_argument('t_start', help="Position of first one-hot-encoded tissue in the training data", type=int)
    parser.add_argument('t_end', help="Position of last one-hot-encoded tissue in the training data", type=int)
    parser.add_argument('s_start', help="Position of first one-hot-encoded species in the training data", type=int)
    parser.add_argument('s_end', help="Position of last one-hot-encoded species in the training data", type=int)
    parser.add_argument('d_start', help="Position of first probe in the training data", type=int)
    parser.add_argument('encoder_save_loc', help='Path to location to save trained encoder model', type=str)
    parser.add_argument('decoder_save_loc', help='Path to location to save trained decoder model', type=str)

    parser.add_argument('n', help='Hidden layer dimension 2**n', type=int)
    parser.add_argument('activation_function', help='Activation function for neural network training (i.e. relu, sigmoid, tanh)', type=str)
    parser.add_argument('latent_space_dimension', help='Size of encoded latent space', type=int)
    parser.add_argument('learning_rate', help='Learning rate used by the Adam optimizer', type=float)
    parser.add_argument('epsilon', help='Epsilon used by the Adam optimizer', type=float)
    parser.add_argument('layout_index', help='Integer between 0 and 4 to select the layout of hidden layers (see paper for options)', type=int)
    parser.add_argument('--val_seed', help="Random seed for selecting the validation dataset", default=-1, type=int)
    parser.add_argument('--seed', help="Random seed used to initiate model training", default=42, type=int)

    args = parser.parse_args()

    if os.path.splitext(args.training_data)[1] == '.pickle':
        training = pd.read_pickle(args.training_data)
    elif os.path.splitext(args.training_data)[1] == 'csv' or args.training_data.split('.', 1)[1] == 'csv.gz':
        training = pd.read_table(args.training_data, sep=',', index_col=0)
    else:
        training = pd.read_table(args.training_data, index_col=0)
    training = training.dropna(axis=1)
    print('Training data dimensions: ' + str(training.shape))

    tissue_index = training.columns.values[args.t_start:args.t_end+1]
    species_index = training.columns.values[args.s_start:args.s_end+1]

    train_data, val_data = helper.get_training_val_datasets(training, tissue_index, args.t_start, args.t_end, species_index, args.s_start, args.s_end, args.val_seed)

    Xtrain = train_data[train_data.columns[args.d_start:]]
    ytrain = train_data[train_data.columns[:args.d_start]]
    Xval = val_data[val_data.columns[args.d_start:]]
    yval = val_data[val_data.columns[:args.d_start]]

    print('\n' + '='*60)
    print('DATA SHAPES')
    print('='*60)
    print(f'Training features (X):   {Xtrain.shape[0]:>6} samples × {Xtrain.shape[1]:>6} probes')
    print(f'Training labels (y):     {ytrain.shape[0]:>6} samples × {ytrain.shape[1]:>6} features')
    print(f'Validation features (X): {Xval.shape[0]:>6} samples × {Xval.shape[1]:>6} probes')
    print(f'Validation labels (y):   {yval.shape[0]:>6} samples × {yval.shape[1]:>6} features')

    layouts = [(1, [2 ** args.n]), (2, [2 ** args.n, 2 ** args.n]), (3, [2 ** args.n, 2 ** args.n, 2 ** args.n]), (2, [2 ** (args.n + 1), 2 ** args.n]), (3, [2 ** (args.n + 2), 2 ** (args.n + 1), 2 ** args.n])]
    layout = layouts[args.layout_index]
    n_epochs = 50
    batch_size = 32

    print('\n' + '='*60)
    print('MODEL HYPERPARAMETERS')
    print('='*60)
    print(f'Layout index:            {args.layout_index}')
    print(f'Number of layers:        {layout[0]}')
    print(f'Layer dimensions:        {layout[1]}')
    print(f'Activation function:     {args.activation_function}')
    print(f'Latent space dimension:  {args.latent_space_dimension}')
    print(f'Learning rate:           {args.learning_rate}')
    print(f'Epsilon:                 {args.epsilon}')
    print(f'Random seed:             {args.seed}')
    print(f'Validation seed:         {args.val_seed}')
    print(f'Number of epochs:        {n_epochs}')
    print(f'Batch size:              {batch_size}')
    print('='*60 + '\n')

    cvae, encoder, decoder = cvae_general.define_cvae(Xtrain, ytrain, args.latent_space_dimension, layout[0], layout[1], args.activation_function, args.seed)
    trained_cvae = cvae_general.train_cvae(cvae, Xtrain, ytrain, Xval, yval, batch_size, n_epochs, Adam(learning_rate=args.learning_rate, epsilon=args.epsilon), 5)

    # Save models using the new Keras 3 format
    encoder.save(args.encoder_save_loc, save_format='keras')
    decoder.save(args.decoder_save_loc, save_format='keras')

#!/usr/bin/env python
# coding: utf-8

# Created by Jan Steggemann, based on prior work by Markus Seidel

import os
import pathlib
import datetime
import h5py
import optparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Flatten, Reshape, Dense, BatchNormalization, Concatenate, Embedding
from keras import optimizers, initializers
from keras.layers import Lambda
from keras.backend import slice
from keras.layers.advanced_activations import PReLU

import tensorflow as tf
import keras.backend as K

from tensorflow import train

# Local imports
from cyclical_learning_rate import CyclicLR
from AdamW import AdamW
from HDF5MultiFile import DataGenerator, FileInput, FileInputSliceLast
from lr_finder import LRFinder
from weighted_sum_layer import weighted_sum_layer
from recoil_plots import recoil_plots

mpl.use('Agg')

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0.00001, 20])
    plt.yscale('log')
    plt.legend()


def custom_loss(y_true, y_pred, abs_resp=False):
    px_truth = K.flatten(y_true[:,0])
    py_truth = K.flatten(y_true[:,1])
    px_pred = K.flatten(y_pred[:,0])
    py_pred = K.flatten(y_pred[:,1])

    pt_truth = K.sqrt(px_truth*px_truth + py_truth*py_truth)

    px_truth1 = px_truth / pt_truth
    py_truth1 = py_truth / pt_truth

    if abs_resp:
        upar_pred = px_truth1 * px_pred + py_truth1 * py_pred - pt_truth
        # Note 50 = 50*norm_fac = 2500 GeV (if norm_fac = 50)
        upar_pred_l100 = tf.boolean_mask(upar_pred, upar_pred < 100.)
        upar_pred_plus = tf.boolean_mask(upar_pred_l100, upar_pred_l100 > 0.)
        upar_pred_minus = tf.boolean_mask(upar_pred_l100, upar_pred_l100 < 0.)

        dev = ((tf.reduce_sum(upar_pred_plus) + tf.reduce_sum(upar_pred_minus))/tf.reduce_sum(upar_pred_plus - upar_pred_minus))**2
    else: # Relative response
        # Secretly using absolute response
        # upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred)/pt_truth
        upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred) - pt_truth
        # upar_pred = tf.boolean_mask(upar_pred, pt_truth > 20./50.) - 1.
        pt_cut = pt_truth > 0./50.
        upar_pred = tf.boolean_mask(upar_pred, pt_cut)
        pt_truth_filtered = tf.boolean_mask(pt_truth, pt_cut)

        filter_bin0 = pt_truth_filtered < 5./50.
        filter_bin1 = tf.logical_and(pt_truth_filtered > 5./50., pt_truth_filtered < 10./50.)
        filter_bin2 = pt_truth_filtered > 10./50.

        upar_pred_pos_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred > 0.))
        upar_pred_neg_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred < 0.))
        upar_pred_pos_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred > 0.))
        upar_pred_neg_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred < 0.))
        upar_pred_pos_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred > 0.))
        upar_pred_neg_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred < 0.))
        norm = tf.reduce_sum(pt_truth_filtered)
        dev = tf.abs(tf.reduce_sum(upar_pred_pos_bin0) + tf.reduce_sum(upar_pred_neg_bin0))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin1) + tf.reduce_sum(upar_pred_neg_bin1))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin2) + tf.reduce_sum(upar_pred_neg_bin2))
        dev /= norm

        # upar_pred_gr1 = tf.boolean_mask(upar_pred, upar_pred > 0.)
        # upar_pred_le1 = tf.boolean_mask(upar_pred, upar_pred < 0.)

        # dev = tf.abs((tf.reduce_sum(upar_pred_gr1) + tf.reduce_sum(upar_pred_le1))/(tf.reduce_sum(pt_truth_filtered)))

    # uperp_pred = px_truth1 * py_pred - py_truth1 * px_pred

    # loss = K.mean(upar_pred**2 + uperp_pred**2)
    loss = 0.5*K.mean((px_pred - px_truth)**2 + (py_pred - py_truth)**2)
    # dev = (K.mean(upar_pred_plus) + K.mean(upar_pred_minus))**2

    loss += 200.*dev
    return loss

def create_output_graph(n_features=8, n_features_cat=3, n_graph_layers=0, n_dense_layers=3, n_dense_per_graph_net=1, activation='tanh', do_weighted_sum=True, with_bias=True):
    # [b'PF_dxy', b'PF_dz', b'PF_eta', b'PF_mass', b'PF_puppiWeight', b'PF_charge', b'PF_fromPV', b'PF_pdgId',  b'PF_px', b'PF_py']
    inputs = Input(shape=(maxNPF, n_features), name='input')
    pxpy = Lambda(lambda x: slice(x, (0, 0, n_features-2), (-1, -1, -1)))(inputs)

    if activation == 'prelu':
        activation = PReLU()

    if opt.embedding:
        embeddings = []
        for i_emb in range(n_features_cat):
            input_cat = Input(shape=(maxNPF, 1), name='input_cat{}'.format(i_emb))
            if i_emb == 0:
                inputs = [inputs, input_cat]
            else:
                inputs.append(input_cat)
            embedding = Embedding(input_dim=emb_input_dim[i_emb], output_dim=emb_out_dim, embeddings_initializer=initializers.RandomNormal(mean=0., stddev=0.4/emb_out_dim), name='embedding{}'.format(i_emb))(input_cat)
            embedding = Reshape((maxNPF, 8))(embedding)
            embeddings.append(embedding)

        throughput = Concatenate()([inputs[0]] + [emb for emb in embeddings])

    # x = GlobalExchange()(inputs if not opt.embedding else throughput)
    if opt.embedding:
        x = throughput

    for i_graph in range(n_graph_layers):
        if i_graph > 0:
            x = GlobalExchange()(x)
        for __ in range(n_dense_per_graph_net):
            x = Dense(64, activation=activation, kernel_initializer='lecun_uniform')(x)
            # x = BatchNormalization(momentum=0.8)(x)
        x = GravNet(n_neighbours=20, n_dimensions=4,
                    n_filters=42, n_propagate=18)(x)
        x = BatchNormalization(momentum=0.8)(x)

    dense_layers = [] # [4]
    if do_weighted_sum:
        for i_dense in range(n_dense_layers):
            x = Dense(64//2**i_dense, activation=activation, kernel_initializer='lecun_uniform')(x)
            x = BatchNormalization(momentum=0.95)(x)
        # List of weights. Increase to 3 when operating with biases
        # x = Dense(3 if with_bias else 1, activation='linear', kernel_initializer='lecun_uniform')(x)

        # Expect typical weights to not be of order 1 but somewhat smaller, so apply explicit scaling
        x = Dense(3 if with_bias else 1, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        print('Shape of last dense layer', x.shape)
        x = Concatenate()([x, pxpy])
        #x = Flatten()(x)
        x = weighted_sum_layer(with_bias)(x)
    else:
        for i_dense in range(n_dense_layers):
            x = Dense(64//2**i_dense, activation=activation, kernel_initializer='lecun_uniform')(x)
        x = Flatten()(x)
        dense_layers = [32, 16, 8]

    dense_activation = 'relu'
    for dense_size in dense_layers:
        x = Dense(dense_size, activation=dense_activation,
                  kernel_initializer='lecun_uniform')(x)

    x = Dense(2, activation='linear', name='output')(x)
    return inputs, x


def create_simple_graph(n_dense=3, kernel_initializer='lecun_uniform'):
    inputs = Input(shape=(Z.shape[1],), name='input')
    x = Dense(64, activation='relu', kernel_initializer=kernel_initializer)(inputs)
    for _ in range(n_dense - 1):
        x = Dense(64, activation='relu', kernel_initializer=kernel_initializer)(x)
    x = Dense(2, activation='linear', name='output')(x)
    return inputs, x



# configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='input',
                  help='input file', default='tree_100k.h5', type='string')
parser.add_option('-l', '--load', dest='load',
                  help='load model from timestamp', default='', type='string')
parser.add_option('--notrain', dest='notrain',
                  help='do not train model', default=False, action='store_true')
parser.add_option('--simple', dest='simple',
                  help='use simple model', default=False, action='store_true')
parser.add_option('--embedding', dest='embedding',
                  help='use embeddings', default=False, action='store_true')
parser.add_option('--find_lr', dest='find_lr',
                  help='run learning rate finder', default=False, action='store_true')
(opt, args) = parser.parse_args()

# general setup
maxNPF = 4500
n_features_pf = 8
n_features_pf_cat = 3
normFac = 50.
epochs = 50
batch_size = 192*32 if opt.simple else 256//8
preprocessed = True
emb_out_dim = 8


with h5py.File(opt.input, 'r', swmr=True) as h5f:
    # Y = h5f['Y'][:]
    # Z = h5f['Z'][:]
    if not opt.simple:
        # X = h5f['X'] #[:]
        if opt.embedding:
            X_c = [h5f[f'X_c_{i}'] for i in range(n_features_pf_cat)]

    if not opt.simple:
        # print(X.shape)
        # n_features = X.shape[2]
        if opt.embedding:
             # categorical inputs
            print([x.shape for x in X_c])

            emb_input_dim = {
                i:np.max(X_c[i][0:1000]) + 1 for i in range(n_features_pf_cat)
            }

            # for i in range(n_features_cat):
            #     print('Embedding', i, 'max value', np.max(X_c[:,:,i]))
            # n_max_embed = np.max(np.max(X_c[:,i] for i in range(n_features_cat)))


print('Embedding dimensions', emb_input_dim)

# met_flavours = ['', 'Chs', 'NoPU', 'Puppi', 'PU', 'PUCorr', 'Raw']
met_flavours = ['',  'Puppi', 'Raw']

Y = FileInput(opt.input, 'Y')
print('Y.shape', Y.shape)
# print('Z.shape', Z.shape)

# inputs, outputs = create_output_graph()
if opt.simple:
    inputs, outputs = create_simple_graph()
else:
    inputs, outputs = create_output_graph(n_features=n_features_pf, n_features_cat=n_features_pf_cat)

lr_scale = 1.

# lr = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=len(Y)/5., mode='triangular2')
clr = CyclicLR(base_lr=0.0003*lr_scale, max_lr=0.001*lr_scale, step_size=len(Y)/batch_size, mode='triangular2')

# create the model
model = Model(inputs=inputs, outputs=outputs)
# compile the model
optimizer = optimizers.Adam(lr=1., clipnorm=1.)
# optimizer = optimizers.SGD(lr=0.0001, decay=0., momentum=0., nesterov=False)
# optimizer = AdamW(lr=0.0000, beta_1=0.8, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.000, batch_size=batch_size, samples_per_epoch=int(len(Z)*0.8), epochs=epochs)

model.compile(loss='mse', optimizer=optimizer,
              metrics=['mean_absolute_error', 'mean_squared_error'])
# model.compile(loss=custom_loss, optimizer=optimizer, 
              # metrics=['mean_absolute_error', 'mean_squared_error'])
# print the model summary
model.summary()

if opt.load:
    timestamp = opt.load
else:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
path = f'models/{timestamp}'
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

plot_model(model, to_file=f'{path}/model.png', show_shapes=True)

if opt.load:
    model.load_weights(f'{path}/model.h5')
    print(f'Restored model {timestamp}')


with open(f'{path}/summary.txt', 'w') as txtfile:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: txtfile.write(x + '\n'))

Yr = Y

indices = np.array([i for i in range(len(Yr)//batch_size)])
indices_train, indices_test = train_test_split(indices, test_size=0.2, random_state=7)

Z = FileInput(opt.input, 'Z')

# Now we need a generator that takes these indices and splits on the h5py file
Xr = FileInput(opt.input, 'X') if not opt.simple else Z
if not opt.simple and opt.embedding:
    # Xr = [FileInput(opt.input, 'X')] + [X_c[:,:,i:i+1] for i in range(n_features_cat)]
    Xr = [FileInput(opt.input, 'X')] + [FileInput(opt.input, f'X_c_{i}') for i in range(n_features_pf_cat)]

gen_x_train = DataGenerator(Xr, [Yr], batch_size, indices_train)
gen_x_test = DataGenerator(Xr, [Yr], batch_size, indices_test)

# early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# model checkpoint callback
# this saves our model architecture + parameters into dense_model.h5
model_checkpoint = ModelCheckpoint(f'{path}/model.h5', monitor='val_loss',
                                   verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto',
                                   period=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=0.000001, cooldown=3, verbose=1)

stop_on_nan = keras.callbacks.TerminateOnNaN()


# LR finder
if opt.find_lr:
    # pre-train to avoid model being too far away from interesting range
    history = model.fit_generator(gen_x_train, epochs=2, verbose=1, callbacks=[clr])
    lr_finder = LRFinder(model)
    lr_finder.find_generator(gen_x_train, 0.00001, 1.0, 5)
    lr_finder.plot_loss()
    import pdb; pdb.set_trace()


# Run training
if not opt.notrain:
    # Train classifier
    history = model.fit_generator(gen_x_train,
                                  epochs=epochs,
                                  verbose=1,  # switch to 1 for more verbosity
                                  callbacks=[early_stopping, clr, stop_on_nan],#, reduce_lr], #, lr,   reduce_lr],
                                  # callbacks=[early_stopping, reduce_lr], #, lr, reduce_lr],
                                  use_multiprocessing=True,
                                  workers=8,
                                  validation_data=gen_x_test)
    # ## Plot performance
    # Here, we plot the history of the training and the performance in a ROC curve

    plot_history(history)
    plt.savefig(f'{path}/history.pdf', bbox_inches='tight')
    model.save(f'{path}/model.h5')

    from tensorflow import saved_model
    saved_model.simple_save(K.get_session(), f'{path}/saved_model', inputs={t.name:t for t in model.input}, outputs={t.name:t for t in model.outputs})

    from tensorflow.python.framework import graph_util
    frozen_graph = graph_util.convert_variables_to_constants(K.get_session(), K.get_session().graph_def, ['output/BiasAdd'])
    # train.write_graph(graph_or_graph_def=K.get_session().graph_def, logdir=f'{path}', name='saved_model.pb', as_text=False)
    train.write_graph(graph_or_graph_def=frozen_graph, logdir=f'{path}', name='saved_model.pb', as_text=False)

# Print info about weights
names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()
for name, weight in zip(names, weights):
    print(name, weight.mean(), weight.var())


### need to get Z here if not there ###

# Print info about physically meaningful quantities
selectors = {
    'ptgr100':np.where(np.sqrt(Y[:, 1]**2 + Y[:, 0]**2) > 100./normFac),
    'ptgr50':np.where(np.sqrt(Y[:, 1]**2 + Y[:, 0]**2) > 50./normFac),
    'all':np.ones(len(Z), dtype=bool)
}

# Get predictions
pred = model.predict(Xr)
px_pred = pred[:, 0].flatten()*normFac
py_pred = pred[:, 1].flatten()*normFac

truth = Y
px_truth = truth[:, 0]*normFac
py_truth = truth[:, 1]*normFac

for sel_name, sel in selectors.items():
    print('Selection:', sel_name)
    for i, met_flavour in enumerate(met_flavours + ['Zero']):
        pred_px = Z[:, i*3]*normFac if met_flavour != 'Zero' else np.zeros(len(Z))
        pred_py = Z[:, i*3 + 1]*normFac if met_flavour != 'Zero' else np.zeros(len(Z))
        mse_px = np.mean((pred_px[sel] - px_truth[sel])**2)
        mse_py = np.mean((pred_py[sel] - py_truth[sel])**2)
        print(f'Algo {met_flavour:8} MSE px {mse_px:.1f} MSE py {mse_py:.1f}')
    mse_px = np.mean((px_pred[sel] - px_truth[sel])**2)
    mse_py = np.mean((py_pred[sel] - py_truth[sel])**2)
    met_flavour = 'DNN'
    print(f'Algo {met_flavour:8} MSE px {mse_px:.1f} MSE py {mse_py:.1f}')


pt_pred = np.sqrt(px_pred*px_pred + py_pred*py_pred)
pt_truth = np.sqrt(px_truth*px_truth + py_truth*py_truth)

px_truth1 = px_truth / pt_truth
py_truth1 = py_truth / pt_truth

par_pred = px_truth1 * px_pred + py_truth1 * py_pred

px_pred_pfmet = Z[:, 0].flatten()*normFac
py_pred_pfmet = Z[:, 1].flatten()*normFac
par_pred_pfmet = px_truth1 * px_pred_pfmet + py_truth1 * py_pred_pfmet

px_pred_puppi = Z[:, met_flavours.index('Puppi')*3].flatten()*normFac
py_pred_puppi = Z[:, met_flavours.index('Puppi')*3 + 1].flatten()*normFac
par_pred_puppi = px_truth1 * px_pred_puppi + py_truth1 * py_pred_puppi

truth = px_truth, py_truth, pt_truth
pred = px_pred, py_pred, pt_pred, par_pred
pfmet = px_pred_pfmet, py_pred_pfmet, par_pred_pfmet
puppi = px_pred_puppi, py_pred_puppi, par_pred_puppi

recoil_plots(truth, pred, pfmet, puppi, path)

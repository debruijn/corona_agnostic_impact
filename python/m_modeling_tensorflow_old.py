import pandas as pd
import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from m_create_modeling_data import create_modeling_data
from python.m_postmodeling import model_forecast
import tf_keras
import tensorflow_probability as tfp
tfd = tfp.distributions


def seq2seq_window_dataset(series, window_size, batch_size=32,
                           shuffle_buffer=1000):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def run(config):
    for kpi in config.kpis:
        data = pd.read_pickle(f'data/processed/deaths_by_full_week_{config.data_version}.pkl')
        model_estimations = {}

        for period in config.periods.keys():
            print(f'Running model for kpi {kpi} for data {config.data_version}, and period "{period}" up to '
                  f'{config.periods[period]}')

            sdata = create_modeling_data(data, kpi, period)

            # Build model.
            model = tf_keras.Sequential([
                tf_keras.layers.Dense(2),
                tf_keras.layers.Dense(2),
                tfp.layers.DistributionLambda(
                    lambda t: tfd.Normal(loc=t[..., :1],
                                         scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
            ])

            negloglik = lambda y, rv_y: -rv_y.log_prob(y)

            # Do inference.
            model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.01), loss=negloglik)
            x = np.hstack((np.expand_dims(sdata['I_Y'], 1), np.expand_dims(sdata['I_W'], 1)))
            # x = np.expand_dims(sdata['I_W'], 1)
            x = np.hstack((pd.get_dummies(sdata['I_Y']).values.astype(float), pd.get_dummies(sdata['I_W']).values.astype(float)))
            y = sdata['y'].astype(float)
            model.fit(x, y, epochs=1000, verbose=True)
            [print(np.squeeze(w.numpy())) for w in model.weights]
            print(model(np.hstack([np.ones((1, 1)), np.zeros((1, 75)), np.ones((1, 1))])).mean().numpy())
            print(model(np.hstack([np.zeros((1, 24)), np.ones((1, 1)), np.zeros((1, 51)), np.ones((1, 1))])).mean().numpy())

            # Build model.
            model = tf_keras.Sequential([
                tf_keras.layers.Dense(64),
                tf_keras.layers.Dense(2),
                tfp.layers.DistributionLambda(
                    lambda t: tfd.Normal(loc=t[..., :1],
                                         scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
            ])

            negloglik = lambda y, rv_y: -rv_y.log_prob(y)

            # Do inference.
            model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.01), loss=negloglik)
            x = np.hstack((np.expand_dims(sdata['I_Y'], 1), np.expand_dims(sdata['I_W'], 1)))
            # x = np.expand_dims(sdata['I_W'], 1)
            # x = np.hstack((pd.get_dummies(sdata['I_Y']).values.astype(float), pd.get_dummies(sdata['I_W']).values.astype(float)))
            y = sdata['y'].astype(float)
            model.fit(x, y, epochs=1000, verbose=True)
            [print(np.squeeze(w.numpy())) for w in model.weights]
            print(model(np.array([1, 51]).reshape((1, 2))).mean().numpy())
            print(model(np.array([24, 51]).reshape((1, 2))).mean().numpy())

            # Build model.
            model = tf_keras.Sequential([
                tf_keras.layers.Dense(4),
                tf_keras.layers.Dense(2),
                tfp.layers.DistributionLambda(
                    lambda t: tfd.Normal(loc=t[..., :1] + 2000,
                                         scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
            ])

            negloglik = lambda y, rv_y: -rv_y.log_prob(y)

            # Do inference.
            model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.01), loss=negloglik)
            x = np.hstack((np.expand_dims(sdata['I_Y'], 1), np.expand_dims(sdata['I_W'], 1)))
            # x = np.expand_dims(sdata['I_W'], 1)
            # x = np.hstack((pd.get_dummies(sdata['I_Y']).values.astype(float), pd.get_dummies(sdata['I_W']).values.astype(float)))
            y = sdata['y'].astype(float)
            x = np.hstack([x, np.vstack((np.zeros((1,1)), np.expand_dims(y[:-1], 1)))])
            model.fit(x, y, epochs=1000, verbose=True)
            [print(np.squeeze(w.numpy())) for w in model.weights]

            # print(model(np.array([1, 51]).reshape((1, 2))).mean().numpy())
            # print(model(np.array([24, 51]).reshape((1, 2))).mean().numpy())

            model_estimations[period] = model
            model_estimations[f'model_{period}'] = model

        model_estimations['full_sdata'] = create_modeling_data(data, kpi, "incl_1st_wave")
        model_estimations['data'] = data

        if not os.path.exists(f'output/{kpi}'):
            os.mkdir(f'output/{kpi}/')
        pd.to_pickle(model_estimations, f'output/{kpi}/model_{config.data_version}_tensorflow.pkl')


if __name__ == "__main__":
    from config import config as run_config

    os.chdir('..')
    run(run_config)

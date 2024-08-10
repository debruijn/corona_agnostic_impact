import pandas as pd
import os
import tensorflow as tf
import numpy as np

from python.m_create_modeling_data import create_modeling_data
import tf_keras
import tensorflow_probability as tfp

from python.m_postmodeling import model_eval

tfd = tfp.distributions


# Output in modeling, for each model A and B:
# - Each model produces a fit and a OOS predict
# - Using the fit, calculate RMSE/R^2/etc
# - Using the predict, the same for two periods
# - Then store: model, sdata, fit, predict, error, and potentially covid
# - Then just skip pomo (or make that the output)


n_epoch = 1000
verbose = False


def run(config):
    for kpi in config.kpis:
        data = pd.read_pickle(f'data/processed/deaths_by_full_week_{config.data_version}.pkl')
        model_estimations = {}

        for period in config.periods.keys():
            print(f'Running model for kpi {kpi} for data {config.data_version}, and period "{period}" up to '
                  f'{config.periods[period]}')

            sdata = create_modeling_data(data, kpi, period)

            # Model definition:
            model = tf_keras.Sequential([
                tf_keras.layers.Dense(64, activation='relu'),  # none, relu, sigmoid
                tf_keras.layers.Dense(2),
                tfp.layers.DistributionLambda(
                    lambda t: tfd.Normal(loc=sdata['y'].min() + t[..., :1],
                                         scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
            ])

            # Run the model estimation
            negative_loglikelihood = lambda y, rv_y: -rv_y.log_prob(y)
            model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.1), loss=negative_loglikelihood)
            x = np.hstack((np.expand_dims(sdata['I_Y'], 1), np.expand_dims(sdata['I_W'], 1)))
            y = sdata['y'].astype(float)
            model.fit(x, y, epochs=n_epoch, verbose=verbose)

            # Get in-sample fit and forecast for next 8 weeks
            in_sample_fit = model(x).mean().numpy().squeeze()
            x_next_8_weeks = np.array([np.repeat(x[:, 0].max(), 8), x[-1, 1] + range(1, 9)]).transpose()
            next_8_weeks_fit = model(x_next_8_weeks).mean().numpy().squeeze()
            next_8_weeks_y = np.array(
                [data.loc[(data.year == row[0] + data.year.min()) & (data.week == row[1] + 1), kpi].values[0]
                 for row in x_next_8_weeks])

            model_eval_in_sample = model_eval(in_sample_fit, y)
            model_eval_next_8_weeks = model_eval(next_8_weeks_fit, next_8_weeks_y)
            model_estimations[period] = [model_eval_in_sample, model_eval_next_8_weeks]

            if period == "train_test":
                n_needed = 51 - x[-1, 1]
                x_rest_of_year = np.array(
                    [np.repeat(x[:, 0].max(), n_needed), x[-1, 1] + range(1, n_needed + 1)]).transpose()
                rest_of_year_fit = model(x_rest_of_year).mean().numpy().squeeze()
                rest_of_year_y = np.array([
                    data.loc[(data.year == row[0] + data.year.min()) & (data.week == row[1] + 1), kpi].values[0] for row in
                    x_rest_of_year])
                model_eval_rest_of_year = model_eval(rest_of_year_fit, rest_of_year_y)
                model_estimations[period].append(model_eval_rest_of_year)

        model_estimations['full_sdata'] = create_modeling_data(data, kpi, "incl_1st_wave")
        model_estimations['data'] = data

        if not os.path.exists(f'output/{kpi}'):
            os.mkdir(f'output/{kpi}/')
        pd.to_pickle(model_estimations, f'output/{kpi}/model_{config.data_version}_tensorflow.pkl')


if __name__ == "__main__":
    from config import config as run_config

    os.chdir('..')
    run(run_config)

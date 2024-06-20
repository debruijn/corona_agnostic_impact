
from jax import random
from numpyro.infer import NUTS, MCMC
import pandas as pd
from numpyro_models.BaseModel import BaseModel
import os
import arviz as az
import numpyro


def run(config):

    chains = 2
    numpyro.set_host_device_count(chains)

    for kpi in config.kpis:

        print(f'Running model for kpi {kpi} for data {config.data_version}')
        data = pd.read_pickle(f'data/processed/deaths_by_full_week_{config.data_version}.pkl')

        data['covid_week'] = 0
        data.loc[(data.year == 2020) & (data.week > 9), 'covid_week'] = \
            data.loc[(data.year == 2020) & (data.week > 9), 'week'] - 9

        sdata = {
            'N': data.shape[0],
            'W': data.week.max(),
            'Y': data.year.max() - data.year.min() + 1,
            'C': data.covid_week.values.max(),
            'I_W': data.week.values - 1,
            'I_Y': data.year.values - data.year.values.min(),
            'I_C': data.covid_week.values,
            'y': data[kpi].values
        }
        sdata.update({'mu_y': sdata['y'].mean(),
                      'sigma_y': sdata['y'].std(),
                      'sigma_w': 0.025,
                      'sigma_c': sdata['y'].mean(),
                      'sigma_s': sdata['y'].std()})

        model = BaseModel(sdata)

        # pyro_models modelling
        nuts_kernel = NUTS(model.model)
        mcmc = MCMC(sampler=nuts_kernel,
                    num_samples=1000,
                    num_warmup=1000,
                    num_chains=chains)

        rng_key = random.PRNGKey(6)
        # rng_key, rng_key_ = random.split(rng_key)
        mcmc.run(y=sdata['y'], rng_key=rng_key)

        if not os.path.exists(f'output/{kpi}'):
            os.mkdir(f'output/{kpi}/')
        pd.to_pickle(az.from_numpyro(mcmc), f'output/{kpi}/model_{config.data_version}.pkl')


if __name__ == "__main__":
    from config import config as run_config
    os.chdir('..')
    run(run_config)

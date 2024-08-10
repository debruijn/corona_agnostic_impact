from jax import random
from numpyro.infer import NUTS, MCMC
import pandas as pd
import os
import arviz as az
import numpyro

from numpyro_models.BaseModel import BaseModel
from original_analysis.m_create_modeling_data import create_modeling_data


def run(config):

    chains = 2
    numpyro.set_host_device_count(chains)

    for kpi in config.kpis:

        print(f'Running model for kpi {kpi} for data {config.data_version}')
        data = pd.read_pickle(f'data/processed/deaths_by_full_week_{config.data_version}.pkl')
        sdata = create_modeling_data(data, kpi=kpi)
        model = BaseModel(sdata, allow_dynamics=True)

        # pyro_models modelling
        nuts_kernel = NUTS(model.model)
        mcmc = MCMC(sampler=nuts_kernel,
                    num_samples=1000,
                    num_warmup=500,
                    num_chains=chains)

        rng_key = random.PRNGKey(6)
        mcmc.run(y=sdata['y'], rng_key=rng_key)

        if not os.path.exists(f'output/{kpi}'):
            os.mkdir(f'output/{kpi}/')
        pd.to_pickle(az.from_numpyro(mcmc), f'output/{kpi}/model_{config.data_version}.pkl')


if __name__ == "__main__":
    from config import config as run_config
    os.chdir('..')
    run(run_config)

import os
from jax import random
from numpyro.infer import NUTS, MCMC
import pandas as pd
import arviz as az
import numpyro

from numpyro_models.BaseModel import BaseModel
from original_analysis.m_create_modeling_data import create_modeling_data
from comparison_pkgs.m_postmodeling import model_eval


def get_fit(results, year=-1, week=8):
    n_years = results['posterior']['year_eff'].data.shape[2]
    chains, iters = results['posterior']['year_eff'].data.shape[:2]
    n_weeks = 52
    week_eff = results['posterior']['week_eff'].data.reshape(chains*iters, n_weeks)
    year_eff = results['posterior']['year_eff'].data.reshape(chains*iters, n_years)
    return (week_eff[:, week] * year_eff[:, year]).mean()


def run(config):

    chains = 2
    numpyro.set_host_device_count(chains)

    for kpi in config.kpis:
        data = pd.read_pickle(f'data/processed/deaths_by_full_week_{config.data_version}.pkl')
        model_estimations = {}

        for period in config.periods.keys():
            print(f'Running model for kpi {kpi} for data {config.data_version}, and period "{period}" up to '
                  f'{config.periods[period]}')
            sdata = create_modeling_data(data, kpi, period)

            model = BaseModel(sdata, allow_dynamics=False)
            nuts_kernel = NUTS(model.model)
            mcmc = MCMC(sampler=nuts_kernel,
                        num_samples=1000,
                        num_warmup=500,
                        num_chains=chains)

            rng_key = random.PRNGKey(42)
            mcmc.run(y=sdata['y'], rng_key=rng_key)
            results = az.from_numpyro(mcmc)

            # Get in-sample fit and forecast for next 8 weeks
            this_yr, this_wk = config.periods[period]
            y = results.observed_data['y'].data
            in_sample_fit = [get_fit(results, year=row.iloc[0] - min(data.year), week=row.iloc[1]-1)
                             for i, row in data[['year', 'week']].iterrows()
                             if (row.iloc[0] < this_yr) or (row.iloc[0] == this_yr and row.iloc[1] <= this_wk)]
            model_eval_in_sample = model_eval(in_sample_fit, y)

            next_8_weeks_y = data.loc[(data.year == this_yr) & (data.week <= this_wk + 8) & (data.week > this_wk), kpi].values
            next_8_weeks_fit = [get_fit(results, year=row.iloc[0] - min(data.year), week=row.iloc[1]-1)
                                for i, row in data[['year', 'week']].iterrows()
                                if row.iloc[0] == this_yr and this_wk < row.iloc[1] <= this_wk + 8]
            model_eval_next_8_weeks = model_eval(next_8_weeks_fit, next_8_weeks_y)
            model_estimations[period] = [model_eval_in_sample, model_eval_next_8_weeks]

            if period == "train_test":
                rest_of_year_y = data.loc[(data.year == this_yr) & (data.week > this_wk), kpi].values
                rest_of_year_fit = [get_fit(results, year=row.iloc[0] - min(data.year), week=row.iloc[1] - 1)
                                    for i, row in data[['year', 'week']].iterrows()
                                    if row.iloc[0] == this_yr and this_wk < row.iloc[1]]
                model_eval_rest_of_year = model_eval(rest_of_year_fit, rest_of_year_y)
                model_estimations[period].append(model_eval_rest_of_year)

        model_estimations['full_sdata'] = create_modeling_data(data, kpi, "incl_1st_wave")
        model_estimations['data'] = data

        if not os.path.exists(f'output/{kpi}'):
            os.mkdir(f'output/{kpi}/')
        pd.to_pickle(model_estimations, f'output/{kpi}/model_{config.data_version}_numpyro.pkl')


if __name__ == "__main__":
    from config import config as run_config
    os.chdir('..')
    run(run_config)

import pandas as pd
import os
import arviz as az
import statsmodels.api as sm


def run(config):

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

        sm.OLS(pd.Series(sdata['y']), pd.concat([pd.get_dummies(sdata['I_Y'], prefix='y'),
                                                 pd.get_dummies(sdata['I_W'], prefix='w', drop_first=True),
                                                 pd.get_dummies(sdata['I_C'], prefix='c', drop_first=True)],
                                                axis=1)).fit().summary()
        params = sm.OLS(pd.Series(sdata['y']), pd.concat([pd.get_dummies(sdata['I_Y'], prefix='y'),
                                                          pd.get_dummies(sdata['I_W'], prefix='w', drop_first=True),
                                                          pd.get_dummies(sdata['I_C'], prefix='c', drop_first=True)],
                                                         axis=1)).fit().params
        sum_covid_effect_significant = sum(params[f'c_{i}'] for i in range(1, 10))
        sum_covid_effect_all = sum(params[f'c_{i}'] for i in range(1, 33))
        if not os.path.exists(f'output/{kpi}'):
            os.mkdir(f'output/{kpi}/')
        # pd.to_pickle(az.from_numpyro(mcmc), f'output/{kpi}/model_{config.data_version}.pkl')


if __name__ == "__main__":
    from config import config as run_config
    os.chdir('..')
    run(run_config)

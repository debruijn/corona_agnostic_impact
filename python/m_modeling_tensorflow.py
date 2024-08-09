import pandas as pd
import os
import statsmodels.api as sm

from m_create_modeling_data import create_modeling_data


def run(config):
    for kpi in config.kpis:
        data = pd.read_pickle(f'data/processed/deaths_by_full_week_{config.data_version}.pkl')
        model_estimations = {}

        for period in config.periods.keys():
            print(f'Running model for kpi {kpi} for data {config.data_version}, and period "{period}" up to '
                  f'{config.periods[period]}')

            sdata = create_modeling_data(data, kpi, period)

            model = sm.OLS(pd.Series(sdata['y']),
                           pd.concat([pd.get_dummies(sdata['I_Y'], prefix='y'),
                                      pd.get_dummies(sdata['I_W'], prefix='w', drop_first=True)],
                                     axis=1))

            model_estimations[period] = model.fit()

        model_estimations['full_sdata'] = create_modeling_data(data, kpi, "incl_1st_wave")
        model_estimations['data'] = data

        if not os.path.exists(f'output/{kpi}'):
            os.mkdir(f'output/{kpi}/')
        pd.to_pickle(model_estimations, f'output/{kpi}/model_{config.data_version}_statsmodels.pkl')


if __name__ == "__main__":
    from config import config as run_config

    os.chdir('..')
    run(run_config)

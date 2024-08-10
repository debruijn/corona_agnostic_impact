import pandas as pd
import os
import statsmodels.api as sm
import numpy as np

from python.m_create_modeling_data import create_modeling_data
from python.m_postmodeling import model_eval


def get_fit(model_fit, year=-1, week=8, count_years=25, min_year=1995):
    return model_fit.predict(to_dummy(year, week - 1, count_years, min_year))


def to_dummy(year, week, nr_years, min_year, nr_weeks=52):
    year -= min_year
    if week == 0:
        [False] * year + [True] + [False] * (nr_years - year - 1) + [False] * (nr_weeks - 1)
    return ([False] * year + [True] + [False] * (nr_years - year - 1) +
            [False] * (week - 1) + [True] + [False] * (nr_weeks - week - 1))


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
            model_fit = model.fit()

            # Get in-sample fit and forecast for next 8 weeks
            this_yr, this_wk = config.periods[period]
            n_years = sdata['I_Y'].max() + 1
            min_year = data.year.min()
            y = sdata['y']
            in_sample_fit = model_fit.fittedvalues
            model_eval_in_sample = model_eval(in_sample_fit, y)

            next_8_weeks_y = data.loc[
                (data.year == this_yr) & (data.week <= this_wk + 8) & (data.week > this_wk), kpi].values
            next_8_weeks_fit = np.concatenate([get_fit(model_fit, row.iloc[0], row.iloc[1] - 1, n_years, min_year)
                                for i, row in data[['year', 'week']].iterrows()
                                if row.iloc[0] == this_yr and this_wk < row.iloc[1] <= this_wk + 8])
            model_eval_next_8_weeks = model_eval(next_8_weeks_fit, next_8_weeks_y)
            model_estimations[period] = [model_eval_in_sample, model_eval_next_8_weeks]

            if period == "train_test":
                rest_of_year_y = data.loc[(data.year == this_yr) & (data.week > this_wk), kpi].values
                rest_of_year_fit = np.concatenate([get_fit(model_fit, row.iloc[0], row.iloc[1] - 1, n_years, min_year)
                                    for i, row in data[['year', 'week']].iterrows()
                                    if row.iloc[0] == this_yr and this_wk < row.iloc[1]])
                model_eval_rest_of_year = model_eval(rest_of_year_fit, rest_of_year_y)
                model_estimations[period].append(model_eval_rest_of_year)

        model_estimations['full_sdata'] = create_modeling_data(data, kpi, "incl_1st_wave")
        model_estimations['data'] = data

        if not os.path.exists(f'output/{kpi}'):
            os.mkdir(f'output/{kpi}/')
        pd.to_pickle(model_estimations, f'output/{kpi}/model_{config.data_version}_statsmodels.pkl')


if __name__ == "__main__":
    from config import config as run_config

    os.chdir('..')
    run(run_config)

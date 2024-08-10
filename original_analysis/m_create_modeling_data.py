from config import config
import pandas as pd


def create_modeling_data(data, kpi, period="train_test"):

    data['covid_week'] = 0
    data.loc[(data.year == 2020) & (data.week > 9), 'covid_week'] = \
        data.loc[(data.year == 2020) & (data.week > 9), 'week'] - 9

    data = pd.concat([data.loc[data.year < config.periods[period][0]],
                      data.loc[(data.year == config.periods[period][0]) & (data.week <= config.periods[period][1])]])

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

    return sdata

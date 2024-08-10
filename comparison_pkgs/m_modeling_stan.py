import pystan
import pandas as pd


def run():

    data = pd.read_pickle('data/processed/deaths_by_full_week.pkl')

    data['covid_week'] = 0
    data.loc[(data.year == 2020) & (data.week > 9), 'covid_week'] = \
        data.loc[(data.year == 2020) & (data.week > 9), 'week'] - 9

    sdata = {
        'N': data.shape[0],
        'W': data.week.max(),
        'Y': data.year.max() - data.year.min() + 1,
        'C': 16,
        'I_W': data.week.values,
        'I_Y': data.year.values - data.year.values.min() + 1,
        'I_C': data.covid_week.values + 1,
        'y': data.all_A.values

    }

    model = pystan.StanModel('stan/model_v6.stan')
    fit1 = model.sampling(data=sdata, iter=1000)
    print(fit1.stansummary(pars=['year_eff', 'week_eff', 'covid_eff', 'sigma', 'sigma_year', 'sigma_week']))
    tmp = fit1.extract(pars='base_covid_eff')
    pd.Series(tmp['base_covid_eff'].mean(axis=0)).sum()
    print(fit1.stansummary(pars='total_covid_deaths'))

    sdata['y'] = data.over80_A.values
    fit2 = model.sampling(data=sdata, iter=1000)
    print(fit2.stansummary(pars=['year_eff', 'week_eff', 'covid_eff', 'sigma', 'sigma_year', 'sigma_week']))
    tmp = fit2.extract(pars='base_covid_eff')
    pd.Series(tmp['base_covid_eff'].mean(axis=0)).sum()
    print(fit2.stansummary(pars='total_covid_deaths'))

    sdata['y'] = data['65to80_A'].values
    fit3 = model.sampling(data=sdata, iter=1000)
    print(fit3.stansummary(pars=['year_eff', 'week_eff', 'covid_eff', 'sigma', 'sigma_year', 'sigma_week']))
    tmp = fit3.extract(pars='base_covid_eff')
    pd.Series(tmp['base_covid_eff'].mean(axis=0)).sum()
    print(fit3.stansummary(pars='total_covid_deaths'))

    sdata['y'] = data.under65_A.values
    fit4 = model.sampling(data=sdata, iter=1000)
    print(fit4.stansummary(pars=['year_eff', 'week_eff', 'covid_eff', 'sigma', 'sigma_year', 'sigma_week']))
    tmp = fit4.extract(pars='base_covid_eff')
    pd.Series(tmp['base_covid_eff'].mean(axis=0)).sum()
    print(fit4.stansummary(pars='total_covid_deaths'))


if __name__ == "__main__":
    import os
    os.chdir('..')
    run()

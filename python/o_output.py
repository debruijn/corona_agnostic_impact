import pandas as pd
import matplotlib.pyplot as plt
import os


def output_calc(config):
    metrics = ['rmse', 'r2', 'covid', 'data_fit']
    total_res = {}
    for kpi in config.kpis:
        res = {}
        for metric in metrics:
            print(f"Reading {metric} results for kpi {kpi} for all packages.")
            this_res = []
            for pkg in config.packages:
                this_res.append(pd.read_csv(f'output/{kpi}/pomo_{metric}_{pkg}.csv'))
            res[metric] = pd.concat(this_res, axis=1, names=config.packages)

        res['rmse'] = res['rmse'].transpose().drop_duplicates().transpose()
        res['rmse'].columns = ['rmse'] + config.packages

        res['r2'] = res['r2'].transpose().drop_duplicates().transpose()
        res['r2'].columns = ['r2'] + config.packages

        tmp_covid = res['covid'].transpose().drop_duplicates()
        tmp_covid['total'] = tmp_covid[0]
        for i in range(1, 8):
            tmp_covid['total'] += tmp_covid[i]
        res['covid'] = tmp_covid.transpose()
        res['covid'].columns = ['covid_week'] + config.packages
        res['covid']['covid_week'] += 1
        res['covid']['covid_week'] = res['covid']['covid_week'].reset_index().drop(columns=['index'])

        res['data_fit'].drop(columns=['Unnamed: 0'], inplace=True)
        res['data_fit'] = res['data_fit'].transpose().drop_duplicates().transpose()

        keep = res['data_fit'][['y', 'is_train', 'is_before_covid']]
        other = res['data_fit'][['fit_train', 'fit_before_covid']]
        other.columns = [x + '_' + y for y in ['fit_train', 'fit_before_covid'] for x in config.packages]
        res['data_fit'] = keep.join(other)

        total_res[kpi] = res

    return total_res


def output_plots(out_calcs, config):
    for kpi in config.kpis:
        if not os.path.exists(f'output/{kpi}/plots'):
            os.mkdir(f'output/{kpi}/plots')

        # Bar plots:
        # rmse - in-sample for both periods for each package
        out_calcs[kpi]['rmse'].loc[out_calcs[kpi]['rmse']['rmse'].apply(
            lambda x: x.endswith('_in') and not x.startswith('incl_'))].plot(kind='bar', figsize=(16, 10))
        plt.savefig(f'output/{kpi}/plots/rmse_in_sample.png')
        # rmse - in-sample/out-of-sample for train_test for each package
        out_calcs[kpi]['rmse'].loc[out_calcs[kpi]['rmse']['rmse'].apply(
            lambda x: x.startswith('train_test_'))].plot(kind='bar')
        plt.savefig(f'output/{kpi}/plots/rmse_train_test.png')
        # rmse - out-of-sample for all three periods for each package
        out_calcs[kpi]['rmse'].loc[out_calcs[kpi]['rmse']['rmse'].apply(
            lambda x: not x.endswith('_in') and not x.startswith('incl_'))].plot(kind='bar', figsize=(16, 10))
        plt.savefig(f'output/{kpi}/plots/rmse_out_of_sample.png')

        # r2 - in-sample for both periods for each package
        out_calcs[kpi]['r2'].loc[out_calcs[kpi]['r2']['r2'].apply(
            lambda x: x.endswith('_in') and not x.startswith('incl_'))].plot(kind='bar', figsize=(16, 10))
        plt.savefig(f'output/{kpi}/plots/r2_in_sample.png')

        # total_covid - out-of-sample for all three periods for each package
        out_calcs[kpi]['covid'].loc[out_calcs[kpi]['covid'].index == 'total', config.packages].plot(kind='bar',
                                                                                                    figsize=(16, 10))
        plt.savefig(f'output/{kpi}/plots/total_covid.png')

        # Line plots:
        # covid - each week for each package
        out_calcs[kpi]['covid'].loc[out_calcs[kpi]['covid'].index != 'total', config.packages].plot(figsize=(16, 10))
        plt.savefig(f'output/{kpi}/plots/covid_per_week.png')

        # data + fit -> full period, in-sample "before_covid" for each package
        out_calcs[kpi]['data_fit'].loc[
            out_calcs[kpi]['data_fit']['is_before_covid'] == 1, ['y', 'statsmodels_fit_before_covid',
                                                                 'numpyro_fit_before_covid',
                                                                 'tensorflow_fit_before_covid']].plot(figsize=(16, 10))
        plt.savefig(f'output/{kpi}/plots/in_sample_fit_full_period.png')

        # data + fit -> zoomed in on 2019, using "train_test" for each package
        out_calcs[kpi]['data_fit'][1232:1284][['y', 'statsmodels_fit_train',
                                               'numpyro_fit_train',
                                               'tensorflow_fit_train']].plot(figsize=(16, 10))
        plt.savefig(f'output/{kpi}/plots/forecast_test_train.png')

        # data + fit -> zoomed in on 2020, using "before_covid" for each package
        out_calcs[kpi]['data_fit'].iloc[-50:].drop(
            columns=['is_train', 'is_before_covid'])[['y', 'statsmodels_fit_before_covid',
                                                      'numpyro_fit_before_covid',
                                                      'tensorflow_fit_before_covid']].plot(figsize=(16, 10))
        plt.savefig(f'output/{kpi}/plots/forecast_covid_period.png')


def run(config):
    out_calcs = output_calc(config)
    output_plots(out_calcs, config)
    return None


if __name__ == "__main__":
    from config import config as run_config

    os.chdir('..')
    run(run_config)

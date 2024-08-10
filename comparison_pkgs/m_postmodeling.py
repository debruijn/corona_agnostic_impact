import pandas as pd
import numpy as np


def model_eval(fit, y):
    err = y - fit
    rmse = (err**2).mean()**0.5
    r2 = 1 - np.var(err) / np.var(y)
    sum_err = err.sum()
    return {'fit': fit, 'y': y, 'err': err, 'rmse': rmse, 'r2': r2, 'sum_err': sum_err}


def get_metric(results, period='train_test', metric='rmse'):
    res = {}
    res.update({f'{period}_in': results[period][0][metric],
                 f'{period}_next': results[period][1][metric]})
    if period == 'train_test':
        res[f'{period}_full'] = results[period][2][metric]
    return res


def run(config):
    for kpi in config.kpis:
        for pkg in config.packages:
            print(f'Running postmodeling for kpi {kpi} for data {config.data_version} for {pkg}')
            results = pd.read_pickle(f'output/{kpi}/model_{config.data_version}_{pkg}.pkl')

            rmse = {}
            r2 = {}
            for period in config.periods:
                rmse.update(get_metric(results, period, metric='rmse'))
                r2.update(get_metric(results, period, metric='r2'))

            covid = results['before_covid'][1]['err']

            y = np.hstack([results['before_covid'][0]['y'], results['before_covid'][1]['y']])
            fit_train_test = np.hstack([results['train_test'][0]['fit'], results['train_test'][2]['fit']])
            fit_before_covid = np.hstack([results['before_covid'][0]['fit'], results['before_covid'][1]['fit']])
            is_train = np.ones(len(results['train_test'][0]['y']))
            is_before_covid = np.ones(len(results['before_covid'][0]['y']))
            data_fit = pd.Series(y, name='y').to_frame().join(
                pd.Series(fit_train_test, name='fit_train')).join(
                pd.Series(fit_before_covid, name='fit_before_covid')).join(
                pd.Series(is_train, name='is_train')).join(
                pd.Series(is_before_covid, name='is_before_covid'))

            pomo_results = {'rmse': pd.Series(rmse), 'r2': pd.Series(r2), 'covid': pd.Series(covid),
                            'data_fit': data_fit}

            for key, value in pomo_results.items():
                filename = f'output/{kpi}/pomo_{key}_{pkg}.csv'
                value.to_csv(filename)

            print(pomo_results)
            filename = f'output/{kpi}/pomo_{config.data_version}_{pkg}.pkl'
            pd.to_pickle(pomo_results, filename)


if __name__ == "__main__":
    import os
    from config import config as run_config
    os.chdir('..')
    run(run_config)

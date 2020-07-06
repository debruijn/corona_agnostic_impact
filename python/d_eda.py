import pandas as pd


def plots_by_week(data):

    data.loc[data.year<2020].filter(regex="week|all_").groupby('week').mean().plot()

    save_loc = 'eda/by_week/'

    for col in data.columns:
        if col not in ('year', 'week', 'nr_days'):
            file_name = 'plot_by_week_' + col + '.png'
            data.loc[data.year < 2020].filter(items=('week', col)).groupby('week').mean().plot().figure.\
                savefig(save_loc + file_name)

    ages = ("all", "under65", "65to80", "over80")

    for col in ages:
        file_name = 'plot_by_week_' + col + '_MF.png'
        data.loc[data.year < 2020].filter(items=('week', col+'_M', col+'_F')).groupby('week').mean().plot().figure. \
            savefig(save_loc + file_name)


def run():

    # Load data
    data = pd.read_pickle('data/processed/deaths_by_full_week.pkl')

    # Make plots by week, excl 2020
    plots_by_week(data)

    # Make plots by full year, excl 2020

    # Make plots until week 25, incl 2020

    return None


if __name__ == "__main__":
    import os
    os.chdir('..')
    run()

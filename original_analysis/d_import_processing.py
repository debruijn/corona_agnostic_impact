import pandas as pd


def import_data(config):
    data = pd.read_csv(f'data/raw/Overledenen__geslacht_en_leeftijd__per_week_{config.data_version}.csv', header=[4, 5],
                       index_col=0, skipfooter=1, engine='python')
    return data


def process_week_data(data, full_week=False):

    data = data.reset_index()
    data = pd.concat((data, data['index'].str.split(' week', expand=True).rename(
        columns={0: 'year', 1: 'week'})), axis=1)
    data = data.drop('index', axis=1)
    data.year = data.year.astype('int64')

    # Remove text from week ('22*') and days ('(2 dagen)')
    data_weeks = data.week.str.split('(', expand=True).rename(columns={0: 'week', 1: 'nr_days'})
    data_weeks['week'] = data_weeks['week'].str.strip().str.replace('*', '')
    if not full_week:
        data_weeks['nr_days'] = data_weeks.nr_days.str.split(' ', expand=True).rename(columns={0: 'nr_days'}).nr_days
        data_weeks.loc[data_weeks.nr_days.isna(), 'nr_days'] = 7

    # Add processed week and nr days back to original data
    data['week'] = data_weeks['week'].astype('int64')
    if not full_week:
        data['nr_days'] = data_weeks['nr_days'].astype('int64')
    else:
        data['nr_days'] = 7

    return data


def create_day_data_by_week(data):

    day_data = data.filter(like="_").divide(data.nr_days, axis=0).drop('nr_days', axis=1)
    day_data = day_data.combine_first(data)

    return day_data


def process_data(data):

    # Fix headers
    data = data.drop('Perioden')
    data = data.loc[data[data != "."].isna().any(axis=1) == False]  # Drop rows with incomplete data; can be done nicer
    data = data.astype('int64')
    data = data.rename(columns={'Totaal leeftijd': 'all',
                                '0 tot 65 jaar': 'under65',
                                '65 tot 80 jaar': '65to80',
                                '80 jaar of ouder': 'over80',
                                'Mannen': 'M',
                                'Vrouwen': 'F',
                                'Totaal mannen en vrouwen': 'A'})
    data.columns = ['_'.join(col) for col in data.columns]

    # Split year data from week data
    year_indices = [str(x) for x in range(1995, 2020, 1)]
    year_data = data.loc[year_indices]
    week_data = data.loc[[x for x in data.index if x not in year_indices]]
    full_week_data = week_data.loc[[x for x in week_data.index if 'dag' not in x]]

    week_data = process_week_data(week_data)
    full_week_data = process_week_data(full_week_data, full_week=True)

    return {'year': year_data, 'week': week_data, 'full_week': full_week_data}


def check_totals(data_dict):

    year_data = data_dict['year']
    week_data = data_dict['week']
    diffs = {i: week_data.loc[week_data.year == int(i),].all_A.sum() - year_data.loc[i].all_A.sum() for i in year_data.index}

    if not all(diffs[year] == 0 for year in diffs):
        raise(ValueError, 'Year data and week data don\'t align')

    # TODO: check day totals after creating them


def save_data(data_dict, config):

    save_dir = 'data/processed/'

    data_dict['year'].to_pickle(save_dir + f'deaths_by_year_{config.data_version}.pkl')
    data_dict['week'].to_pickle(save_dir + f'deaths_by_week_{config.data_version}.pkl')
    data_dict['full_week'].to_pickle(save_dir + f'deaths_by_full_week_{config.data_version}.pkl')
    data_dict['day'].to_pickle(save_dir + f'deaths_by_day_{config.data_version}.pkl')


def run(config):

    data = import_data(config)
    data_dict = process_data(data)
    data_dict['day'] = create_day_data_by_week(data_dict['week'])
    save_data(data_dict, config)
    check_totals(data_dict)


if __name__ == "__main__":
    import os
    from config import config as run_config
    os.chdir('..')
    run(run_config)

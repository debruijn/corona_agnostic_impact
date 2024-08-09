data_version = ["04072020_132924", "24102020_154220b"]
data_version = data_version[1]
run_eda = False
kpis = ['all_A', 'over80_A', '65to80_A', 'under65_A'][0]
if type(kpis) == str:
    kpis = [kpis]
packages = ['statsmodels', 'numpyro', 'tensorflow']

periods = {"train_test": (2019, 8),
           "before_covid": (2020, 8),
           "incl_1st_wave": (2020, 16)}

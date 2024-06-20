
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def run(config):
    for kpi in config.kpis:
        print(f'Running postmodeling for kpi {kpi} for data {config.data_version}')
        results = pd.read_pickle(f'output/{kpi}/model_{config.data_version}.pkl')

        results.posterior['total_covid_eff'] = results.posterior.covid_eff.sum(axis=2)
        print(results.sample_stats.diverging.sum())

        with PdfPages(f'output/{kpi}/postmodeling_{config.data_version}.pdf') as pdf:
            summary = az.summary(results)
            plt.table(summary.to_numpy(), colLabels=summary.columns, loc='center').figure.set_figheight(10.8)
            pdf.savefig()
            plt.close()

            #az.plot_trace(results, var_names=["sigma", "covid_eff", "total_covid_eff"])
            #pdf.savefig()
            #plt.close()
            az.plot_forest(results, kind="ridgeplot", var_names="covid_eff")[0].set_title(f"For kpi {kpi}")
            pdf.savefig()
            plt.close()
            az.plot_forest(results, kind="ridgeplot", var_names="year_eff")[0].set_title(f"For kpi {kpi}")
            pdf.savefig()
            plt.close()
            az.plot_forest(results, kind="ridgeplot", var_names="week_eff")[0].set_title(f"For kpi {kpi}")
            pdf.savefig()
            plt.close()
            az.plot_forest(results, kind="ridgeplot", var_names="total_covid_eff")[0].set_title(f"For kpi {kpi}")
            pdf.savefig()
            plt.close()
            az.plot_forest(results, kind="ridgeplot", var_names="sigma_w")[0].set_title(f"For kpi {kpi}")
            pdf.savefig()
            plt.close()
            az.plot_forest(results, kind="ridgeplot", var_names="first_year_eff")[0].set_title(f"For kpi {kpi}")
            pdf.savefig()
            plt.close()


if __name__ == "__main__":
    import os
    from config import config as run_config
    os.chdir('..')
    run(run_config)

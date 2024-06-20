# Workflow

from python import d_import_processing, d_eda, m_modeling_numpyro, m_postmodeling, o_output
from config import config


def main():

    d_import_processing.run(config)  # Import files and process to pandas DF with fields as expected
    if config.run_eda:
        d_eda.run(config)  # Visualize the data for EDA purposes
    m_modeling_numpyro.run(config)  # Run model (per country first)
    m_postmodeling.run(config)  # Visualize model results
    o_output.run(config)  # Create plots based on model results


if __name__ == "__main__":
    main()

# Workflow

from python import d_import_processing, d_eda, m_modeling_stan, o_output


def main():
    config = 1

    d_import_processing.run()  # Import files and process to pandas DF with fields as expected
    d_eda.run()  # Visualize the data for EDA purposes
    m_modeling_stan.run()  # Run model (per country first)
    o_output.run()  # Create plots based on model results


if __name__ == "main":
    main()

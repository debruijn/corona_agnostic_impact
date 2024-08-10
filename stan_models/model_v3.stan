data {
    int N; // number of datapoints
    int W; // number of weeks
    int Y; // number of years
    int C; // number of weeks for which Covid estimate is estimated
    int<upper=W> I_W[N]; // indicator for week
    int<upper=Y> I_Y[N]; // indicator for year
    int<upper=C+1> I_C[N]; // indicator for covid period
    vector[N] y; // kpi
}

parameters {
    vector<lower=0>[W-1] base_week_eff;
    vector[C] base_covid_eff;
    real<lower=0> sigma;
    real<lower=0> year_eff_0;
    vector[Y-1] year_eff_shock;
    real<lower=0> sigma_year;
}

transformed parameters {
    vector[N] mu;
    vector[W] week_eff;
    vector[Y] year_eff;
    vector[C+1] covid_eff = append_row(0.0, base_covid_eff);
    week_eff = append_row(1.0, base_week_eff);
    year_eff[1] = year_eff_0;
    for (year in 2:Y) {
        year_eff[year] = year_eff[year-1] + year_eff_shock[year-1];
    }
    mu = year_eff[I_Y] .* week_eff[I_W] + covid_eff[I_C];
}

model {
    sigma ~ cauchy(0, 1);
    sigma_year ~ cauchy(0,1);
    base_week_eff ~ normal(1, 0.4);
    year_eff_0 ~ cauchy(0, 1);
    year_eff_shock ~ normal(0, sigma_year);

    y ~ normal(mu, sigma);
}

generated quantities {
    real total_covid_deaths = sum(covid_eff);
}
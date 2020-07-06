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
    vector<lower=0>[Y] year_eff;
    vector<lower=0>[W-1] base_week_eff;
    vector[C] base_covid_eff;
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu;
    vector[W] week_eff;
    vector[C+1] covid_eff = append_row(0.0, base_covid_eff);
    week_eff = append_row(1.0, base_week_eff);
    mu = year_eff[I_Y] .* week_eff[I_W] + covid_eff[I_C];
}

model {
    base_week_eff ~ normal(1, 0.4);
    year_eff ~ cauchy(0, 1);
    sigma ~ cauchy(0, 1);
    y ~ normal(mu, sigma);
}

generated quantities {
    real total_covid_deaths = sum(covid_eff);
}
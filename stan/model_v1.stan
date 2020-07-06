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
    vector<lower=0>[W] week_eff;
    vector[C] base_covid_eff;
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu;
    vector[C+1] covid_eff = append_row(0.0, base_covid_eff);
    mu = year_eff[I_Y] .* week_eff[I_W] + covid_eff[I_C];
}

model {
    week_eff ~ lognormal(0, 1);
    year_eff ~ normal(1400, 250);
    sigma ~ cauchy(0, 1);
    y ~ normal(mu, sigma);
    }

generated quantities {
    real total_covid_deaths = sum(covid_eff);
}
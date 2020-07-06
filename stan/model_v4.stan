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
    vector[W-1] week_eff_shock;
    vector[C] base_covid_eff;
    real<lower=0> sigma;
    real<lower=0> year_eff_0;
    vector[Y-1] year_eff_shock;
    real<lower=0> sigma_week;
    real<lower=0> sigma_year;
}

transformed parameters {

    vector<lower=0>[W] week_eff;
    vector<lower=0>[Y] year_eff;
    vector[C+1] covid_eff = append_row(0.0, base_covid_eff);

    week_eff[1] = 1.0;
    for (week in 2:W) {
        week_eff[week] = week_eff[week-1] + week_eff_shock[week-1];
    }

    year_eff[1] = year_eff_0;
    for (year in 2:Y) {
        year_eff[year] = year_eff[year-1] + year_eff_shock[year-1];
    }
}

model {
    vector[N] mu;

    sigma ~ cauchy(0, 1);
    sigma_year ~ cauchy(0,1);
    sigma_week ~ cauchy(0,1);

    week_eff_shock ~ normal(0, sigma_week);
    1.0 ~ normal(week_eff[W], sigma_week*8.25/7);
    year_eff_0 ~ cauchy(0, 1);
    year_eff_shock ~ normal(0, sigma_year);

    mu = year_eff[I_Y] .* week_eff[I_W] + covid_eff[I_C];
    y ~ normal(mu, sigma);
}

generated quantities {
    real total_covid_deaths = sum(covid_eff);
}
# corona-agnostic-impact - comparison_pkgs
This is a secondary analysis and write-up, done in 2024 with the goal of comparing three model estimation 
methods/packages: `statsmodels` for a frequentist approach, `numpyro` for a Bayesian approach, and `tensorflow` for a
Machine Learning approach.

## Main idea and setup
I choose to keep the models for all three simple to make the comparison more direct. For all three options, if you would
pick this as your single choice of method, there would be many ways to improve it (see below). Instead, I limited all 
three to the following setup:
- As explanatory variables, use only the seasonal variables for a year effect and a week effect. Don't incorporate
dynamics in these effects directly.
- Don't estimate a covid effect directly, but instead make a precovid model, and use the forecast error as 
"covid estimates" as unexplained additional deaths in those weeks that don't follow the normal patterns.

The latter I actually implement using two different estimation periods:
- One period uses everything up to week 8 of 2019 for estimation, and forecasts the next 8 weeks and the remainder of 
2019\. Since there was no covid effect in 2019, this can act as a baseline and show a prediction works reasonably well.
- Another period uses everything up to week 8 of 2020 for estimation, and makes a forecast for the next 8 weeks using
that. These weeks 9 to 16 formed the impactful first covid wave.

Compared to the `original_analysis`, more of the postmodeling-style calculations are now done in the modeling script.
This is done to allow for a generic postmodeling script: all package specific calculations/transformations are 
immediately done after the modeling fit itself, such that a single postmodeling script can do a single set of 
consistent evaluation calculations. Then, an output script joins these together across packages and creates comparison
plots.

## More details for each package

### Numpyro
Compared to the `original_analysis` approach, I had to make little changes. The most important one was to take out the
dynamics originally in the model (e.g. weekly effect of week 28 would be close to the weekly effect of week 27). Next to
that, the input data wouldn't include the covid period, so that part of the model had no use.

Given that this approach is simpler than the original approach, I have not checked simulation quality and other typical
MCMC checks: the results don't give me a reason to think they should be checked, but in an actual professional setting
this would be included of course.

### Statsmodels
Given that I was only going to use the seasonal pattern as explanatory variables, I could include those in an OLS 
directly, even though the kpi is technically a time series (see below for more discussion on that). The year and week
indicators were converted to dummies, and the first week is dropped for identification purposes.

I directly use the fitted model for prediction without doing model iterations that are normal in a frequentist approach,
to make the comparison with the other approaches more fair.

### Tensorflow
In continuation of the desire to make the comparison fair across approaches, I have setup the TF model with a final
layer that is a Distribution layer using a Normal distribution. This means that in principle the same error distribution
is being minimized. But to also allow Tensorflow to show it's strong sides, I have not just estimated regression 
parameters for the yearly or weekly effect. Instead, I plug the year and week data in raw (so not as dummies, but as
year 12, week 14, or year 5, week 23), and use a Dense layer with 64 units to find a suitable transformation of this.
The number 64 is chosen such that there is enough room for flexible patterns and interactions, while still limiting
overfitting due to it being lower than the sum of weeks and years (52 + about 25 depending on period). This means it is
both more flexible and more restricted compared to the others.

Note that I have also tested models with more units or more layers, which didn't improve the forecast error much but
were more prone to overfitting. In a Tensorflow-focused setup you would use Early Stopping to limit overfitting, but in
this setup the above was sufficient.


## Overview

This table shows a comparison of the three methods in setup, and in how I experienced setting the development
environment up. Based on these experiences, next time I would design a setup that allows different setups (Docker 
images, virtualenvs, Pipenvs) for each approach to avoid conflicts. This might be a future project. :)

|                         |       Statsmodels       |         Numpyro         |     Tensorflow / Keras     |
|:------------------------|:-----------------------:|:-----------------------:|:--------------------------:|
| Filosophy               |       Frequentist       |        Bayesian         |      Machine-learning      |
| Decomposition year/week |     Linear/additive     |     Multiplicative      |         Black-box          |
| Error distribution      | Normal - fixed variance | Normal - fixed variance | Normal - variable variance |
| Main dependencies       |         Pandas          |           JAX           |           Numpy            |
| Install size            |          Small          |     Small to medium     |            Huge            |
| Estimation time         |      Insignificant      |        Moderate         |     Long wait required     |
| Flexibility             |         Limited         |          High           |       Extremely high       |


## Output
By running the project in full, you will get output plots showing various comparisons: 
1. RMSE, both in-sample and out of sample, for both estimation periods
2. R-squared, in-sample for both estimation periods
3. Estimations of the additional deaths due to covid, by week and in total
4. Comparison plots of the actual kpi and the model fits for the entire estimation period up to 2020 week 8
5. Comparison plots of the actual kpi and the model fits/forecasts, zoomed in on both forecast periods 
(2019 from week 9, and 2020 from week 9)

In future I might make a nice PDF of these results for a single KPI, but right now, I plan to move to a different 
project next. So as it is right now, I leave it to the reader to (1) clone this repo, (2) setup the right development 
environment, (3) source the data from CBS, and (4) run the entire project.
This will give some appreciation for the work. :)

The findings you should get are the following:
- The in-sample fit across both estimation periods and all three methods/packages are basically the same.
- The forecast error for 2019 is in line with the model error for all three packages
  - The forecast error is higher for tensorflow though
- The estimated covid effect shows the same pattern for all three packages (and in line with the one in
`original_analysis`), although the tensorflow effect seems to be a bit lower than the others.
- All three methods show a similar correspondence in seasonal patterns when looking at the kpi vs fit plots
- Zooming in on the covid period, it seems the tensorflow fits and forecasts are too high both just before covid and
when it just started. This results in a lower estimated covid effect, even starting negatively.
  - My theory is that the pattern tensorflow picks up on for the most recent years does not hold up towards january and
february of 2020. The other two approaches are less prone to finding such a trend and erroneously extrapolating it.
  - Fortunately, the other two approaches have a forecast error close to 0 (within error standard deviation range) for 
the weeks before covid, and also for the first weeks when the explosive spread still had to occur.

In any case, the total amount of additional unexplained deaths in the first 8 weeks of Covid in the Netherlands across
all ages are 'between 7500 and 8000' for both `numpyro` and `statsmodels` and 'around 6500' for `tensorflow` - way 
higher than would be realistic under patterns up to 2020 week 8 (in other words: very "statistically significant").

## Discussion and improvements
Some general and package-specific comments on this analysis

### General
- Like for the `original_analysis`, the goal of the model setup is to not impose a covid effect, but show that the
patterns in death counts before covid were consistent up to then and not consistent with these covid weeks.
- In case the goal is to more accurately predict death counts, multiple things could be done to enhance the models:
  - Incorporate time series dynamics in the model, ARIMA-style
  - Incorporate dynamics in the yearly and weekly effects, to (partially) pool the estimations
  - For the weekly effects, take weather patterns into account, since those are the main determining factor in the
seasonal effect within a year. For forecasting this of course means you would also need weather forecasts. :)
  - For the covid period, use patterns within the unexplained part to estimate a ramp-up, peak, or decline in these
otherwise unexplained deaths.

### Statsmodels and other frequentist approaches
- The main reason to use something like Statsmodels is because it fits the bill for what you need, and you immediately
get all these extra outputs (model evaluation metrics, tests, et cetera).
- The downside is less flexibility for model structure (such as dynamics in your yearly effects).
- Of course, there is always the option of coding in your own ML estimation (that is the OG ML, Maximum Likelihood, not 
this Machine Learning stuff ;) ), using whatever optimizer is suitable for that.
  - But are you nowadays then not better off just using Tensorflow, Pytorch or Jax instead?
- Technically, the data we model here is a time series, for which Statsmodels offers various ARIMA variants (either
using state-space models, likelihood estimations, or simple least-squares estimations).
  - If I would continue this research with just Statsmodels, I would aim to choose between those options instead of
sticking to just OLS
- Especially for someone like me (with an Econometrics background), setting up initial models with Statsmodels feels
safe and allows me to focus on understanding the patterns. Later steps can then include translating this to a Bayesian
or Machine Learning setup for additional flexibility, if needed.
- Other frequentist approaches exist as well of course, although none in Python are as good in my opinion.

### Numpyro and other Bayesian approaches
- The main reason to use Numpyro is for finding custom, flexibly-defined patterns in your data that you still want to
understand yourself (so the end goal is still Human Learning, not Machine Learning).
  - Extra more reasons if you already have some prior expectation on what these patterns might look like - another
benefit of being Bayesian.
- Performance of Numpyro compared to competitors is great imo, even on a CPU instead of GPU.
- To incorporate the Time Series aspect more, the `original_analysis` includes some of the ideas of that already, both
in that Numpyro Model setup, and even more in the older Stan models. The main gist is to use a RW in both the year and
week effects, and also potentially in a modeled covid effect (if that's part of the estimation period).
- The kpi input can be toggled to estimate by agegroup (or gender) instead of modeling all deaths. The Numpyro model
could relatively easily be extended to estimate the different agegroups at once, with a hierarchy over the agegroup-
specific year and week effects.
- There are no automated test statistics included in your posterior, but they sort-of are: the great thing of a fully
simulated posterior is that you can use it to generate any related distribution that you want. In this context: if you
want a distribution on the total estimated covid effect across all these 8 weeks, you can get that, and with that you
can construct a point estimate, intervals, or whatever you want, without a lot of extra effort.
- There are plenty of Bayesian alternatives in Python compared to Numpyro:
  - Pyro, similar in syntax but based on Pytorch instead of JAX (and in my experience, a lot slower)
  - Stan, using a C-style model structure, included in the `original_analysis`. I have found it to be slower as well, 
and I prefer the Python syntax of (Num)pyro to make it more integrated with the rest of the code.
  - I have written my own MH sampler during my PhD days: fun to do but not smart to use in production
  - Other alternatives: PyMC3 (can do discrete variables!)

### Tensorflow and other neural network approaches
- The main reason to use Tensorflow is that you don't have to define the year and week dynamics: it finds it itself
  (with the risk of overfitting).
- Additionally, it is impressive that you can do this in Tensorflow/Keras even though it is not aimed at just these
probabilistic type of models. So you can use your same skills in structuring a model for a regression model as well as
for your audio or image classification problem.
- But if your goal is to specialize in regression (and regression-style analyses, so in the broadest sense), I would
advise to also learn non-Machine Learning approaches: the algorithms are optimized towards the goal, and the outputs
are enriched with additional statistical output, or you can easily calculate that yourself.
- If I still would continue with Tensorflow in this research, I would look into Time Series specific models, such as
using LSTM cells to recover long-term patterns as well as short-term dynamics in the time series. I actually would be
curious how such LSTM cells would respond to the covid-affected increased deaths.
- Alternatives to Tensorflow for neural networks are Pytorch and JAX (and you can also mingle them).
  - I have only used both of these in the context of the Pyro and Numpyro projects, either for Bayesian simulation
models using HMC-NUTS or for approximations thereof using SVI, but not for neural networks. I might look into that in
the future, especially because I very much disliked working with Tensorflow compared to other packages in a broader
sense: a lot of documentation was outdated, wrong, or badly formatted (PEP-8 is a thing you know), and there is too much
terminal output that you should just ignore - which I don't feel great about.


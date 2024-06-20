# corona-agnostic-impact
A project to estimate the effect of "corona" (as it was referred to the most in Dutch; in English better known as covid)
on the weekly death counts in the Netherlands, by making no assumptions for it being there (so the **impact** is 
estimated in a **corona-agnostic** way).

To run this, you need to get data from the CBS (the Dutch statistics office) and put it in `data/raw/` under the name 
`Overledenen__geslacht_en_leeftijd__per_week_{date}_{time}.csv`.

Note that this has not been tested with data after October 2020.

## Main idea
For the period up to 2020, the death counts are assumed to be made up of a weekly and a yearly component:
- the weekly component reflects weather patterns throughout a year, e.g. it being cold in the winter which will lead to
more deaths, especially among the elderly
- the yearly component reflects medical advancements over time

Then, starting from a specified start date (which is the 9th week of 2020 in the code) there is a weekly additional
effect estimated. This weekly effect should be 0 (or not significantly different from 0) if there is no effect of covid.

Currently, Numpyro is used to estimate this model, but Stan code to do a similar model (including some variants with
extra assumptions) is also included.

## Output
By running this project, we can get output like in [example_output.pdf](example_output.pdf). This shows the posterior 
distribution of additional deaths for the first 32 weeks following week 9, in this case for people aged 65 to 80. 
It can be seen that there was a clear surplus in deaths for the weeks 2 to 9, which was not accounted for by normal 
weekly and yearly patterns. After that, the deaths mostly follow their normal pattern again, so the estimated effects 
are roughly 0, although at the end you can see the start of the second wave appearing. This aligns with the pattern on 
https://www.worldometers.info/coronavirus/country/netherlands/ 

The conclusion of this is that there was a statistically significant increase in the amount of deaths during the first 
wave of the covid crisis. It was not the case that "people who were already dying were being attributed as covid 
deaths", as some people suggested back in 2020.

## Note
In case commits show up to be a bit weird, I had initially set this up with a temporary email address (since this was 
not aimed to be public). On my Github mirror, this address is associated with another account.
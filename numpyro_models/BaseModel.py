"""
Description:
------------
Author: Bert de Bruijn
Date: 22-04-21

"""

# Pyro imports
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


class BaseModel:

    # Initializing the data
    def __init__(self, sdata, allow_dynamics=False):
        # Converting sdata input to model fields
        self.I_W = sdata['I_W']
        self.I_Y = sdata['I_Y']
        self.I_C = sdata['I_C']
        self.mu_y = sdata['mu_y']
        self.sigma_y = sdata['sigma_y']
        self.sigma_w = sdata['sigma_w']
        self.sigma_c = sdata['sigma_c']
        self.sigma_s = sdata['sigma_s']

        # Adding dimensions and names
        self.N = self.I_W.shape[0]
        self.W = self.I_W.max() + 1
        self.Y = self.I_Y.max() + 1
        self.C = self.I_C.max() + 1

        self.allow_dynamics = allow_dynamics
    
    # The model definition itself
    def model(self, y=None):

        if not self.allow_dynamics:
            prior_year_eff = dist.Normal(self.mu_y, self.sigma_y).expand((self.Y,))
            year_eff = numpyro.sample('year_eff', prior_year_eff)
        else:
            prior_first_year_eff = dist.HalfNormal(self.sigma_y).expand((1,))
            first_year_eff = numpyro.sample('first_year_eff', prior_first_year_eff)
            prior_base_year_eff = dist.Normal(0, 100).expand((self.Y-1,))
            base_year_eff = numpyro.sample('year_eff', prior_base_year_eff)
            year_eff = jnp.concatenate([first_year_eff + self.mu_y, base_year_eff]).cumsum()

        prior_sigma_w = dist.HalfNormal(self.sigma_w)
        sigma_w = numpyro.sample('sigma_w', prior_sigma_w)

        if not self.allow_dynamics:
            prior_week = dist.Normal(1, self.sigma_w).expand((self.W,))
            week_eff = numpyro.sample('week_eff', prior_week)
        else:
            prior_week = dist.Normal(0, sigma_w).expand((self.W-1,))
            base_week_eff = numpyro.sample('week_eff', prior_week)
            week_eff = jnp.concatenate([jnp.ones([1]), base_week_eff]).cumsum()

        if self.C > 0:
            prior_covid_eff = dist.Normal(0, self.sigma_c).expand((self.C-1,))
            base_covid_eff = numpyro.sample('covid_eff', prior_covid_eff)
            covid_eff = jnp.concatenate([jnp.zeros([1]), base_covid_eff])
        else:
            covid_eff = jnp.zeros([1,])

        prior_sigma = dist.HalfNormal(self.sigma_s)
        sigma = numpyro.sample('sigma', prior_sigma)

        mu = year_eff[self.I_Y] * week_eff[self.I_W] + covid_eff[self.I_C]

        with numpyro.plate('data', size=self.N):
            numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

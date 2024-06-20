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
    """
    This class contains all functions for creating a base Pyro model.
    It takes as input:
        'sdata': A dictionary containing:
            'x_c': Pandas DataFrame for the control variables
            'x_m': Pandas DataFrame for the media variables
            'y': Pandas Series with the KPI values
            'weight': Pandas Series with the weights
        'priors': A dictionary that contains the location and scale parameters of the priors.
            The format is a dictionary containing a dictionary: {'parameter_1': {'loc': value, 'scl': value}, ... etc.},
            where the parameters are vectors with length of the parameter.
            For instance: { 'alpha': {'loc': 0.0, 'scl': 1.0}, 
                            'beta': {'loc': [0.0]*n_ctrl, 'scl': [1.0]*n_ctrl}, 
                            'potential': {'loc': [0.0]*n_media, 'scl': [0.1]*n_media}, 
                            'sigma': {'loc': 1.0, 'scl': 1.0}}
    """
    
    # Initializing the data
    def __init__(self, sdata):
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
    
    # The model definition itself
    def model(self, y=None):

        # prior_year_eff = dist.Normal(self.mu_y, self.sigma_y).expand((self.Y,))
        # year_eff = numpyro.sample('year_eff', prior_year_eff)

        prior_first_year_eff = dist.HalfNormal(self.mu_y, self.sigma_y).expand((1,))
        first_year_eff = numpyro.sample('first_year_eff', prior_first_year_eff)
        prior_base_year_eff = dist.Normal(0, 100).expand((self.Y-1,))
        base_year_eff = numpyro.sample('year_eff', prior_base_year_eff)

        prior_sigma_w = dist.HalfNormal(self.sigma_w)
        sigma_w = numpyro.sample('sigma_w', prior_sigma_w)
        prior_week = dist.Normal(0, sigma_w).expand((self.W-1,))
        base_week_eff = numpyro.sample('week_eff', prior_week)

        prior_covid_eff = dist.Normal(0, self.sigma_c).expand((self.C-1,))
        base_covid_eff = numpyro.sample('covid_eff', prior_covid_eff)

        prior_sigma = dist.HalfNormal(self.sigma_s)
        sigma = numpyro.sample('sigma', prior_sigma)

        covid_eff = jnp.concatenate([jnp.zeros([1]), base_covid_eff])
        week_eff = jnp.concatenate([jnp.ones([1]), base_week_eff]).cumsum()
        year_eff = jnp.concatenate([first_year_eff, base_year_eff]).cumsum()

        mu = year_eff[self.I_Y] * week_eff[self.I_W] + covid_eff[self.I_C]

        with numpyro.plate('data', size=self.N):
            numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

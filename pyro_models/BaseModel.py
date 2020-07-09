"""
Description:
------------
Author: Bert de Bruijn, based on work by Mark den Hollander
Date: 22-04-21

"""

# Abstract base class imports
from abc import ABC, abstractmethod

# Pyro imports
import torch
import pyro
import pyro.distributions as dist


class BaseModel(ABC):
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
    def __init__(self, sdata, priors):
        # Converting Pandas DataFrames to Torch.tensors
        self.I_W = torch.tensor(sdata['I_W'].values)
        self.I_Y = torch.tensor(sdata['I_Y'].values)
        self.I_C = torch.tensor(sdata['I_C'].values)

        # # Converting prior lists to Torch.tensors
        # self.priors = {par:  {k: torch.tensor(v) for k, v in priors[par].items()} for par in priors.keys()}

        # Adding dimensions and names
        self.N = self.I_W.shape[0]
        self.W = self.I_W.max()
        self.Y = self.I_Y.max()
        self.C = self.I_C.max()  # TODO: might be '+1'
    
    # Sample the intercept, controls and scale parameter from their priors
    @abstractmethod
    def sample_priors(self, parameters):
        # Here: define priors for year_eff, week_eff, base_covid_eff, sigma
        # Year_eff: halfcauchy(0,1)
        # Week_eff: lognormal(0,1)
        # Sigma: halfcauchy(0,1)
        # Base_covid_eff: cauchy(0,1)

        prior_year_eff = dist.HalfCauchy(1).expand(torch.Size([self.Y]))
        year_eff = pyro.sample('year_eff', prior_year_eff)

        prior_covid_eff = dist.Cauchy(0, 1).expand(torch.Size([self.C]))
        base_covid_eff = pyro.sample('base_covid_eff', prior_covid_eff)

        prior_sigma = dist.HalfCauchy(1)
        sigma = pyro.sample('sigma', prior_sigma)

        parameters.update({'sigma': sigma, 'year_eff': year_eff})
        return parameters



    # Transform the parameters
    @abstractmethod
    def transform_parameters(self, parameters):
        # Here: calculate covid_eff as extension of base_covid_eff
        pass  # Should be implemented by subclasses
    
    # Combine all parameters in one dictionary
    def get_all_parameters(self):
        
        parameters = {}
        parameters = self.sample_prior_control(parameters)
        parameters = self.sample_prior_media(parameters)
        parameters = self.transform_parameters(parameters)
        
        return parameters

    # Compute the location of the kpi
    @abstractmethod  # Should be implemented by subclasses
    def get_kpi_parameters(self, parameters, y=None):

        mu = parameters['year_eff'] * parameters['week_eff'] + parameters['covid_eff']
        sigma = parameters['sigma']
        
        return mu, sigma
    
    # The model definition itself
    @abstractmethod  # Should be implemented by subclasses
    def model(self, y=None, seed=None):

        if seed:
            pyro.set_rng_seed(seed)

        y = torch.tensor(y.values)
        
        parameters = self.get_all_parameters()
        mu, sigma = self.get_kpi_parameters(parameters, y=y)
        
        with pyro.plate('data'):
            pyro.sample('y', dist.Normal(mu, sigma), obs=y)

"""
Description:
------------
Author: Bert de Bruijn, based on work by Mark den Hollander
Date: 22-04-21

"""

# Motherclass imports
from pyro_models.BaseModel import BaseModel

# Pyro imports
import torch
import pyro
import pyro.distributions as dist


class TMRModel(BaseModel):
    """
    This class contains all functions for creating a base TMR model.
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
        # Call __init__() of super class
        super().__init__(sdata, priors)

    
    # Sample the intercept, controls and scale parameter from their priors
    def sample_prior_control(self, parameters):
        
        prior_alpha = dist.Normal(self.priors['alpha']['loc'], self.priors['alpha']['scl'])
        alpha = pyro.sample('alpha', prior_alpha)
        
        prior_beta = dist.Normal(self.priors['beta']['loc'], self.priors['beta']['scl'])
        beta = pyro.sample('beta', prior_beta).reshape([1, self.n_ctrl])
        
        prior_sigma = dist.HalfCauchy(self.priors['sigma']['loc'], self.priors['sigma']['scl'])
        sigma = pyro.sample('sigma', prior_sigma)
        
        parameters.update({'alpha': alpha, 'beta': beta, 'sigma': sigma})
        return parameters
    
    # Sample the media parameters from their priors
    def sample_prior_media(self, parameters):
        
        prior_potential = dist.Normal(self.priors['potential']['loc'], self.priors['potential']['scl'])
        r_potential = pyro.sample('r_potential', prior_potential).reshape([1, self.n_media])
        
        prior_speed = dist.LogNormal(self.priors['speed']['loc'], self.priors['speed']['scl'])
        r_speed = pyro.sample('r_speed', prior_speed).reshape([1, self.n_media])
        
        parameters.update({'r_potential': r_potential, 'r_speed': r_speed})
        return parameters
    
    # Transform the parameters
    def transform_parameters(self, parameters):
        
        f_potential = parameters['r_potential'].clone()  # No transformation currently, but can be added at a later moment
        f_speed = parameters['r_speed'].clone().exp()

        parameters.update({'f_potential': f_potential, 'f_speed': f_speed})
        return parameters

    def compute_media_effect(self, parameters):

        return super().compute_media_effect(parameters)
    
    # Compute the location of the kpi
    def get_kpi_parameters(self, parameters, y=None):
        
        return super().get_kpi_parameters(parameters)
    
    # The pyro_models model
    def model(self, y=None, seed=None):

        if seed:
            pyro.set_rng_seed(seed)

        y = torch.tensor(y.values)
        
        parameters = self.get_all_parameters()
        y_loc, y_scl = self.get_kpi_parameters(parameters, y=y)
        
        with pyro.plate('data'):
            pyro.sample('y', dist.Normal(y_loc, y_scl), obs=y)
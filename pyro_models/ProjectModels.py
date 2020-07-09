"""
Description:
------------
Author: Bert de Bruijn
Date: 22-04-21

"""

# Motherclass imports
from model_classes.TMRModel import TMRModel

# Pyro imports
import pyro
import pyro.distributions as dist


class Model_v01(TMRModel):

    # Initializing the data
    def __init__(self, sdata, priors):
        # Call __init__() of super class
        super().__init__(sdata, priors)


    # Sample the media parameters from their priors
    def sample_prior_media(self, parameters):

        prior_potential = dist.LogNormal(self.priors['potential']['loc'], self.priors['potential']['scl'])
        r_potential = pyro.sample('r_potential', prior_potential).reshape([1, self.n_media])

        prior_speed = dist.LogNormal(self.priors['speed']['loc'], self.priors['speed']['scl'])
        r_speed = pyro.sample('r_speed', prior_speed).reshape([1, self.n_media])

        parameters.update({'r_potential': r_potential, 'r_speed': r_speed})
        return parameters


class Model_v02(Model_v01):

    def __init__(self, sdata, priors):
        # Call __init__() of super class
        super().__init__(sdata, priors)


    def sample_prior_media(self, parameters):
        prior_potential = dist.LogNormal(self.priors['potential']['loc'], self.priors['potential']['scl'])
        r_potential = pyro.sample('r_potential', prior_potential).reshape([1, self.n_media])

        prior_speed = dist.LogNormal(self.priors['speed']['loc'], self.priors['speed']['scl'])
        r_speed = pyro.sample('r_speed', prior_speed).reshape([1, 1])

        parameters.update({'r_potential': r_potential, 'r_speed': r_speed})
        return parameters

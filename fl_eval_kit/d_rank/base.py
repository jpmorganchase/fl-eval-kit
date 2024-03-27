# ECIR 2024: Ranking Distance Metric for Privacy Budget in Distributed Learning of Finite Embedding Data
# Georgios Papadopoulos, Yash Satsangi, Shaltiel Eloul, Marco Pistoia
# JPMorgan Chase

from abc import ABC
import numpy as np

class BaseDP(ABC):
    def __init__(self, distribution: int="gaussian",
                 loc: float=.0, scale: float=1.0,
                 random_key: int=None):

        self.distribution = distribution
        self.loc = loc
        self.scale = scale
        self.random_key= random_key if random_key else 147


    def add_noise_to_gradients(self, gradients):
        """
        Add noise to gradients using NumPy

        Args:
            gradients: A list of the gradients to which noise will be added.
            noise_scale (float): The scale of the noise to be added.
            noise_type (str): Type of noise to add ('gaussian' or 'laplacian').

        Returns:
            list of the gradients with noise added.
        """
        rng = np.random.default_rng(self.random_key)

        noised_gradients = []
        for grad in gradients:
            grad_shape = grad.shape
                
            if self.distribution == 'gaussian':
                noise = rng.normal(loc=self.loc, scale=self.scale, size=grad_shape)
            elif self.distribution == 'laplacian':
                noise = rng.laplace(self.loc, self.scale, size=grad_shape)
            else:
                raise ValueError("Invalid noise type. Choose 'gaussian' or 'laplacian'.")

            noised_grad = grad + noise

            noised_gradients.append(noised_grad)

        return noised_gradients
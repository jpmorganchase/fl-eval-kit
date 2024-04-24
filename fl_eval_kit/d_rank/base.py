# ECIR 2024: Ranking Distance Metric for Privacy Budget in Distributed Learning of Finite Embedding Data
# Georgios Papadopoulos, Yash Satsangi, Shaltiel Eloul, Marco Pistoia
# JPMorgan Chase, 2024


import numpy as np

class BaseDP:
    def __init__(self, distribution: int="gaussian",
                 loc: float=.0, scale: float=1.0,
                 random_key: int=None):
        self.distribution = distribution
        self.loc = loc
        self.scale = scale
        self.random_key= random_key if random_key else 147

    def new_noise_seed(self,new_random_key=None):
        if new_random_key:
            self.random_key = new_random_key
        else:
            np.random.seed = self.random_key
            self.random_key =np.random.randint(low=0,high=99999999999)


    def add_noise_to_gradients(self, gradients):
        """
        Add noise to gradients using NumPy

        Args:
            params: A list of the gradients vectors to which noise will be added.
            noise_scale (float): The scale of the noise to be added.
            noise_type (str): Type of noise to add ('gaussian' or 'laplacian').

        Returns:
            list of the gradients with noise added.
        """
        rng = np.random.default_rng(self.random_key)

        noised_gradients = []
        for grads in params:
            grad_shape = grads.shape
                
            if self.distribution == 'gaussian':
                noise = self.rng.normal(loc=self.loc, scale=self.scale, size=grad_shape)
            elif self.distribution == 'laplacian':
                noise = self.rng.laplace(self.loc, self.scale, size=grad_shape)
            else:
                raise ValueError("Invalid noise type. Choose 'gaussian' or 'laplacian'.")

            noised_grad = grads + noise

            noised_gradients.append(noised_grad)

        return noised_gradients

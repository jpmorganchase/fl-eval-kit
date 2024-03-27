# ECIR 2024: Ranking Distance Metric for Privacy Budget in Distributed Learning of Finite Embedding Data
# Georgios Papadopoulos, Yash Satsangi, Shaltiel Eloul, Marco Pistoia
# JPMorgan Chase

import logging
import numpy as np
import scipy
from typing import List
from fractions import Fraction
from fl_eval_kit.d_rank.base import BaseDP
import torch

def mean_absolute_difference(word_gradients, original_gradient):
    mean_diffs = []
    for word_grad, orig_grad in zip(word_gradients, original_gradient):
        if word_grad.shape != orig_grad.shape:
            raise ValueError("Mismatched gradient shapes")
        diff = torch.abs(word_grad - orig_grad)
        mean_diff = torch.mean(diff).item()
        mean_diffs.append(mean_diff)

    return np.mean(mean_diffs)

class NoiseDP(BaseDP):
    def __init__(self, distribution: str="gaussian",
                 loc: float=0, scale: float=1, random_key: int=None):
        super().__init__(distribution, loc, scale, random_key)

    def update(self, weights):
        return self.add_noise_to_gradients(weights)

    @staticmethod
    def calculate_epsilon(batch_size_k, scale):
        """
        :param B: The number of batches
        :param b: The scale of the distribution (standard deviation)
        :return:
        """

        epsilon = 2.0 / (batch_size_k * scale)

        return epsilon

    def calculate_d_rank(self, gradient_data: List[str], batch_size_k: int,
                         vocab: List[str], add_noise: bool=True,
                         noise_type='repeating', return_embedding=False):
        """
        :param gradient_data: The input text to add noise and calculate the d_rank value
        :param batch_size_k: The size of the batch to mix
        :param vocab: The vocabulary of the embedding model
        :param add_noise: If we want to add noise or not
        :param noise_type: The type of noise we add
        :return:
        """

        grad_list = []
        vocab_median_length = int(len(vocab)/2)

        original_gradient = gradient_data.copy()
        original_grad_distances = [mean_absolute_difference(gradients, original_gradient)
                                for gradients in vocab.values()]
        closest_original_grad_index_vocab = original_grad_distances.index(np.min(original_grad_distances))
        original_token_string = list(vocab)[closest_original_grad_index_vocab]

        if (add_noise == True) and (noise_type != 'repeating'):
            private_grad = self.update(original_gradient) # add noise

        elif (add_noise == True) and (noise_type == 'repeating'):

            f = Fraction(batch_size_k).limit_denominator()
            weighted_original_grad = f.denominator * original_gradient
            private_grad = self.update(weighted_original_grad)  # add noise only once


            for batch_index in range(f.numerator):
                private_grad += original_gradient

        grad_list.append(private_grad)

        priv_grad_distances = [mean_absolute_difference(gradients, private_grad)
                                   for gradients in vocab.values()]
        noisy_token_distances_dictionary = dict(zip(vocab, priv_grad_distances))
        sorted_noisy_token_dict = dict(sorted(noisy_token_distances_dictionary.items(), key= lambda item: item[1]))
        index_original_token = list(sorted_noisy_token_dict).index(original_token_string)


        average_d_rank_sentence = np.mean(index_original_token)
        average_d_rank_sentence = np.clip(average_d_rank_sentence, 0, vocab_median_length)


        if return_embedding:
            return 1 - (1 - 2*average_d_rank_sentence/len(vocab)), grad_list
        else:
            return 1 - (1 - 2*average_d_rank_sentence/len(vocab))


    def optimise_noisy_embedding(self, user_d_rank_threshold: float,
                                 gradient_data: List[str],
                                 batch_size_k: int,
                                 vocab: List[str], noise_type='repeating',
                                 min_noise=0, max_noise=200):

        print("This algorithm modifies the scale of the NoiseDP class. To reset you need to set the class again.")

        self.scale = min_noise
        fa = self.calculate_d_rank(gradient_data=gradient_data,
                                   batch_size_k=batch_size_k,
                                   vocab=vocab, add_noise=True,
                                   noise_type=noise_type, return_embedding=False)

        print(fa)

        self.scale = max_noise
        fb = self.calculate_d_rank(gradient_data=gradient_data,
                                   batch_size_k=batch_size_k,
                                   vocab=vocab, add_noise=True,
                                   noise_type=noise_type, return_embedding=False)
        print(fb)

        frange =(fb-fa)


        root = (user_d_rank_threshold-fa)/frange
        if root > fb:
            logging.warning("d_rank value results in too high noise, taking max_noise instead and d_rank is: "+str(fb))
            return max_noise
        if user_d_rank_threshold < fa:
            logging.warning("calc. d_rank is already bigger than threshold")
            return 0.0


        def d_rank_diff_function_noise(noise_level):

            self.scale = noise_level
            computed_d_rank = self.calculate_d_rank(gradient_data=gradient_data,
                                                    batch_size_k=batch_size_k,
                                                    vocab=vocab, add_noise=True,
                                                    noise_type=noise_type, return_embedding=False)

            diff =  (computed_d_rank-fa)/frange - root

            return diff

        # We use the bisect method to optimise it
        optimised_noise = scipy.optimize.bisect(d_rank_diff_function_noise, min_noise, max_noise, xtol=0.01, rtol=0.1)

        return optimised_noise

# ECIR 2024: Ranking Distance Metric for Privacy Budget in Distributed Learning of Finite Embedding Data
# Georgios Papadopoulos, Yash Satsangi, Shaltiel Eloul, Marco Pistoia
# JPMorgan Chase, 2024

import logging
import time
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

        #mean_diff = torch.mean(diff).item()
        mean_diffs.extend(diff.flatten())

    return np.mean(mean_diffs)

def average_gradients(grad_dict):
    sum_grads = None
    n = len(grad_dict)

    j = 0
    for key in grad_dict:
        j+=1
        grads = grad_dict[key]

        if sum_grads is None:
            sum_grads = [torch.zeros_like(g) for g in grads]

        for i, grad in enumerate(grads):
            sum_grads[i] += grad


        avg_grads = [g / j for g in sum_grads]
        diff = mean_absolute_difference(avg_grads, grad_dict[0])
        print('mixture of: ', j)
        print(diff)


    avg_grads = [g / n for g in sum_grads]

    return avg_grads

def rank_calculation(A, B):
    result = {}
    for key in B:
        # A is the noisy rank of words
        # B is the original rank of words
        first_key_of_A = next(iter(A[key])) # Find first word in private_words_ranked_distances
        keys_of_B = list(B[key].keys()) # List the words in the words_ranked_distances
        index = keys_of_B.index(first_key_of_A) # Find the index of the noisy A to the original B
        result[key] = index

    return result

class DrankDP(BaseDP):
    def __init__(self, vocab, gradient_vocab, distribution: str="gaussian",
                 loc: float=0, scale: float=1, random_key: int=None):
        self.vocab = vocab
        self.gradient_vocab = gradient_vocab

        super().__init__(distribution, loc, scale, random_key)
        self.new_noise_seed()

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

    def calculate_d_rank(self,
                         inputs,
                         zero_gradient_vector: List[float],
                         add_noise: bool=True,
                         noise_type='none', return_embedding=False,
                         mixing_type='both'):
        """
        :param gradient_data: The input text to add noise and calculate the d_rank value
        :param batch_size_k: The size of the batch to mix
        :param vocab: The vocabulary of the embedding model
        :param add_noise: If we want to add noise or not
        :param noise_type: The type of noise we add
        :return:
        """

        print("token_gradients_in_sentence")
        token_gradients_in_sentence = {}
        # We take the tokens of the FIRST sentence
        # and we asign them their gradients (we already have them from the gradient_vocab
        for i, token in enumerate(inputs[0]):
            if token==0:
                token_gradients_in_sentence[i] = zero_gradient_vector.copy()
            else:
                word = list(self.vocab.keys())[list(self.vocab.values()).index(token)]
                token_gradients_in_sentence[i] = self.gradient_vocab[word].copy()

        print("words_ranked_distances")
        # this takes quite some time actually
        words_ranked_distances = {}
        # Take the tokens-gradient list of the first sentence
        # and rank it, using mse from lower to higher distance
        for k, tkn_grd in token_gradients_in_sentence.items():
            grad_distances = [mean_absolute_difference(gradients, tkn_grd)
                              for gradients in self.gradient_vocab.values()]
            distances_vector = dict(zip(self.vocab, grad_distances))
            sorted_distances_vector = dict(sorted(distances_vector.items(), key= lambda item: item[1]))
            words_ranked_distances[k] = sorted_distances_vector.copy()

        print("d-rank calculation for each token")
        private_token_gradients_in_sentence = {}
        token_ranks = {}
        # Now add to the token-gradients of the first sentence the batch and noise
        # In this example the noise is 0 so we only mix the token gradients with the batch gradient
        for k, tk_grd in token_gradients_in_sentence.items():# k loop through the tokens
            noisy_grad = self.update(tk_grd)
            # Take the gradient of each tok
            j=0
            token_ranks_for_each_cum_batch = []
            token_noisy_gradients = {}

            if mixing_type=='both':
                token_ranks_for_each_cum_batch_sum = []
                token_ranks_for_each_cum_batch_avg = []

            for j in range(inputs.shape[0]-1):# j loop through the batch
                j+=1
                print("the token ", k, "out of ", len(token_gradients_in_sentence), "; and the batch ", j, " out of ", inputs.shape[0])
                # Check if token=0

                batch_tkn = inputs[j][k]
                if batch_tkn==0:
                    batch_token_grad = zero_gradient_vector.copy()
                else:
                    word = list(self.vocab.keys())[list(self.vocab.values()).index(batch_tkn)]
                    batch_token_grad = self.gradient_vocab[word].copy()
                if j==1:
                    gradient_sum = [a + b for a, b in zip(noisy_grad, batch_token_grad)]
                else:
                    gradient_sum = [a + b for a, b in zip(gradient_sum, batch_token_grad)]

                token_noisy_gradients[j] = gradient_sum.copy()
                # Here, we want to find the drank for each extra batch addition
                # to the original token noisy_grad
                avg_grads = [g / (j+1) for g in gradient_sum]
                if mixing_type=='both':
                    grad_distances_sum = [mean_absolute_difference(gradients, gradient_sum)
                                          for gradients in self.gradient_vocab.values()]
                    distances_vector_sum = dict(zip(self.vocab, grad_distances_sum))
                    sorted_distances_vector_sum = dict(sorted(distances_vector_sum.items(), key= lambda item: item[1]))
                    first_key_of_noisy_vector = next(iter(sorted_distances_vector_sum)) # Find first word in private_words_ranked_distances
                    keys_of_vocabulary = list(words_ranked_distances[k].keys()) # List the words in the words_ranked_distances
                    index = keys_of_vocabulary.index(first_key_of_noisy_vector) # Find the index of the noisy A to the original B
                    token_ranks_for_each_cum_batch_sum.append(index)

                    grad_distances_avg = [mean_absolute_difference(gradients, avg_grads)
                                          for gradients in self.gradient_vocab.values()]
                    distances_vector_avg = dict(zip(self.vocab, grad_distances_avg))
                    sorted_distances_vector_avg = dict(sorted(distances_vector_avg.items(), key= lambda item: item[1]))
                    first_key_of_noisy_vector = next(iter(sorted_distances_vector_avg)) # Find first word in private_words_ranked_distances
                    keys_of_vocabulary = list(words_ranked_distances[k].keys()) # List the words in the words_ranked_distances
                    index = keys_of_vocabulary.index(first_key_of_noisy_vector) # Find the index of the noisy A to the original B
                    token_ranks_for_each_cum_batch_avg.append(index)

                elif mixing_type=='sum':
                    grad_distances = [mean_absolute_difference(gradients, gradient_sum)
                                          for gradients in self.gradient_vocab.values()]
                    distances_vector = dict(zip(self.vocab, grad_distances))
                    sorted_distances_vector = dict(sorted(distances_vector.items(), key= lambda item: item[1]))
                    first_key_of_noisy_vector = next(iter(sorted_distances_vector)) # Find first word in private_words_ranked_distances
                    keys_of_vocabulary = list(words_ranked_distances[k].keys()) # List the words in the words_ranked_distances
                    index = keys_of_vocabulary.index(first_key_of_noisy_vector) # Find the index of the noisy A to the original B
                    token_ranks_for_each_cum_batch.append(index)

                else:
                    grad_distances = [mean_absolute_difference(gradients, avg_grads)
                                          for gradients in self.gradient_vocab.values()]
                    distances_vector = dict(zip(self.vocab, grad_distances))
                    sorted_distances_vector = dict(sorted(distances_vector.items(), key= lambda item: item[1]))

                    first_key_of_noisy_vector = next(iter(sorted_distances_vector)) # Find first word in private_words_ranked_distances
                    keys_of_vocabulary = list(words_ranked_distances[k].keys()) # List the words in the words_ranked_distances
                    index = keys_of_vocabulary.index(first_key_of_noisy_vector) # Find the index of the noisy A to the original B
                    token_ranks_for_each_cum_batch.append(index)


                # compare noisy grad with original
                sum_diff = mean_absolute_difference(tk_grd, gradient_sum)
                print('sum diff:', sum_diff)
                mean_diff = mean_absolute_difference(tk_grd, avg_grads)
                print('avg diff:', mean_diff)

            if mixing_type=='both':

                for a in range(len(token_ranks_for_each_cum_batch_sum)):
                    token_ranks_for_each_cum_batch.append((token_ranks_for_each_cum_batch_sum[a] + token_ranks_for_each_cum_batch_avg[a])/2)

            private_token_gradients_in_sentence[k] = token_noisy_gradients
            token_ranks[k] = token_ranks_for_each_cum_batch

        max_rank_values_per_token = [np.max(token_ranks[k]) for k in token_ranks.keys()]
        max_rank_index_per_token = [np.argmax(token_ranks[k])+1 for k in token_ranks.keys()] # I need it to find the gradient
        list_of_private_max_gradients = [private_token_gradients_in_sentence[tk][max_rank_index_per_token[tk]] for tk, tk_gr_dict in private_token_gradients_in_sentence.items()]
        average_d_rank_sentence = np.mean(np.array(max_rank_values_per_token))
        drank = 1 - (1 - average_d_rank_sentence/len(self.vocab)) # I removed the 2 because for this specific the average rank is above the median of the vocab.

        if return_embedding:
            return drank, token_gradients_in_sentence, list_of_private_max_gradients
        else:
            return drank

    def optimise_noisy_embedding_vector(self, inputs,
                                 zero_gradient_vector,
                                 user_d_rank_threshold=0.8,
                                 noise_type='none',
                                 mixing_type='sum',
                                 min_noise=0, max_noise=200):

        list_of_tokens_in_sentence = [t for ls in inputs for t in ls if t!=0]
        print('batch token volume:', len(list_of_tokens_in_sentence))



        start = time.time()
        optim_noise = self.optimise_noisy_embedding(user_d_rank_threshold=user_d_rank_threshold,
                                                    inputs=inputs,
                                                    zero_gradient_vector=zero_gradient_vector,
                                                    noise_type=noise_type,
                                                    mixing_type=mixing_type,
                                                    min_noise=min_noise, max_noise=max_noise)
        end = time.time()
        print("elapsed time (hrs):", (end - start)/60/60)
        #     optim_noise_list.append(optim_noise)
        #

        self.scale=optim_noise
        drank, token_gradients_in_sentence, list_of_private_max_gradients = self.calculate_d_rank(inputs,
                                                        zero_gradient_vector=zero_gradient_vector,
                                                        add_noise=True,
                                                        noise_type=noise_type, return_embedding=True,
                                                        mixing_type=mixing_type)
        print("Noise to add: {0} for D-rank: {1}".format(self.scale, drank ))
        return drank, token_gradients_in_sentence, list_of_private_max_gradients

    def optimise_noisy_embedding(self, user_d_rank_threshold: float,
                                 inputs,
                                 zero_gradient_vector,
                                 noise_type='none',
                                 mixing_type='sum',
                                 min_noise=0, max_noise=200):

        self.scale = min_noise
        fa = self.calculate_d_rank(inputs,
                                   zero_gradient_vector=zero_gradient_vector,
                                   add_noise=True,
                                   noise_type=noise_type, return_embedding=False,
                                   mixing_type=mixing_type)

        print("fa-drank:", fa)

        self.scale = max_noise
        fb = self.calculate_d_rank(inputs,
                                   zero_gradient_vector=zero_gradient_vector,
                                   add_noise=True,
                                   noise_type=noise_type, return_embedding=False,
                                   mixing_type=mixing_type)
        print("fb-drank:", fb)

        frange =(fb-fa)


        root = (user_d_rank_threshold-fa)/frange
        if root > fb:
            #logging.warning("d_rank value results in too high noise, taking max_noise instead and d_rank is: "+str(fb))
            return max_noise
        if user_d_rank_threshold < fa:
            #logging.warning("calc. d_rank is already bigger than threshold")
            return 0.0


        def d_rank_diff_function_noise(noise_level):

            self.scale = noise_level
            computed_d_rank = self.calculate_d_rank(inputs,
                                                    zero_gradient_vector=zero_gradient_vector,
                                                    add_noise=True,
                                                    noise_type=noise_type, return_embedding=False,
                                                    mixing_type=mixing_type)

            diff =  (computed_d_rank-fa)/frange - root

            return diff

        # We use the bisect method to optimise it
        optimised_noise = scipy.optimize.bisect(d_rank_diff_function_noise,
                                                min_noise, max_noise, xtol=0.01, rtol=0.01)

        return optimised_noise


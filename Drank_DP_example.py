# ECIR 2024: Ranking Distance Metric for Privacy Budget in Distributed Learning of Finite Embedding Data
# Georgios Papadopoulos, Yash Satsangi, Shaltiel Eloul, Marco Pistoia
# JPMorgan Chase, 2024

from fl_eval_kit.d_rank.noisedp import DrankDP
from example_models.torch_NLP_models import SentimentRNN
from example_models.torch_NLP_data import LoadData
from fl_eval_kit.tools import generate_vocab_gradients, get_gradients, average_gradients

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# Example parameters
batch_size = 4
v_padding = 20
no_layers = 1
embedding_dim = 8
output_dim = 1
hidden_dim = 32
drop_prob = 0.5
lr = 5e-3
epochs = 1 # 3-4 is approx where I noticed the validation loss stop decreasing
print_every = 5
clip=5 # gradient clipping

# Load the IMDB dataset
dataloader = LoadData()
train_loader, valid_loader, vocab = dataloader.preprocess_and_return_tokens(batch_size=batch_size,
                                                                            v_padding=v_padding)
vocab_size = len(vocab) + 1 #extra 1 for padding

# Load the RNN model
model = SentimentRNN(no_layers, vocab_size, hidden_dim, embedding_dim, output_dim, drop_prob=drop_prob)
model.to(device)
# Set optimizers and loss function
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
criterion = criterion.to(device)

# Create a vocabulary of words and their gradients (the gradient of the token)
gradient_vocab = generate_vocab_gradients(vocab, model, criterion, v_padding, device)

# Initialise the d-rank privacy evaluation model
drank_dp = DrankDP(vocab, gradient_vocab)


# Start ML model training
model.train()

counter = 0



for e in range(epochs):

    for inputs, labels in train_loader:
        #
        #
        print('batch gradient')
        model.zero_grad()
        h = model.init_hidden(batch_size, device)
        gradient_batch = get_gradients(inputs, h, labels, model, criterion)
        #
        print('gradient of first sentence')
        model.zero_grad()
        h = model.init_hidden(1, device)
        first_sentence = torch.reshape(inputs[0], (1, -1)).detach().clone()
        private_gradient_vector = get_gradients(first_sentence, h, labels[:1], model, criterion)

        print('0 gradient')
        model.zero_grad()
        zero_inputs = torch.zeros((1, v_padding)).long()
        h = model.init_hidden(1, device)
        dummy_label = torch.tensor([0.])
        zero_gradient_vector = get_gradients(zero_inputs, h, dummy_label, model, criterion)

        print('gradient of each sentence')
        gradient_sentences = {}
        for s in range(inputs.shape[0]):

            model.zero_grad()
            h = model.init_hidden(1, device)
            sentence = torch.reshape(inputs[s], (1, -1)).detach().clone()
            gradient_batch_single = get_gradients(sentence, h, labels[s], model, criterion)
            gradient_sentences[s] = gradient_batch_single


        print("average grads")
        average_grads = average_gradients(gradient_sentences)

        # avoid bisect and calculate by the distance needs to be added. 
        drank, token_gradients_in_sentence, list_of_private_max_gradients = drank_dp.optimise_noisy_embedding_vector(
            inputs,
            zero_gradient_vector,
            user_d_rank_threshold=0.80,
            noise_type='none',
            mixing_type='sum',
            min_noise=0, max_noise=800) # probably I will need like 900 for 1 d-rank

        dict_private_grads = dict(enumerate(list_of_private_max_gradients))
        private_average_token_grads = average_gradients(dict_private_grads)

        # this will be used to infuse it into the model gradient
        print('pause')
        drank_dp.new_noise_seed()

        # set modified gradients to the model
        for param, new_grad in zip(model.parameters(), private_average_token_grads):
            new_grad_converted = new_grad.to(dtype=param.dtype)
            param.grad = new_grad_converted.clone()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if counter % print_every == 0:
            val_h = model.init_hidden(batch_size, device)
            val_losses = []
            model.eval()
            for inputs, labels in valid_loader:

                val_h = tuple([each.data for each in val_h])
                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
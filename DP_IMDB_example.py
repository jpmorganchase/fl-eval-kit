# ECIR 2024: Ranking Distance Metric for Privacy Budget in Distributed Learning of Finite Embedding Data
# Georgios Papadopoulos, Yash Satsangi, Shaltiel Eloul, Marco Pistoia
# JPMorgan Chase

from fl_eval_kit.d_rank.noisedp import NoiseDP
from example_models.torch_NLP_models import SentimentRNN
from example_models.torch_NLP_data import LoadData

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
batch_size = 1
v_padding = 500
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
gradient_vocab = {}
vocab_batch_size = 1
v_h = model.init_hidden(vocab_batch_size, device)
dummy_label = torch.tensor([0.])

for word, token_id in vocab.items():

    v_inputs = torch.zeros((1, v_padding)).long()
    v_inputs[0, -1] = token_id
    v_h = tuple([each.data for each in v_h])

    model.zero_grad()
    v_output, v_h = model(v_inputs, v_h)
    loss = criterion(v_output, dummy_label)
    loss.backward()

    gradients = []
    for param in model.parameters():
        gradients.append(param.grad.clone())

    # Store gradients
    gradient_vocab[word] = gradients


# Start model training
model.train()

counter = 0

for e in range(epochs):

    h = model.init_hidden(batch_size, device)

    for inputs, labels in train_loader:
        counter += 1
        h = tuple([each.data for each in h])

        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        gradient_list=[]
        for param in model.parameters():
            gradient_list.append(param.grad)

        if counter==1:
            list_of_tokens_in_sentence = [t for t in inputs[0, :] if t!=0]
            optim_noise_list = []

            for t in list_of_tokens_in_sentence:
                word = list(vocab.keys())[list(vocab.values()).index(t)]
                single_word_grad = gradient_vocab[word]

                privacy_model = NoiseDP()
                optim_noise = privacy_model.optimise_noisy_embedding(user_d_rank_threshold=0.96,
                                                       gradient_data=single_word_grad,
                                                       batch_size_k=10, vocab=gradient_vocab,
                                                       noise_type='repeating',
                                                       min_noise=0, max_noise=200)
                optim_noise_list.append(optim_noise)

            average_optim_noise = np.mean(optim_noise_list)

        privacy_model = NoiseDP(scale=average_optim_noise)
        _, private_grad_list = privacy_model.calculate_d_rank(gradient_data=gradient_list,
                                       batch_size_k=10,
                                       vocab=gradient_vocab, add_noise=True,
                                       noise_type='repeating', return_embedding=True)


        for param, new_grad in zip(model.parameters(), private_grad_list[0]):
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
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
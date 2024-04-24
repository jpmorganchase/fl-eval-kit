
import torch

def generate_vocab_gradients(vocab, model, criterion, v_padding, device):
    # helper function to create output of grads for the vocab
    gradient_vocab = {}
    v_h = model.init_hidden(1, device)
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
    return gradient_vocab

def get_gradients(inputs, h, labels, model, criterion):
    model.zero_grad()
    output, h = model(inputs, h)
    loss = criterion(output.squeeze(), labels.float())
    loss.backward()

    # collect gradients from model
    gradient_list=[]
    for param in model.parameters():
        gradient_list.append(param.grad)
    return gradient_list
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

    avg_grads = [g / n for g in sum_grads]

    return avg_grads


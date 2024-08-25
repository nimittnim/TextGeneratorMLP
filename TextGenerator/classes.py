####################### Importing Libraries ################
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt 
from pprint import pprint
import random
import torch._dynamo
torch._dynamo.config.suppress_errors = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time




########################### Model ##############################
class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = self.lin2(x)
    return x

class NextCharDense(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size, hidden_size_2):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, hidden_size_2)
    self.lin3 = nn.Linear(hidden_size_2, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = self.lin2(x)
    x = torch.sin(x)
    x = self.lin3(x)
    return x


########################## training ############################

# Train the model

def train(model, X, Y, opt, model_path):
    loss_fn = nn.CrossEntropyLoss()
    # Mini-batch training
    batch_size = 10000
    print_every = 100
    elapsed_time = []
    for epoch in range(10000):
        start_time = time.time()
        for i in range(0, X.shape[0], batch_size):
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        end_time = time.time()
        elapsed_time.append(end_time - start_time)
        if epoch % print_every == 0:
            print(epoch, loss.item())


        if (epoch % 100 == 0):
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': loss,
                        }, model_path)


########################## generate text ########################


def generate_text(seed, model, prompt, itos, stoi, block_size, max_len=50):
    g = torch.Generator()
    g.manual_seed(seed)
    context = []
    for j in range(len(prompt)):
        context = context + [stoi[prompt[j]]]
    if len(context) > block_size:
        context = context[-block_size:]
    if len(context) < block_size:
        while(len(context)!=block_size):
            context.insert(0,0)
        
    name = ''
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)

        k = 34
        top_k_values, top_k_indices = torch.topk(y_pred, k)
        ix = top_k_indices[k-1].item()
        # ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item() 
        ch = itos[ix]
        # if ch == '~':
        #     break
        name += ch
        context = context[1:] + [ix]
    return name

######################## Testing ################################


# Load checkpoints
def load_check_points(model, opt, model_path):
    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

def print_text(text):
    for chr in text:
        if chr == '~':
            print("\n",end='')
        else:
            print(chr,end="")
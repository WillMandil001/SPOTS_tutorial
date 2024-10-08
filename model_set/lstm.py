import torch
import torch.nn as nn
from torch.autograd import Variable


class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, device):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.device = device
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            # nn.BatchNorm1d(output_size),
            nn.Tanh())

    def init_hidden(self, batch_size):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).to(self.device)),
                           Variable(torch.zeros(batch_size, self.hidden_size).to(self.device))))
        return hidden

    def forward(self, input):
        batch_size = input.size(0)
        self.hidden = self.init_hidden(batch_size)

        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)


class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, device):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.device = device
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).to(self.device)),
                           Variable(torch.zeros(batch_size, self.hidden_size).to(self.device))))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        batch_size = input.size(0)
        self.hidden = self.init_hidden(batch_size)
      
        embedded = self.embed(input.view(-1, self.input_size))  # input = [32,32]
        h_in = embedded  # embedded = [32,256]
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)  # h_in = [32,256]
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)  # mu = [32,10], logvar = [32,10]
        return z, mu, logvar  # z = [32, 10]

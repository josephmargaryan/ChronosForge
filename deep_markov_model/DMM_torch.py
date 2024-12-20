import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def sequence_mask(seq_len, max_len=None):
    """
    Convert sequence lengths to masking vectors
    """
    if max_len is None:
        max_len = seq_len.data.max()
    batch_size = seq_len.size(0)
    seq_range = torch.arange(0, max_len, dtype=torch.long, device=seq_len.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_len_expand = seq_len.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_len_expand


def reverse_sequence(seqs, seq_lens):
    """
    this function takes a torch mini-batch and reverses each sequence(w.r.t. the temporal axis, i.e. axis=1)
    in contrast to `reverse_sequences_numpy`, this function plays nice with torch autograd
    """
    batch_size, max_seq_len, dim = seqs.size()
    rev_seqs = seqs.new_zeros(seqs.size())
    for b in range(batch_size):
        T = seq_lens[b]
        time_slice = torch.arange(T - 1, -1, -1, device=seqs.device)
        rev_seq = torch.index_select(seqs[b, :, :], 0, time_slice)
        rev_seqs[b, 0:T, :] = rev_seq
    return rev_seqs


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        arch,
        output_size,
        activation=nn.ReLU(),
        batch_norm=True,
        init_w=0.02,
        discriminator=False,
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_w = init_w

        layer_sizes = [input_size] + [int(x) for x in arch.split("-")]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)
            if batch_norm and not (
                discriminator and i == 0
            ):  # if used as discriminator, then there is no batch norm in the first layer
                bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)
            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], output_size)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, self.init_w)
                layer.bias.data.fill_(0)
            except:
                pass


class Encoder(nn.Module):
    def __init__(
        self,
        embedder,
        input_size,
        hidden_size,
        bidir,
        n_layers,
        dropout=0.5,
        noise_radius=0.2,
    ):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.noise_radius = noise_radius
        self.n_layers = n_layers
        self.bidir = bidir
        assert type(self.bidir) == bool
        self.dropout = dropout

        self.embedding = embedder  # nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(
            input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidir
        )
        self.init_h = nn.Parameter(
            torch.randn(self.n_layers * (1 + self.bidir), 1, self.hidden_size),
            requires_grad=True,
        )  # learnable h0
        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():  # initialize the gate weights with orthogonal
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, inputs, input_lens=None, init_h=None, noise=False):
        # init_h: [n_layers*n_dir x batch_size x hid_size]
        if self.embedding is not None:
            inputs = self.embedding(
                inputs
            )  # input: [batch_sz x seq_len] -> [batch_sz x seq_len x emb_sz]

        batch_size, seq_len, emb_size = inputs.size()
        inputs = F.dropout(inputs, self.dropout, self.training)  # dropout

        if input_lens is not None:  # sort and pack sequence
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)
            inputs = pack_padded_sequence(
                inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True
            )

        if init_h is None:
            init_h = self.init_h.expand(
                -1, batch_size, -1
            ).contiguous()  # use learnable initial states, expanding along batches
        # self.rnn.flatten_parameters() # time consuming!!
        hids, h_n = self.rnn(inputs, init_h)  # hids: [b x seq x (n_dir*hid_sz)]
        # h_n: [(n_layers*n_dir) x batch_sz x hid_sz] (2=fw&bw)
        if input_lens is not None:  # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(
            self.n_layers, (1 + self.bidir), batch_size, self.hidden_size
        )  # [n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1]  # get the last layer [n_dirs x batch_sz x hid_sz]
        enc = (
            h_n.transpose(0, 1).contiguous().view(batch_size, -1)
        )  # [batch_sz x (n_dirs*hid_sz)]
        # if enc.requires_grad:
        #    enc.register_hook(self.store_grad_norm) # store grad norm
        # norms = torch.norm(enc, 2, 1) # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        # enc = torch.div(enc, norms.unsqueeze(1).expand_as(enc)+1e-5)
        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(
                means=torch.zeros(enc.size(), device=inputs.device),
                std=self.noise_radius,
            )
            enc = enc + gauss_noise

        return enc, hids


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, trans_dim):
        super(GatedTransition, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(z_dim, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, z_dim),
            nn.Sigmoid(),
        )
        self.proposed_mean = nn.Sequential(
            nn.Linear(z_dim, trans_dim), nn.ReLU(), nn.Linear(trans_dim, z_dim)
        )
        self.z_to_mu = nn.Linear(z_dim, z_dim)
        # modify the default initialization of z_to_mu so that it starts out as the identity function
        self.z_to_mu.weight.data = torch.eye(z_dim)
        self.z_to_mu.bias.data = torch.zeros(z_dim)
        self.z_to_logvar = nn.Linear(z_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        gate = self.gate(z_t_1)  # compute the gating function
        proposed_mean = self.proposed_mean(z_t_1)  # compute the 'proposed mean'
        mu = (1 - gate) * self.z_to_mu(
            z_t_1
        ) + gate * proposed_mean  # compute the scale used to sample z_t, using the proposed mean from
        logvar = self.z_to_logvar(self.relu(proposed_mean))
        epsilon = torch.randn(
            z_t_1.size(), device=z_t_1.device
        )  # sampling z by re-parameterization
        z_t = mu + epsilon * torch.exp(0.5 * logvar)  # [batch_sz x z_sz]
        return z_t, mu, logvar


class PostNet(nn.Module):
    """
    Parameterizes `q(z_t|z_{t-1}, x_{t:T})`, which is the basic building block of the inference (i.e. the variational distribution).
    The dependence on `x_{t:T}` is through the hidden state of the RNN
    """

    def __init__(self, z_dim, h_dim):
        super(PostNet, self).__init__()
        self.z_to_h = nn.Sequential(nn.Linear(z_dim, h_dim), nn.Tanh())
        self.h_to_mu = nn.Linear(h_dim, z_dim)
        self.h_to_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, z_t_1, h_x):
        """
        Given the latent z at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t|z_{t-1}, x_{t:T})`
        """
        h_combined = 0.5 * (
            self.z_to_h(z_t_1) + h_x
        )  # combine the rnn hidden state with a transformed version of z_t_1
        mu = self.h_to_mu(h_combined)
        logvar = self.h_to_logvar(h_combined)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn(
            z_t_1.size(), device=z_t_1.device
        )  # sampling z by re-parameterization
        z_t = epsilon * std + mu  # [batch_sz x z_sz]
        return z_t, mu, logvar


class DHMM(nn.Module):
    """
    The Deep Markov Model
    """

    def __init__(self, config):
        super(DHMM, self).__init__()
        self.input_dim = config["input_dim"]
        self.z_dim = config["z_dim"]
        self.emission_dim = config["emission_dim"]
        self.trans_dim = config["trans_dim"]
        self.rnn_dim = config["rnn_dim"]
        self.clip_norm = config["clip_norm"]

        self.emitter = nn.Sequential(  # Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(self.z_dim, self.emission_dim),
            nn.ReLU(),
            nn.Linear(self.emission_dim, self.emission_dim),
            nn.ReLU(),
            nn.Linear(self.emission_dim, self.input_dim),
            nn.Sigmoid(),
        )
        self.trans = GatedTransition(self.z_dim, self.trans_dim)
        self.postnet = PostNet(self.z_dim, self.rnn_dim)
        self.rnn = Encoder(None, self.input_dim, self.rnn_dim, False, 1)
        # nn.RNN(input_size=self.input_dim, hidden_size=self.rnn_dim, nonlinearity='relu', \
        # batch_first=True, bidirectional=False, num_layers=1)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn_dim))

        self.optimizer = Adam(
            self.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"])
        )

    def kl_div(self, mu1, logvar1, mu2=None, logvar2=None):
        one = torch.ones(1, device=mu1.device)
        if mu2 is None:
            mu2 = torch.zeros(1, device=mu1.device)
        if logvar2 is None:
            logvar2 = torch.zeros(1, device=mu1.device)
        return torch.sum(
            0.5
            * (
                logvar2
                - logvar1
                + (torch.exp(logvar1) + (mu1 - mu2).pow(2)) / torch.exp(logvar2)
                - one
            ),
            1,
        )

    def infer(self, x, x_rev, x_lens):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        batch_size, _, x_dim = x.size()
        T_max = x_lens.max()
        h_0 = self.h_0.expand(1, batch_size, self.rnn.hidden_size).contiguous()

        _, rnn_out = self.rnn(
            x_rev, x_lens, h_0
        )  # push the observed x's through the rnn;
        rnn_out = reverse_sequence(
            rnn_out, x_lens
        )  # reverse the time-ordering in the hidden state and un-pack it
        rec_losses = torch.zeros((batch_size, T_max), device=x.device)
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        z_prev = self.z_q_0.expand(
            batch_size, self.z_q_0.size(0)
        )  # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
        for t in range(T_max):
            z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)  # p(z_t| z_{t-1})
            z_t, z_mu, z_logvar = self.postnet(
                z_prev, rnn_out[:, t, :]
            )  # q(z_t | z_{t-1}, x_{t:T})
            kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
            kl_states[:, t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
            logit_x_t = self.emitter(z_t).contiguous()  # p(x_t|z_t)
            rec_loss = nn.BCEWithLogitsLoss(reduction="none")(
                logit_x_t.view(-1), x[:, t, :].contiguous().view(-1)
            ).view(batch_size, -1)
            rec_losses[:, t] = rec_loss.mean(dim=1)
            z_prev = z_t
        x_mask = sequence_mask(x_lens)
        x_mask = x_mask.gt(0).view(-1)
        rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
        kl_loss = kl_states.view(-1).masked_select(x_mask).mean()
        return rec_loss, kl_loss

    def train_AE(self, x, x_rev, x_lens, kl_anneal):
        self.rnn.train()  # put the RNN back into training mode (i.e. turn on drop-out if applicable)

        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        loss = rec_loss + kl_anneal * kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        self.optimizer.step()

        return {"train_loss_AE": loss.item(), "train_loss_KL": kl_loss.item()}

    def valid(self, x, x_rev, x_lens):
        self.eval()
        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        loss = rec_loss + kl_loss
        return loss

    def generate(self, x, x_rev, x_lens):
        """
        Generation model: p(x_{1:T} | z_{1:T}), p(z_{1:T})
        """
        batch_size, _, x_dim = x.size()  # Number of time steps in the mini-batch
        T_max = x_lens.max()
        z_prev = self.z_0.expand(
            batch_size, self.z_0.size(0)
        )  # Set z_prev=z_0 to initialize p(z_t|z_{t-1})
        generated_sequences = []  # List to store generated x_t sequences

        for t in range(T_max):
            # Sample z_t ~ p(z_t | z_{t-1}) one time step at a time
            z_t, _, _ = self.trans(z_prev)  # p(z_t | z_{t-1})
            p_x_t = torch.sigmoid(
                self.emitter(z_t)
            )  # Compute the probabilities that parameterize the Bernoulli likelihood
            x_t = torch.bernoulli(p_x_t)  # Sample x_t ~ p(x_t | z_t)
            generated_sequences.append(x_t)
            z_prev = z_t  # Update z_prev for the next time step

        # Stack generated sequences into a tensor of shape [batch_size, T_max, x_dim]
        return torch.stack(generated_sequences, dim=1)


def test():
    import torch

    # Configuration for DHMM
    config = {
        "input_dim": 10,  # Observation dimension
        "z_dim": 5,  # Latent state dimension
        "emission_dim": 20,  # Emission network hidden dimension
        "trans_dim": 20,  # Transition network hidden dimension
        "rnn_dim": 15,  # RNN hidden dimension
        "clip_norm": 5.0,  # Gradient clipping norm
        "lr": 0.001,  # Learning rate
        "beta1": 0.9,  # Adam optimizer beta1
        "beta2": 0.999,  # Adam optimizer beta2
    }

    # Generate dummy data
    batch_size = 4
    sequence_length = 10
    input_dim = config["input_dim"]

    x = torch.rand(batch_size, sequence_length, input_dim)  # Observed data
    x_rev = torch.flip(x, dims=[1])  # Reverse sequence for RNN processing
    x_lens = torch.randint(
        low=5, high=sequence_length + 1, size=(batch_size,)
    )  # Variable sequence lengths

    # Initialize the DHMM
    dhmm = DHMM(config)

    # Train the model
    num_epochs = 50
    kl_anneal = 0.1  # Annealing factor for KL divergence

    print("Starting training...")
    for epoch in range(num_epochs):
        metrics = dhmm.train_AE(x, x_rev, x_lens, kl_anneal)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Total Loss = {metrics['train_loss_AE']:.4f}, KL Loss = {metrics['train_loss_KL']:.4f}"
            )

    # Validate the model
    print("Validating...")
    val_loss = dhmm.valid(x, x_rev, x_lens)
    print(f"Validation Loss: {val_loss.item():.4f}")

    # Generate data
    print("Generating sequences...")
    generated_x = dhmm.generate(x, x_rev, x_lens)

    print("Generated sequences:")
    print(generated_x)


if __name__ == "__main__":
    test()

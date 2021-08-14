import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


class TrajectoryGenerator2d:

    def __init__(self, height, width):
        self.height = height
        self.width = width

        self.action_offsets = torch.tensor([
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
            [0, 0]
        ])

        map_positions = []
        map = torch.zeros(height*width, height, width)
        for y in range(height):
            for x in range(width):
                i = y*width + x
                map[i, y, x] = 1
                pos = torch.tensor([[y, x]])
                map_positions.append(pos)

        self.map = map
        self.map_positions = torch.cat(map_positions)

    def generate_batch(self, batch_size):
        start_indices = torch.randint(0, len(self.map_positions), [batch_size])
        x = self.map[start_indices]

        source_positions = self.map_positions[start_indices]
        
        actions = torch.randint(0, len(self.action_offsets), [batch_size])

        target_positions = source_positions.add(self.action_offsets[actions])

        # replace all invalid actions with 0 action
        replacement_action = len(self.action_offsets) - 1
        invalid_idx = torch.logical_or(
            torch.logical_or(target_positions[:, 0] < 0, target_positions[:, 1] < 0),
            torch.logical_or(target_positions[:, 0] >= self.height, target_positions[:, 1] >= self.width)
        )
        actions[invalid_idx] = replacement_action

        target_positions = source_positions.add(self.action_offsets[actions])
        target_indices = target_positions[:, 0] * self.width + target_positions[:, 1]

        y = self.map[target_indices]
    
        return x, y, actions


class Mlp(nn.Module):
    def __init__(self, layer_sizes, activation='lrelu'):
        super().__init__()

        layers = []
        num_layers = len(layer_sizes)-1
        if num_layers < 1:
            raise RuntimeError('at least input and output layer size have to be specified')

        for i in range(num_layers):
            if i > 0:
                if activation == 'lrelu':
                    layers.append(nn.LeakyReLU(0.2))
                else:
                    layers.append(nn.Tanh())
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DistModel(nn.Module):
    def __init__(self, input_size, factor):
        super().__init__()

        a = factor
        self.encoder = Mlp([input_size, 128*a, 32*a, 16*a, 2], activation='tanh')
        self.decoder = Mlp([2, 32*a, 128*a, 64*a, input_size], activation='tanh')
        self.action_encoder = Mlp([5, 32, 16, 2], activation='tanh')

    def forward(self, x, a):
        input_shape = x.shape
        x = x.view(input_shape[0], -1)

        y = self.encoder(x)

        a_ = F.one_hot(a, num_classes=5).float()
        
        z = self.action_encoder(a_)
        
        y = y + z
        
        di = y  # decoder input
        
        y = self.decoder(y)

        y = y.view(input_shape)
        
        return y, di


def plot(z, az, i):
    z = z.detach()
    az = az.detach()    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(z[:,0], z[:,1], 'o', color='green')

    b = z[0:1] + az
    ax.plot(b[:,0], b[:,1], 'x', color='red')

    c = z[10:11] + az
    ax.plot(c[:,0], c[:,1], 'x', color='blue')

    fig.savefig('step_{:06}.png'.format(i), format='png')
    plt.close(fig)


def main():
    print(torch.__version__)

    w,h = 4,4
    model_size_factor = 1
    learning_rate = 1e-4

    """
    w,h = 6,6
    model_size_factor = 5
    learning_rate = 5e-5
    """

    g = TrajectoryGenerator2d(h, w)
    model = DistModel(g.width * g.height, model_size_factor)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.MSELoss(reduction='sum')

    batch_size = 250
    plot_interval = 1000
    print_interval = 100

    for step in range(1, 20000):
        x, y, a = g.generate_batch(batch_size)
        y_, di = model.forward(x, a)
        prediction_loss = loss_fn(y_, y)    # reconstruction loss

        # auto encoder loss (output == input)
        x2 = g.map.clone().view(g.width*g.height, -1)
        z = model.encoder.forward(x2)
        y2_ = model.decoder.forward(z)
        auto_encoder_loss = 5 * loss_fn(x2, y2_)        

        # next step consistency loss
        z = model.encoder.forward(y.view(y.shape[0], -1))
        consistency_loss = loss_fn(z, di)

        loss = prediction_loss + consistency_loss + auto_encoder_loss

        if step % print_interval == 0:
            print("{}: loss: {:.2f}; prediction: {:.2f}; consistency: {:.2f};".format(step, loss.item(), prediction_loss.item(), consistency_loss.item()))

        if step % plot_interval == 0:
            test_input = g.map.clone().view(g.width*g.height, -1)
            state_projections = model.encoder.forward(test_input)

            a_ = F.one_hot(torch.arange(5), num_classes=5).float()
            action_offset = model.action_encoder(a_)
            plot(state_projections, action_offset, step)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()

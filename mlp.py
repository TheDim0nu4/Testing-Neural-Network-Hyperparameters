from torch import nn



class MLP(nn.Module):

    def __init__(self, input_size, topology, output_size, activation_func=nn.ReLU):

        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        for count_neurons in topology:

            layers.append(nn.Linear(prev_size, count_neurons))
            layers.append(activation_func())
            prev_size = count_neurons


        layers.append(nn.Linear(topology[-1], output_size))
        self.model = nn.Sequential(*layers)


    def forward(self, x):

        return self.model(x)

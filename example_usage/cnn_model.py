import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, neighbor_count:int=3):
        super(Model, self).__init__()
        self.neighbor_count = neighbor_count
        self.neighborhood_size = 2 * self.neighbor_count + 1
        # reconstruction":
        # in n, neighborhood_size, 512, 512
        # out n, 512, 512

        self.conv1 = nn.Conv2d(in_channels=self.neighborhood_size, 
                               out_channels=128, 
                               kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=128, 
                               out_channels=64, 
                               kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, 
                               out_channels=self.neighborhood_size, 
                               kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=self.neighborhood_size, 
                               out_channels=1, 
                               kernel_size=3, padding=1)
        nn.init.uniform_(self.conv1.weight, a=-0.1, b=0.1)
        nn.init.zeros_(self.conv1.bias)
        nn.init.uniform_(self.conv2.weight, a=-0.1, b=0.1)
        nn.init.zeros_(self.conv2.bias)
        nn.init.uniform_(self.conv3.weight, a=-0.1, b=0.1)
        nn.init.zeros_(self.conv3.bias)
        nn.init.uniform_(self.conv4.weight, a=-0.1, b=0.1)
        nn.init.zeros_(self.conv4.bias)

    def forward(self, x):
        # x shape: (n, neighborhood_size, 512, 512) or (neighborhood_size, 512, 512)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # x shape: (n, neighborhood_size, 512, 512)
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.view(x.shape[0], *x.shape[-2:])
        
        if self.training:
            return x
        else:
            return torch.clamp(x, 0, 1)

def test_run():
    model = Model()
    model.to("cuda:0")
    x = torch.randn(2, 7, 512, 512)
    y = model(x.to("cuda:0"))
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

if __name__ == "__main__":
    test_run()
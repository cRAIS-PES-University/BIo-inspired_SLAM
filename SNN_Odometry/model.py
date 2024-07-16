import snntorch as snn
from snntorch import spikegen
import torch
from torch import nn
from base import MetaTensor, TensorLayout, DataType

class EnhancedVelocityEstimationSNN(nn.Module):
    def __init__(self, num_channels_event=2, num_channels_flow=2):
        super().__init__()
        self.num_channels_event = num_channels_event
        self.num_channels_flow = num_channels_flow
        self.beta = 0.5  # Example decay factor for phase 2

        # Define layers for event data
        self.conv1_event = nn.Conv2d(num_channels_event, 16, 3, padding=1)
        self.lif1_event = snn.Leaky(beta=0.9)
        self.conv2_event = nn.Conv2d(16, 32, 3, padding=1)
        self.lif2_event = snn.Leaky(beta=0.9)
        self.pool_event = nn.MaxPool2d(2)

        # Define layers for optical flow data
        self.conv1_flow = nn.Conv2d(num_channels_flow, 16, 3, padding=1)
        self.lif1_flow = snn.Leaky(beta=0.9)
        self.conv2_flow = nn.Conv2d(16, 32, 3, padding=1)
        self.lif2_flow = snn.Leaky(beta=0.9)
        self.pool_flow = nn.MaxPool2d(2)

        # Integration and adaptive layer
        self.conv1_integration = nn.Conv2d(64, 64, 3, padding=1)  # Combines features from event and flow
        self.lif1_integration = snn.Leaky(beta=0.9)
        self.conv2_integration = nn.Conv2d(64, 128, 3, padding=1)
        self.lif2_integration = snn.Leaky(beta=0.9)
        self.pool_integration = nn.MaxPool2d(2)

        # More layers can be added as necessary
        self.fc1 = nn.Linear(128 * 15 * 15, 512)
        self.lif_fc1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(512, 2)  # Adjust size accordingly

    def forward(self, x_event, x_flow):
        x_event = MetaTensor(x_event, TensorLayout.Conv, DataType.Spike)
        x_flow = MetaTensor(x_flow, TensorLayout.Conv, DataType.Spike)

        x_event = self.lif1_event(self.conv1_event(x_event.getTensor()))
        x_event = self.lif2_event(self.conv2_event(x_event))
        x_event = self.pool_event(x_event)

        x_flow = self.lif1_flow(self.conv1_flow(x_flow.getTensor()))
        x_flow = self.lif2_flow(self.conv2_flow(x_flow))
        x_flow = self.pool_flow(x_flow)

        # Apply decay factor to flow features during phase 2
        x_flow *= self.beta

        # Combine features from both pathways
        x = torch.cat((x_event, x_flow), dim=1)
        x = self.lif1_integration(self.conv1_integration(x))
        x = self.lif2_integration(self.conv2_integration(x))
        x = self.pool_integration(x)

        x = x.view(x.size(0), -1)
        x = self.lif_fc1(self.fc1(x))
        x = self.fc2(x)

        return x
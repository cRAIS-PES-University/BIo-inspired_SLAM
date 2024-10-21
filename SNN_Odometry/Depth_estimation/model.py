import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF

class SNNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(SNNBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = snn.Leaky(beta=0.9)

        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.lif2 = snn.Leaky(beta=0.9)

        self.downsample = downsample

    def forward(self, x, mem=None):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out, mem = self.lif1(out, mem)

        out = self.conv2(out)
        out = self.bn2(out)
        out, mem = self.lif2(out, mem)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out, mem

    def reset_neurons(self):
        self.lif1.reset_mem()
        self.lif2.reset_mem()


class SNNResNetEncoder(nn.Module):
    def __init__(self, block, layers, input_channels=2):
        super(SNNResNetEncoder, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.lif1 = snn.Leaky(beta=0.9)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride !=1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.ModuleList(layers)

    def forward(self, x):
        #print(f"Input to conv1: {x.shape}")  # Debug print
        out = self.conv1(x)
        out = self.bn1(out)
        out, mem = self.lif1(out)
        #print(f"After conv1 and lif1: {out.shape}")  # Debug print
        out = self.maxpool(out)
        #print(f"After maxpool: {out.shape}")  # Debug print

        encoder_outputs = []

        # Layer 1
        for layer_idx, layer in enumerate(self.layer1):
            out, mem = layer(out, mem)
            #print(f"After layer1_{layer_idx}: {out.shape}")  # Debug print
        encoder_outputs.append(out)  # After layer1

        # Layer 2
        for layer_idx, layer in enumerate(self.layer2):
            out, mem = layer(out, mem)
            #print(f"After layer2_{layer_idx}: {out.shape}")  # Debug print
        encoder_outputs.append(out)  # After layer2

        # Layer 3
        for layer_idx, layer in enumerate(self.layer3):
            out, mem = layer(out, mem)
            #print(f"After layer3_{layer_idx}: {out.shape}")  # Debug print
        encoder_outputs.append(out)  # After layer3

        # Layer 4
        for layer_idx, layer in enumerate(self.layer4):
            out, mem = layer(out, mem)
            #print(f"After layer4_{layer_idx}: {out.shape}")  # Debug print
        encoder_outputs.append(out)  # After layer4

        return encoder_outputs  # List of 4 feature maps

    def reset_neurons(self):
        self.lif1.reset_mem()
        for layer in self.layer1:
            layer.reset_neurons()
        for layer in self.layer2:
            layer.reset_neurons()
        for layer in self.layer3:
            layer.reset_neurons()
        for layer in self.layer4:
            layer.reset_neurons()


class SNNDecoder(nn.Module):
    def __init__(self):
        super(SNNDecoder, self).__init__()
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.lif4 = snn.Leaky(beta=0.9)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.lif3 = snn.Leaky(beta=0.9)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.lif2 = snn.Leaky(beta=0.9)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.lif1 = snn.Leaky(beta=0.9)

        self.dispconv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_features):
        mem = {}
        f_e1, f_e2, f_e3, f_e4 = encoder_features  # [B,64,80,160], [B,128,40,80], [B,256,20,40], [B,512,10,20]

        x = self.upconv4(f_e4)  # [B,256,20,40]
        x, mem['lif4'] = self.lif4(x)
        #print(f"After upconv4 and lif4: {x.shape}")  # Debug print
        x = x + f_e3  # [B,256,20,40] + [B,256,20,40] = [B,256,20,40]
        #print(f"After skip connection with f_e3: {x.shape}")  # Debug print

        x = self.upconv3(x)  # [B,128,40,80]
        x, mem['lif3'] = self.lif3(x)
        #print(f"After upconv3 and lif3: {x.shape}")  # Debug print
        x = x + f_e2  # [B,128,40,80] + [B,128,40,80] = [B,128,40,80]
        #print(f"After skip connection with f_e2: {x.shape}")  # Debug print

        x = self.upconv2(x)  # [B,64,80,160]
        x, mem['lif2'] = self.lif2(x)
        #print(f"After upconv2 and lif2: {x.shape}")  # Debug print
        x = x + f_e1  # [B,64,80,160] + [B,64,80,160] = [B,64,80,160]
        #print(f"After skip connection with f_e1: {x.shape}")  # Debug print

        x = self.upconv1(x)  # [B,64,160,320]
        x, mem['lif1'] = self.lif1(x)
        #print(f"After upconv1 and lif1: {x.shape}")  # Debug print

        disparity = self.dispconv(x)  # [B,1,160,320]
        disparity = self.sigmoid(disparity)  # Normalize to [0,1]
        #print(f"Disparity Shape: {disparity.shape}")  # Debug print
        return disparity

    def reset_neurons(self):
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()
        self.lif4.reset_mem()


class SNNHybridPoseNet(nn.Module):
    def __init__(self):
        super(SNNHybridPoseNet, self).__init__()
        # Separate encoder for pose estimation
        self.encoder = SNNResNetEncoder(SNNBasicBlock, [2, 2, 2, 2], input_channels=3)

        # After concatenation, reduce channels
        self.reduce_channels = nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(512)
        self.lif_reduce = snn.Leaky(beta=0.9)

        # Adaptive pooling to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # ANN regression head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # 6-DoF pose parameters
        )

    def forward(self, I_k, I_k_prime):
        mem = {}
        #print(f"PoseNet Encoder - I_k Shape: {I_k.shape}")  # Debug print
        f_k = self.encoder(I_k)
        #print(f"PoseNet Encoder - I_k after encoder: {[out.shape for out in f_k]}")  # Debug print

        #print(f"PoseNet Encoder - I_k_prime Shape: {I_k_prime.shape}")  # Debug print
        f_k_prime = self.encoder(I_k_prime)
        #print(f"PoseNet Encoder - I_k_prime after encoder: {[out.shape for out in f_k_prime]}")  # Debug print

        # Use the last feature maps
        f_k_last = f_k[-1]      # [B,512,10,20]
        f_k_prime_last = f_k_prime[-1]  # [B,512,10,20]
        #print(f"PoseNet Encoder - f_k_last Shape: {f_k_last.shape}")  # Debug print
        #print(f"PoseNet Encoder - f_k_prime_last Shape: {f_k_prime_last.shape}")  # Debug print

        # Concatenate features
        features = torch.cat((f_k_last, f_k_prime_last), dim=1)  # [B,1024,10,20]
        #print(f"PoseNet Encoder - Features after concatenation: {features.shape}")  # Debug print

        # Reduce channels
        features = self.reduce_channels(features)  # [B,512,10,20]
        features = self.bn_reduce(features)
        features, mem['lif_reduce'] = self.lif_reduce(features)
        #print(f"PoseNet Encoder - Features after channel reduction and lif: {features.shape}")  # Debug print

        # Adaptive pooling
        features = self.pool(features)  # [B,512,1,1]
        #print(f"PoseNet Encoder - Features after pooling: {features.shape}")  # Debug print

        # Flatten and regress pose
        pose = self.fc(features)  # [B,6]
        #print(f"Pose Shape: {pose.shape}")  # Debug print
        return pose

    def reset_neurons(self):
        self.encoder.reset_neurons()
        self.lif_reduce.reset_mem()


class SNNDepthPoseEstimator(nn.Module):
    def __init__(self):
        super(SNNDepthPoseEstimator, self).__init__()
        self.encoder = SNNResNetEncoder(SNNBasicBlock, [2, 2, 2, 2], input_channels=2)
        self.decoder = SNNDecoder()
        self.pose_net = SNNHybridPoseNet()

    def forward(self, event_voxel_grid, I_k, I_k_prime):
        #print(f"SNNDepthPoseEstimator - event_voxel_grid Shape: {event_voxel_grid.shape}")  # Debug print
        # Encode event voxel grid
        encoder_outputs = self.encoder(event_voxel_grid)
        #print(f"SNNDepthPoseEstimator - encoder_outputs: {[out.shape for out in encoder_outputs]}")  # Debug print

        # Decode to get disparity
        disparity = self.decoder(encoder_outputs)
        #print(f"SNNDepthPoseEstimator - disparity Shape: {disparity.shape}")  # Debug print

        # Estimate pose
        pose = self.pose_net(I_k, I_k_prime)
        #print(f"SNNDepthPoseEstimator - pose Shape: {pose.shape}")  # Debug print

        return disparity, pose

    def reset_neurons(self):
        self.encoder.reset_neurons()
        self.decoder.reset_neurons()
        self.pose_net.reset_neurons()

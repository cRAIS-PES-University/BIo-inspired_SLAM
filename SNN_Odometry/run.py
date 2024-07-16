import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import snntorch as snn
import snntorch.functional as SF
from snntorch import spikeplot as splt
import matplotlib.pyplot as plt
from dataloader import DSECIntegratedDataset
from model import EnhancedVelocityEstimationSNN
import trace
import sys

# Configure logging
logging.basicConfig(
    filename='application.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


# Console handler for critical issues and general info
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

logging.info("Starting the application")

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 16
learning_rate = 1e-3
num_epochs = 50
time_step = 1  # Assuming 1 time unit between frames, adjust accordingly

# Load dataset

dataset = DSECIntegratedDataset(
    event_dir='/home/crais/PES1PG22EC027/DSEC',
    flow_dir='/home/crais/PES1PG22EC027/DSEC',
    transform=None
)
logging.info(f"Dataset loaded with {len(dataset)} items")

# Use the tracer to run the load_dataset function
#dataset = tracer.runfunc(dataset)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
logging.info("DataLoader setup complete")

# Initialize model
logging.debug("Initializing model")
model = EnhancedVelocityEstimationSNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Custom loss function
def self_supervised_loss(predictions, targets):
    return SF.mse_count_loss(predictions, targets)

# Function to calculate linear and angular velocity from optical flow
def calculate_velocity_from_optical_flow(flow_tensor, time_step):
    flow_x = flow_tensor[0]
    flow_y = flow_tensor[1]
    displacement_x = flow_x * time_step
    displacement_y = flow_y * time_step
    velocity_x = torch.sum(displacement_x) / time_step
    velocity_y = torch.sum(displacement_y) / time_step
    linear_velocity = torch.sqrt(velocity_x**2 + velocity_y**2).mean()
    angular_velocity = torch.atan2(velocity_y, velocity_x).mean()
    return torch.tensor([linear_velocity, angular_velocity]).to(device)

# Training loop
loss_hist = []
accuracy_hist = []

def profile(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@profile
def train_epoch(epoch, model, data_loader, optimizer):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for i, (data, target) in enumerate(data_loader):
        event_data, flow_data = data[0].to(device), data[1].to(device)
        pseudo_target = calculate_velocity_from_optical_flow(flow_data, time_step)

        optimizer.zero_grad()
        outputs = model(event_data, flow_data)
        loss = self_supervised_loss(outputs, pseudo_target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += (torch.abs(outputs - pseudo_target) < 0.1).sum().item()
        total += event_data.size(0)

    loss_hist.append(epoch_loss / len(data_loader))
    accuracy_hist.append(100 * correct / total)
    logging.debug(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")



for epoch in range(num_epochs):
    train_epoch(epoch, model, data_loader, optimizer)

logging.info("Training complete")

# Plotting loss and accuracy
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(loss_hist, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(accuracy_hist, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), 'enhanced_velocity_estimation_snn.pth')

# Optional: Visualize spikes using snntorch.spikeplot
for data, _ in data_loader:
    splt.spikeRaster(data[0].cpu().detach().numpy())
    plt.show()
    break  # Only visualize the first batch

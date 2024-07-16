import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
from utils.eventslicer import EventSlicer

class DSECIntegratedDataset(Dataset):
    def __init__(self, event_dir, flow_dir, transform=None, delta_t_ms=50):
        self.event_dir = Path(event_dir)
        self.flow_dir = Path(flow_dir)
        self.transform = transform
        self.delta_t_us = delta_t_ms * 1000
        self.height = 480
        self.width = 640

        self.valid_recordings = set([
            'thun_00_a', 'zurich_city_01_a', 'zurich_city_02_a', 'zurich_city_02_c',
            'zurich_city_02_d', 'zurich_city_02_e', 'zurich_city_03_a', 'zurich_city_05_a',
            'zurich_city_05_b', 'zurich_city_06_a', 'zurich_city_07_a', 'zurich_city_08_a',
            'zurich_city_09_a', 'zurich_city_10_a', 'zurich_city_10_b', 'zurich_city_11_a',
            'zurich_city_11_b', 'zurich_city_11_c'
        ])

        self.recordings = [d for d in self.event_dir.iterdir() if d.is_dir() and d.name in self.valid_recordings]
        print(f"Found valid recordings: {self.recordings}")

        self.flow_timestamps = {}
        for rec in self.recordings:
            fwd_timestamp_path = self.flow_dir / rec.name / 'flow' / 'forward_timestamps.txt'
            bwd_timestamp_path = self.flow_dir / rec.name / 'flow' / 'backward_timestamps.txt'
            if fwd_timestamp_path.exists():
                self.flow_timestamps[rec.name + '_forward'] = np.loadtxt(fwd_timestamp_path, delimiter=',')
            if bwd_timestamp_path.exists():
                self.flow_timestamps[rec.name + '_backward'] = np.loadtxt(bwd_timestamp_path, delimiter=',')
        print(f"Loaded flow timestamps: {self.flow_timestamps.keys()}")

        self.event_slicers = {}
        self.event_files = []
        for rec in self.recordings:
            left_events = rec / 'events' / 'left' / 'events.h5'
            right_events = rec / 'events' / 'right' / 'events.h5'
            if left_events.exists():
                self.event_files.append(left_events)
                self.event_slicers[left_events] = EventSlicer(h5py.File(str(left_events), 'r'))
            if right_events.exists():
                self.event_files.append(right_events)
                self.event_slicers[right_events] = EventSlicer(h5py.File(str(right_events), 'r'))
            print(f"Checking for event files in {rec / 'events' / 'left'} and {rec / 'events' / 'right'}")
            if not left_events.exists():
                print(f"No event files found in {rec / 'events' / 'left'}")
            if not right_events.exists():
                print(f"No event files found in {rec / 'events' / 'right'}")
        print(f"Total event files found: {len(self.event_files)}")

    def __len__(self):
        return len(self.event_files)

    def __getitem__(self, idx):
        event_file = self.event_files[idx]
        rec_name = event_file.parent.parent.parent.name
        direction = 'forward' if 'left' in event_file.parent.name else 'backward'
        flow_dir = 'forward' if direction == 'forward' else 'backward'

        flow_file = self.flow_dir / rec_name / 'flow' / flow_dir / f"{str(idx).zfill(6)}.png"
        if Path(flow_file).exists():
            flow_x, flow_y, valid = self.load_optical_flow(flow_file)
            flow_tensor = self.preprocess_flow(flow_x, flow_y, valid)

        ts_end = self.flow_timestamps[f"{rec_name}_{direction}"][idx]
        ts_start = ts_end - self.delta_t_us

        events = self.event_slicers[event_file].get_events(np.asscalar(ts_start), np.asscalar(ts_end))
        spike_tensor = self.process_events(events)

        if self.transform:
            spike_tensor = self.transform(spike_tensor)
            if 'flow_tensor' in locals():
                flow_tensor = self.transform(flow_tensor)

        combined_data = torch.cat([spike_tensor, flow_tensor], dim=0) if 'flow_tensor' in locals() else spike_tensor
        proxy_target = self.generate_proxy_target(flow_tensor) if 'flow_tensor' in locals() else torch.zeros(1)
        return combined_data, proxy_target

    def load_optical_flow(self, flow_path):
        flow_img = cv2.imread(str(flow_path), cv2.IMREAD_UNCHANGED)
        flow_x = (flow_img[:, :, 2].astype(np.float32) - 2**15) / 128.0
        flow_y = (flow_img[:, :, 1].astype(np.float32) - 2**15) / 128.0
        valid = flow_img[:, :, 0] > 0
        return flow_x, flow_y, valid

    def preprocess_flow(self, flow_x, flow_y, valid):
        flow_x[~valid] = 0
        flow_y[~valid] = 0
        return torch.from_numpy(np.stack((flow_x, flow_y), axis=0))

    def process_events(self, events):
        x = events['x']
        y = events['y']
        p = events['p']
        t = events['t']
        return torch.tensor([x, y, p, t], dtype=torch.float32)





    def generate_proxy_target(self, flow_tensor, time_step):
        # Calculate displacement in the x and y directions
        displacement_x = flow_tensor[0] * time_step
        displacement_y = flow_tensor[1] * time_step
    
        # Calculate velocity components
        velocity_x = torch.sum(displacement_x) / time_step
        velocity_y = torch.sum(displacement_y) / time_step
    
        # Calculate overall velocity
        velocity = torch.sqrt(velocity_x**2 + velocity_y**2).mean()
    
        # Debugging statements
        print(f"Displacement X: {displacement_x}")
        print(f"Displacement Y: {displacement_y}")
        print(f"Velocity X: {velocity_x}")
        print(f"Velocity Y: {velocity_y}")
        print(f"Overall Velocity: {velocity}")
    
        return velocity.unsqueeze(0)



# Usage Example
dataset = DSECIntegratedDataset('/home/crais/PES1PG22EC027/train_events', '/home/crais/PES1PG22EC027/train_optical_flow')
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
for data, target in data_loader:
    print(data.shape, target.shape)  # Debugging line to check tensor shapes


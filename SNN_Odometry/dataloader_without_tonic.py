import logging
import time
from typing import Dict, Tuple, Any
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
from utils.eventslicer import EventSlicer

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def profile(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

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
        self.recordings = self.recordings[:1]  # Debugging with only the first recording
        logging.debug(f"Found valid recordings: {self.recordings}")

        self.flow_timestamps = {}
        for rec in self.recordings:
            fwd_timestamp_path = self.flow_dir / rec.name / 'flow' / 'forward_timestamps.txt'
            bwd_timestamp_path = self.flow_dir / rec.name / 'flow' / 'backward_timestamps.txt'
            if fwd_timestamp_path.exists():
                self.flow_timestamps[rec.name + '_forward'] = np.loadtxt(fwd_timestamp_path, delimiter=',')
            if bwd_timestamp_path.exists():
                self.flow_timestamps[rec.name + '_backward'] = np.loadtxt(bwd_timestamp_path, delimiter=',')
        logging.debug(f"Loaded flow timestamps: {self.flow_timestamps.keys()}")

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
            logging.debug(f"Checking for event files in {rec / 'events' / 'left'} and {rec / 'events' / 'right'}")
            if not left_events.exists():
                logging.debug(f"No event files found in {rec / 'events' / 'left'}")
            if not right_events.exists():
                logging.debug(f"No event files found in {rec / 'events' / 'right'}")
        self.event_files = self.event_files[:1]  # Debugging with only the first event file
        logging.debug(f"Total event files found: {len(self.event_files)}")

    def __len__(self):
        return len(self.event_files)

    @profile
    def __getitem__(self, idx):
        logging.debug(f"Processing index: {idx}")
        event_file = self.event_files[idx]
        rec_name = event_file.parent.parent.parent.name
        direction = 'forward' if 'left' in event_file.parent.name else 'backward'
        flow_dir = 'forward' if direction == 'forward' else 'backward'

        flow_file_path = self.flow_dir / rec_name / 'flow' / flow_dir
        flow_files = sorted(flow_file_path.glob('*.png'))

        if len(flow_files) > 0:
            flow_file = flow_files[idx % len(flow_files)]
            logging.debug(f"Flow file path: {flow_file}")
            flow_x, flow_y, valid = self.load_optical_flow(flow_file)
            flow_tensor = self.preprocess_flow(flow_x, flow_y, valid)
        else:
            logging.debug(f"No flow files found in {flow_file_path}")
            flow_tensor = torch.zeros((2, self.height, self.width), dtype=torch.float32)

        logging.debug(f"Accessing timestamps for {rec_name}_{direction} at index {idx}")
        ts = self.flow_timestamps[f"{rec_name}_{direction}"]
        logging.debug(f"Timestamps: {ts}")

        if ts.ndim == 2:
            ts = ts.flatten()

        ts_end = ts[idx % len(ts)].item() if isinstance(ts[idx % len(ts)], np.ndarray) and ts[idx % len(ts)].size == 1 else ts[idx % len(ts)]
        ts_start = ts_end - self.delta_t_us

        logging.debug(f"Timestamp start: {ts_start}, end: {ts_end}")
        logging.debug(f"Getting events from {ts_start} to {ts_end}")
        events = self.event_slicers[event_file].get_events(ts_start, ts_end)
        spike_tensor = self.process_events(events)

        if self.transform:
            spike_tensor = self.transform(spike_tensor)
            if 'flow_tensor' in locals():
                flow_tensor = self.transform(flow_tensor)

        combined_data = torch.cat([spike_tensor, flow_tensor], dim=0) if 'flow_tensor' in locals() else spike_tensor
        proxy_target = self.generate_proxy_target(flow_tensor) if 'flow_tensor' in locals() else torch.zeros(1)

        return combined_data, proxy_target

    @profile
    def load_optical_flow(self, flow_path):
        logging.debug(f"Loading optical flow from {flow_path}")
        flow_img = cv2.imread(str(flow_path), cv2.IMREAD_UNCHANGED)
        flow_x = (flow_img[:, :, 2].astype(np.float32) - 2**15) / 128.0
        flow_y = (flow_img[:, :, 1].astype(np.float32) - 2**15) / 128.0
        valid = flow_img[:, :, 0] > 0
        return flow_x, flow_y, valid

    @profile
    def preprocess_flow(self, flow_x, flow_y, valid):
        logging.debug("Preprocessing flow")
        flow_x[~valid] = 0
        flow_y[~valid] = 0
        return torch.from_numpy(np.stack((flow_x, flow_y), axis=0))

    @profile
    def process_events(self, events: Dict[str, np.ndarray]) -> torch.Tensor:
        logging.debug("Processing events")
        spike_tensor = torch.zeros((2, self.height, self.width), dtype=torch.float32)

        x = events['x']
        y = events['y']
        p = events['p']
        t = events['t']

        for i in range(len(t)):
            spike_tensor[0, y[i], x[i]] = float(p[i])  # polarities
            spike_tensor[1, y[i], x[i]] = (t[i] - t.min()) / (t.max() - t.min())  # normalized time

        return spike_tensor

    @profile
    def generate_proxy_target(self, flow_tensor, time_step=1):
        logging.debug("Generating proxy target")
        displacement_x = flow_tensor[0] * time_step
        displacement_y = flow_tensor[1] * time_step

        velocity_x = torch.sum(displacement_x) / time_step
        velocity_y = torch.sum(displacement_y) / time_step

        velocity = torch.sqrt(velocity_x**2 + velocity_y**2).mean()

        logging.debug(f"Displacement X: {displacement_x}")
        logging.debug(f"Displacement Y: {displacement_y}")
        logging.debug(f"Velocity X: {velocity_x}")
        logging.debug(f"Velocity Y: {velocity_y}")
        logging.debug(f"Overall Velocity: {velocity}")

        return velocity.unsqueeze(0)

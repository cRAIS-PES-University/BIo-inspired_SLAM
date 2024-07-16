'''
#version 2
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tonic import transforms
import cv2
import logging
from utils.eventslicer import EventSlicer
from local_dsec import LocalDSEC  # Ensure LocalDSEC is imported correctly

# Configure logging for dataloader
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DSECIntegratedDataset(Dataset):
    def __init__(self, event_dir, flow_dir, transform=None, delta_t_ms=50):
        self.event_dir = event_dir
        self.flow_dir = flow_dir
        self.transform = transform or transforms.Compose([
            transforms.Denoise(filter_time=10000), 
            transforms.ToFrame(sensor_size=(640, 480), time_window=5000)
        ])
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

        # Load dataset using the modified DSEC class
        self.dataset = LocalDSEC(
            save_to=self.event_dir,
            split='train',
            data_selection=['events_left', 'events_right'],
            target_selection=['optical_flow_forward_event', 'optical_flow_backward_event'],
            transform=self.transform
        )
        
        # Prepare for optical flow and events handling
        self.prepare_flow_and_events()

    def prepare_flow_and_events(self):
        self.flow_timestamps = {}
        self.event_slicers = {}
        for rec_name in self.valid_recordings:
            fwd_timestamp_path = os.path.join(self.flow_dir, rec_name, 'forward_timestamps.txt')
            bwd_timestamp_path = os.path.join(self.flow_dir, rec_name, 'backward_timestamps.txt')
            left_events_path = os.path.join(self.event_dir, rec_name, 'events_left', 'events.h5')
            right_events_path = os.path.join(self.event_dir, rec_name, 'events_right', 'events.h5')
            
            if os.path.exists(fwd_timestamp_path) and os.path.exists(bwd_timestamp_path):
                self.flow_timestamps[rec_name + '_forward'] = np.loadtxt(fwd_timestamp_path, delimiter=',')
                self.flow_timestamps[rec_name + '_backward'] = np.loadtxt(bwd_timestamp_path, delimiter=',')
            
            if os.path.exists(left_events_path) and os.path.exists(right_events_path):
                self.event_slicers[left_events_path] = EventSlicer(h5py.File(left_events_path, 'r'))
                self.event_slicers[right_events_path] = EventSlicer(h5py.File(right_events_path, 'r'))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        logging.debug(f"Fetching index: {idx}")
        try:
            data, target, timestamps = self.dataset[idx]
            logging.debug(f"Data and target fetched for index {idx}")
            
            event_file = list(self.event_slicers.keys())[idx]
            rec_name = event_file.split('/')[-3]  # Adjust based on directory structure
            direction = 'forward' if 'left' in event_file else 'backward'
            flow_dir = 'forward' if direction == 'forward' else 'backward'
    
            flow_file = f"{self.flow_dir}/{rec_name}/optical_flow_{flow_dir}_event/{str(idx).zfill(6)}.png"
            logging.debug(f"Flow file path: {flow_file}")
            
            if os.path.exists(flow_file):
                flow_x, flow_y, valid = self.load_optical_flow(flow_file)
                flow_tensor = self.preprocess_flow(flow_x, flow_y, valid)
                logging.debug(f"Optical flow loaded and preprocessed for index {idx}")
    
            ts = self.flow_timestamps[f"{rec_name}_{flow_dir}"]
            ts_end = ts[idx].item() if isinstance(ts[idx], np.ndarray) and ts[idx].size == 1 else ts[idx]
            ts_start = ts_end - self.delta_t_us
    
            events = self.event_slicers[event_file].get_events(ts_start, ts_end)
            spike_tensor = self.process_events(events)
    
            if self.transform:
                spike_tensor = self.transform(spike_tensor)
                if 'flow_tensor' in locals():
                    flow_tensor = self.transform(flow_tensor)
    
            combined_data = torch.cat([spike_tensor, flow_tensor], dim=0) if 'flow_tensor' in locals() else spike_tensor
            return combined_data, flow_tensor
    
        except Exception as e:
            logging.error(f"Error at index {idx}: {e}")
            raise e

    def load_optical_flow(self, flow_path):
        flow_img = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
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
        spike_tensor = torch.zeros((4, self.height, self.width), dtype=torch.float32)
        spike_tensor[0, x, y] = 1  # x coordinates
        spike_tensor[1, x, y] = y  # y coordinates
        spike_tensor[2, x, y] = p  # polarities
        spike_tensor[3, x, y] = t  # timestamps
        return spike_tensor
'''

'''
#version 1
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tonic import transforms
import cv2
import logging
from utils.eventslicer import EventSlicer
from local_dsec import LocalDSEC  # Ensure LocalDSEC is imported correctly

# Configure logging for dataloader
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DSECIntegratedDataset(Dataset):
    def __init__(self, event_dir, flow_dir, transform=None, delta_t_ms=50):
        self.event_dir = event_dir
        self.flow_dir = flow_dir
        self.transform = transform or transforms.Compose([
            transforms.Denoise(filter_time=10000), 
            transforms.ToFrame(sensor_size=(640, 480), time_window=5000)
        ])
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

        self.dataset = LocalDSEC(
            save_to=self.event_dir,
            split='train',
            data_selection=['events_left', 'events_right'],
            target_selection=['optical_flow_forward_event', 'optical_flow_backward_event'],
            transform=self.transform
        )
        self.prepare_flow_and_events()

    def prepare_flow_and_events(self):
        self.flow_timestamps = {}
        self.event_slicers = {}
        for rec_name in self.valid_recordings:
            fwd_timestamp_path = os.path.join(self.flow_dir, rec_name, 'forward_timestamps.txt')
            bwd_timestamp_path = os.path.join(self.flow_dir, rec_name, 'backward_timestamps.txt')
            left_events_path = os.path.join(self.event_dir, rec_name, 'events_left', 'events.h5')
            right_events_path = os.path.join(self.event_dir, rec_name, 'events_right', 'events.h5')
            
            if os.path.exists(fwd_timestamp_path) and os.path.exists(bwd_timestamp_path):
                self.flow_timestamps[rec_name + '_forward'] = np.loadtxt(fwd_timestamp_path, delimiter=',')
                self.flow_timestamps[rec_name + '_backward'] = np.loadtxt(bwd_timestamp_path, delimiter=',')
            
            if os.path.exists(left_events_path) and os.path.exists(right_events_path):
                self.event_slicers[left_events_path] = EventSlicer(h5py.File(left_events_path, 'r'))
                self.event_slicers[right_events_path] = EventSlicer(h5py.File(right_events_path, 'r'))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        logging.debug(f"Fetching index: {idx}")
        data, target, timestamps = self.dataset[idx]
        event_file = list(self.event_slicers.keys())[idx]
        rec_name = event_file.split('/')[-3]
        direction = 'forward' if 'left' in event_file else 'backward'
        flow_dir = 'forward' if direction == 'forward' else 'backward'

        flow_file = f"{self.flow_dir}/{rec_name}/optical_flow_{flow_dir}_event/{str(idx).zfill(6)}.png"
        if os.path.exists(flow_file):
            flow_x, flow_y, valid = self.load_optical_flow(flow_file)
            flow_tensor = self.preprocess_flow(flow_x, flow_y, valid)
            ts = self.flow_timestamps[f"{rec_name}_{flow_dir}"]
            ts_end = ts[idx]
        me}/optical_    ts_start = ts_end - self.delta_t_us
            events = self.event_slicers[event_file].get_events(ts_start, ts_end)
            spike_tensor = self.process_events(events)
            combined_data = torch.cat([spike_tensor, flow_tensor], dim=0) if 'flow_tensor' in locals() else spike_tensor
            return combined_data, flow_tensor

        logging.error(f"Missing data at index {idx}")
        raise FileNotFoundError(f"Data file not found: {flow_file}")

    def load_optical_flow(self, flow_path):
        flow_img = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        flow_x = (flow_img[:, :, 2].astype(np.float32) - 2**15) / 128.0
        flow_y = (flow_img[:, :, 1].astype(np.float32) - 2**15) / 128.0
        valid = flow_img[:, :, 0] > 0
        return flow_x, flow_y, valid

    def preprocess_flow(self, flow_x, flow_y, valid):
        flow_x[~valid] = 0
        flow_y[~valid] = 0
        return torch.from_numpy(np.stack((flow_x, flow_y), axis=0))

    def process_events(self, events):
        spike_tensor = torch.zeros((4, self.height, self.width), dtype=torch.float32)
        spike_tensor[0, events['x'], events['y']] = 1
        spike_tensor[1, events['x'], events['y']] = events['y']
        spike_tensor[2, events['x'], events['y']] = events['p']
        spike_tensor[3, events['x'], events['y']] = events['t']
        return spike_tensor
'''

#Version 3
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tonic import transforms
import cv2
import logging
from utils.eventslicer import EventSlicer
from local_dsec import LocalDSEC  # Ensure LocalDSEC is imported correctly

# Configure logging to file
logging.basicConfig(filename='run.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DSECIntegratedDataset(Dataset):
    def __init__(self, event_dir, flow_dir, transform=None, delta_t_ms=50):
        self.event_dir = event_dir
        self.flow_dir = flow_dir
        self.transform = transform or transforms.Compose([
            transforms.Denoise(filter_time=10000), 
            transforms.ToFrame(sensor_size=(640, 480, 2), time_window=5000)
        ])
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

        # Load dataset using the modified DSEC class
        self.dataset = LocalDSEC(
            save_to=self.event_dir,
            split='train',
            data_selection=['events_left', 'events_right'],
            target_selection=['optical_flow_forward_event', 'optical_flow_backward_event'],
            transform=self.transform
        )
        
        # Prepare for optical flow and events handling
        self.prepare_flow_and_events()

    def prepare_flow_and_events(self):
        self.flow_timestamps = {}
        self.event_slicers = {}
        for rec_name in self.valid_recordings:
            fwd_timestamp_path = os.path.join(self.flow_dir, rec_name, 'forward_timestamps.txt')
            bwd_timestamp_path = os.path.join(self.flow_dir, rec_name, 'backward_timestamps.txt')
            left_events_path = os.path.join(self.event_dir, rec_name, 'events_left', 'events.h5')
            right_events_path = os.path.join(self.event_dir, rec_name, 'events_right', 'events.h5')
            
            if os.path.exists(fwd_timestamp_path) and os.path.exists(bwd_timestamp_path):
                self.flow_timestamps[rec_name + '_forward'] = np.loadtxt(fwd_timestamp_path, delimiter=',')
                self.flow_timestamps[rec_name + '_backward'] = np.loadtxt(bwd_timestamp_path, delimiter=',')
            
            if os.path.exists(left_events_path) and os.path.exists(right_events_path):
                self.event_slicers[left_events_path] = EventSlicer(h5py.File(left_events_path, 'r'))
                self.event_slicers[right_events_path] = EventSlicer(h5py.File(right_events_path, 'r'))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        logging.debug(f"Fetching index: {idx}")
        try:
            data, target, timestamps = self.dataset[idx]
            logging.debug(f"Data and target fetched for index {idx}")
            
            event_file = list(self.event_slicers.keys())[idx]
            rec_name = event_file.split('/')[-3]  # Adjust based on directory structure
            direction = 'forward' if 'left' in event_file else 'backward'
            flow_dir = 'forward' if direction == 'forward' else 'backward'

            flow_file = f"{self.flow_dir}/{rec_name}/optical_flow_{flow_dir}_event/{str(idx).zfill(6)}.png"
            logging.debug(f"Flow file path: {flow_file}")

            if os.path.exists(flow_file):
                flow_x, flow_y, valid = self.load_optical_flow(flow_file)
                flow_tensor = self.preprocess_flow(flow_x, flow_y, valid)
                logging.debug(f"Optical flow loaded and preprocessed for index {idx}")

            ts = self.flow_timestamps[f"{rec_name}_{flow_dir}"]
            ts_end = ts[idx].item() if isinstance(ts[idx], np.ndarray) and ts[idx].size == 1 else ts[idx]
            ts_start = ts_end - self.delta_t_us

            # Ensure ts_start and ts_end are single values, not arrays
            if isinstance(ts_start, np.ndarray):
                ts_start = ts_start[0]
            if isinstance(ts_end, np.ndarray):
                ts_end = ts_end[0]

            events = self.event_slicers[event_file].get_events(ts_start, ts_end)
            logging.debug(f"Events fetched for index {idx}: {events}")

            # Print datatype and structure of events
            logging.debug(f"Events datatype for index {idx}: {type(events)}")
            logging.debug(f"Events keys for index {idx}: {events.keys() if isinstance(events, dict) else 'N/A'}")

            spike_tensor = self.process_events(events)

            # Convert spike_tensor to numpy structured array if transformations are applied
            if self.transform:
                events_np = np.zeros(len(events['t']), dtype=[('x', 'i4'), ('y', 'i4'), ('p', 'i4'), ('t', 'f8')])
                events_np['x'] = events['x']
                events_np['y'] = events['y']
                events_np['p'] = events['p']
                events_np['t'] = events['t']
                logging.debug(f"Before transform: {events_np}")
                events_np = self.transform(events_np)
                logging.debug(f"After transform: {events_np}")

                # Convert back to tensor
                spike_tensor = torch.zeros((4, self.height, self.width), dtype=torch.float32)
                y_indices = torch.tensor(events_np['y'].astype(int).tolist(), dtype=torch.long)
                x_indices = torch.tensor(events_np['x'].astype(int).tolist(), dtype=torch.long)
                spike_tensor[0, y_indices, x_indices] = 1  # x coordinates
                spike_tensor[1, y_indices, x_indices] = torch.tensor(events_np['y'].astype(float).tolist(), dtype=torch.float32)  # y coordinates
                spike_tensor[2, y_indices, x_indices] = torch.tensor(events_np['p'].astype(float).tolist(), dtype=torch.float32)  # polarities
                spike_tensor[3, y_indices, x_indices] = torch.tensor(events_np['t'].astype(float).tolist(), dtype=torch.float32)  # timestamps

            combined_data = torch.cat([spike_tensor, flow_tensor], dim=0) if 'flow_tensor' in locals() else spike_tensor
            return combined_data, flow_tensor

        except Exception as e:
            logging.error(f"Error at index {idx}: {e}")
            raise e

    def load_optical_flow(self, flow_path):
        flow_img = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        flow_x = (flow_img[:, :, 2].astype(np.float32) - 2**15) / 128.0
        flow_y = (flow_img[:, :, 1].astype(np.float32) - 2**15) / 128.0
        valid = flow_img[:, :, 0] > 0
        return flow_x, flow_y, valid

    def preprocess_flow(self, flow_x, flow_y, valid):
        flow_x[~valid] = 0
        flow_y[~valid] = 0
        return torch.from_numpy(np.stack((flow_x, flow_y), axis=0))

    def process_events(self, events):
        spike_tensor = torch.zeros((4, self.height, self.width), dtype=torch.float32)
        y_indices = torch.tensor(events['y'].astype(int).tolist(), dtype=torch.long)
        x_indices = torch.tensor(events['x'].astype(int).tolist(), dtype=torch.long)
        spike_tensor[0, y_indices, x_indices] = 1  # x coordinates
        spike_tensor[1, y_indices, x_indices] = torch.tensor(events['y'].astype(float).tolist(), dtype=torch.float32)  # y coordinates
        spike_tensor[2, y_indices, x_indices] = torch.tensor(events['p'].astype(float).tolist(), dtype=torch.float32)  # polarities
        spike_tensor[3, y_indices, x_indices] = torch.tensor(events['t'].astype(float).tolist(), dtype=torch.float32)  # timestamps
        return spike_tensor

# Running the script now will log all the debug information to 'run.log' and provide detailed insights into the data processing steps.

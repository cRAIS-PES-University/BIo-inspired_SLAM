import os
import h5py
import numpy as np
import imageio
import logging
from PIL import Image
from tonic.datasets import DSEC

# Set up a logger for LocalDSEC
local_dsec_logger = logging.getLogger('LocalDSEC')
fh = logging.FileHandler('local_dsec.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
local_dsec_logger.addHandler(fh)

class LocalDSEC(DSEC):
    def __init__(self, save_to, split, data_selection, target_selection, transform):
        self.save_to = save_to
        super().__init__(save_to=save_to, split=split, data_selection=data_selection, 
                         target_selection=target_selection, transform=transform)
        logging.debug("Initialized LocalDSEC with provided parameters.")

    def _check_exists(self, selections):
        """Ensure all required files exist."""
        missing_files = []
        for recording in self.recording_selection:
            for selection in selections:
                path = os.path.join(self.save_to, recording, selection)
                if not os.path.exists(path):
                    missing_files.append(path)
                    logging.error(f"Missing required data at {path}")
                else:
                    logging.debug(f"Data exists at {path}")
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")


    def get_required_files(self, selection, recording):
        """Determine required files based on the selection type."""
        if 'events' in selection:
            return ['events.h5']
        elif 'optical_flow' in selection:
            direction = 'forward' if 'forward' in selection else 'backward'
            num_images = self.get_number_of_flow_images(recording, selection, direction)
            return [f"{str(i).zfill(6)}.png" for i in range(num_images)]
        elif 'timestamp' in selection:
            return ['forward_timestamp.txt', 'backward_timestamp.txt']
        return []

    def get_number_of_flow_images(self, recording, selection, flow_direction):
        """Count PNG files in the specified flow direction directory."""
        flow_dir = os.path.join(self.save_to, recording, f'optical_flow_{flow_direction}_event')
        if os.path.exists(flow_dir):
            num_images = len([name for name in os.listdir(flow_dir) if name.endswith('.png')])
            local_dsec_logger.debug(f"Found {num_images} flow images in {flow_dir}.")
            return num_images
        else:
            local_dsec_logger.error(f"No such directory: {flow_dir}")
            raise FileNotFoundError(f"No such directory: {flow_dir}")

    def __getitem__(self, index):
        local_dsec_logger.debug(f"Processing index: {index}")
        recording = self.recording_selection[index]
        base_folder = os.path.join(self.save_to, recording)
    
        data_tuple, target_tuple = [], []
        timestamps = {}
    
        try:
            # Load event data
            for data_name in self.data_selection:
                data_path = os.path.join(base_folder, data_name, 'events.h5')
                if os.path.exists(data_path):
                    with h5py.File(data_path, "r") as file:
                        data = {'events': file['events'][()]}
                        data_tuple.append(data)
                        local_dsec_logger.debug(f"Loaded data from {data_path}")
                else:
                    local_dsec_logger.error(f"Data file not found: {data_path}")
    
            # Load target optical flow images and their timestamps
            for target_name in self.target_selection:
                target_path = os.path.join(base_folder, target_name)
                if os.path.exists(target_path):
                    images = [imageio.imread(os.path.join(target_path, f)) for f in sorted(os.listdir(target_path)) if f.endswith('.png')]
                    target_tuple.append(np.stack(images))
                    local_dsec_logger.debug(f"Loaded target data from {target_path}")
    
                    # Load corresponding timestamps
                    timestamp_filename = 'forward_timestamps.txt' if 'forward' in target_name else 'backward_timestamps.txt'
                    timestamp_path = os.path.join(base_folder, timestamp_filename)
                    if os.path.exists(timestamp_path):
                        timestamps[target_name] = np.loadtxt(timestamp_path, delimiter=',')
                        local_dsec_logger.debug(f"Loaded timestamps from {timestamp_path}")
                    else:
                        local_dsec_logger.error(f"Timestamp file not found: {timestamp_path}")
                else:
                    local_dsec_logger.error(f"Target directory not found: {target_path}")
    
        except Exception as e:
            local_dsec_logger.error(f"Error processing index {index}: {e}")
    
        return data_tuple, target_tuple, timestamps  # Including timestamps in the return

    def __len__(self):
        return len(self.recording_selection)

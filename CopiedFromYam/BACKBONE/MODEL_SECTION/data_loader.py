import numpy as np
from torch.utils.data import DataLoader
from CopiedFromYam.DATA import create_dataset as dataset
from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import *

config = load_config()
def load_data(part='train'):
    if config['General']['dataset']=="BIUREV":
        if part=='train':
            trainloader = DataLoader(dataset.ReverbDataset(config['General']['mics_num'], config['General']['dataset'], 'train',config['General']['data_base_path'],data_type=config['General']['DATA_TYPE'],warmup_stage=config['General']['warm_up_step']),
                             config['Train params']['batch_size'],
                             shuffle=True,
                             num_workers=config['Train params']['num_workers'])
            return trainloader
        elif part=='val':
        # valnearloader = DataLoader(dataset.ReverbDataset(config['General']['mics_num'], config['General']['dataset'], 'val_near', 1),
        #                            config['Train params']['batch_size'],
        #                            num_workers=config['Train params']['num_workers'])
            valfarloader = DataLoader(dataset.ReverbDataset(config['General']['mics_num'], config['General']['dataset'], 'val', config['General']['data_base_path'],data_type=config['General']['DATA_TYPE'],warmup_stage=config['General']['warm_up_step']),
                                      config['Train params']['batch_size'],
                                      num_workers=config['Train params']['num_workers'])
            return valfarloader
        elif part=='test':

            testfarloader = DataLoader(dataset.ReverbDataset(config['General']['mics_num'], config['General']['dataset'], 'test', config['General']['data_base_path'],data_type=config['General']['DATA_TYPE'],warmup_stage=config['General']['warm_up_step']),config['Train params']['batch_size'],
                                      num_workers=config['Train params']['num_workers'])
            return testfarloader
        elif part=='cal':

            testfarloader = DataLoader(dataset.ReverbDataset(config['General']['mics_num'], config['General']['dataset'], 'cal', config['General']['data_base_path'],data_type=config['General']['DATA_TYPE'],warmup_stage=config['General']['warm_up_step']),config['Train params']['batch_size'],
                                      num_workers=config['Train params']['num_workers'])
            return testfarloader
        else:
            raise {"Unknown dataset part to downlod, choose one of: train, val, test, cal."}


def save_test_cal_raw_files(testfarloader):
    # Access the dataset instance
    test_dataset = testfarloader.dataset

    # Extract file paths
    file_paths = [str(file) for file in test_dataset.files]

    # Save the file paths as a NumPy array
    file_paths_array = np.array(file_paths)

    # Save to disk
    np.save('test_far_file_paths.npy', file_paths_array)

    print(f"Saved file paths for 'test_far' to test_far_file_paths.npy.")
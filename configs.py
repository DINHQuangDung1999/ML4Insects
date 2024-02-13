from easydict import EasyDict
from copy import deepcopy as dc

config_dwt_Adam = EasyDict({'window_size': 1024, 'hop_length': 256, 'method': 'dwt', 'scale': True, 'outlier_filter': False, # Data specs
                       'optimizer': 'Adam', 'n_epochs': 100, 'lr': 0.0001, 'batch_size': 256, #model specs
                       'scope':4}) # Classifier specs

config_dwt_SGD = EasyDict({'window_size': 1024, 'hop_length': 256, 'method': 'dwt', 'scale': True, 'outlier_filter': False,# Data specs
                       'optimizer': 'SGD', 'n_epochs': 100, 'lr': 0.0001, 'batch_size': 256, #model specs
                       'scope':4}) # Classifier specs

config_fft_Adam = EasyDict({'window_size': 1024, 'hop_length': 256, 'method': 'fft', 'scale': True, 'outlier_filter': False,# Data specs
                       'optimizer': 'Adam', 'n_epochs': 100, 'lr': 0.0001, 'batch_size': 256, #model specs
                       'scope':4}) # Classifier specs

config_fft_SGD = EasyDict({'window_size': 1024, 'hop_length': 256, 'method': 'fft', 'scale': True, 'outlier_filter': False,# Data specs
                       'optimizer': 'SGD', 'n_epochs': 100, 'lr': 0.0001, 'batch_size': 256, #model specs
                       'scope':4}) # Classifier specs

config_raw_Adam = EasyDict({'window_size': 1024, 'hop_length': 256, 'method': 'raw', 'scale': True, 'outlier_filter': False,
                       'optimizer': 'Adam', 'n_epochs': 100, 'lr': 0.0001, 'batch_size': 256, #model specs
                       'scope':4}) # Classifier specs

config_raw_SGD = EasyDict({'window_size': 1024, 'hop_length': 256, 'method': 'raw', 'scale': True, 'outlier_filter': False,
                       'optimizer': 'SGD', 'n_epochs': 100, 'lr': 0.0001, 'batch_size': 256, #model specs
                       'scope':4}) # Classifier specs

config_spec_img_Adam = EasyDict({'window_size': 1024, 'hop_length': 256, 'method': 'spec_img', 'scale': True, 'outlier_filter': False,
                       'optimizer': 'Adam', 'n_epochs': 100, 'lr': 0.0001, 'batch_size': 256, #model specs
                       'scope':4}) # Classifier specs

config_spec_img_SGD = EasyDict({'window_size': 1024, 'hop_length': 256, 'method': 'spec_img', 'scale': True, 'outlier_filter': False,
                       'optimizer': 'SGD', 'n_epochs': 100, 'lr': 0.0001, 'batch_size': 256, #model specs
                       'scope':4}) # Classifier specs
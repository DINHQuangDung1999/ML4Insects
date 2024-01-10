from easydict import EasyDict

sliding_windows_configs = [EasyDict({'window_size': 1024, 'hop_length':128}),
                           EasyDict({'window_size': 1024, 'hop_length':256}),
                           EasyDict({'window_size': 1024, 'hop_length':1024})]

MLP_lvl1_configs = [EasyDict({'device': 'cuda', 'n_classes': 7, 'num_epochs': 10, 'learning_rate': 0.0001,'batch_size': 128,'method':'fft','model_type':'mlp', 'seq_length': 513}),
                    EasyDict({'device': 'cuda', 'n_classes': 7, 'num_epochs': 10, 'learning_rate': 0.0001,'batch_size': 128,'method':'raw','model_type':'mlp', 'seq_length': 1024}),
                    EasyDict({'device': 'cuda', 'n_classes': 7, 'num_epochs': 15, 'learning_rate': 0.0001,'batch_size': 128,'method':'fft','model_type':'mlp', 'seq_length': 513}),
                    EasyDict({'device': 'cuda', 'n_classes': 7, 'num_epochs': 15, 'learning_rate': 0.0001,'batch_size': 128,'method':'fft','model_type':'mlp', 'seq_length': 1024})]

MLP_lvl2_configs = [EasyDict({'device': 'cuda', 'n_classes': 7, 'num_epochs': 10, 'learning_rate': 0.0001,'batch_size': 128,'method':'fft','model_type':'mlp', 'seq_length': 513}),
                    EasyDict({'device': 'cuda', 'n_classes': 7, 'num_epochs': 10, 'learning_rate': 0.0001,'batch_size': 128,'method':'raw','model_type':'mlp', 'seq_length': 1024}),
                    EasyDict({'device': 'cuda', 'n_classes': 7, 'num_epochs': 10, 'learning_rate': 0.0001,'batch_size': 128,'method':'fft','model_type':'mlp', 'seq_length': 513}),
                    EasyDict({'device': 'cuda', 'n_classes': 7, 'num_epochs': 10, 'learning_rate': 0.0001,'batch_size': 128,'method':'raw','model_type':'mlp', 'seq_length': 1024}),]
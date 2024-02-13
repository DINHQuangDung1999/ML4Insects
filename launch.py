# Import modules
import datagenerator
import utils
import configs
from train import Trainer, kfold_CV
from test import Classifier
from models import MLP, CNN1D, CNN2D, ResNet

name_wheat = ['day2run2', 'reutday3', 'reutwheat', 'reutwheat2', 'reutwheatday', 'reutwheatrun3', 'wheatday2run2', 'wheatday2run3', 'wheatrun']
name_zt = ['0zt','8zt','16zt']
name_hemp = ['non-viruliferous-hemp', 'non-viruliferous-potato','viruliferous-hemp', 'viruliferous-potato']

def main():
    experiment = [
                #   {'model': CNN2D(), 'config': configs.config_spec_img_Adam, 'data': utils.name_zt, 'description': 'zt_data'},
    #               {'model': CNN2D(), 'config': configs.config_spec_img_SGD, 'data': utils.name_zt, 'description': 'zt_data'},
    #               {'model': CNN2D(), 'config': configs.config_spec_img_Adam, 'data': utils.name_hemp, 'description': 'hemp_data'},
    #               {'model': CNN2D(), 'config': configs.config_spec_img_Adam, 'data': utils.name_wheat, 'description': 'wheat_data'},
    #               {'model': MLP(), 'config': configs.config_fft_Adam, 'data': utils.name_zt, 'description': 'zt_data'},
    #               {'model': MLP(), 'config': configs.config_fft_SGD, 'data': utils.name_zt, 'description': 'zt_data'},
    #               {'model': MLP(), 'config': configs.config_fft_Adam, 'data': utils.name_hemp, 'description': 'hemp_data'},
    #               {'model': MLP(), 'config': configs.config_fft_Adam, 'data': utils.name_wheat, 'description': 'wheat_data'},
    #               {'model': CNN1D(), 'config': configs.config_raw_Adam, 'data': utils.name_zt, 'description': 'zt_data'},
    #               {'model': CNN1D(), 'config': configs.config_raw_SGD, 'data': utils.name_zt, 'description': 'zt_data'},
    #               {'model': CNN1D(), 'config': configs.config_raw_Adam, 'data': utils.name_hemp, 'description': 'hemp_data'},
    #               {'model': CNN1D(), 'config': configs.config_raw_Adam, 'data': utils.name_wheat, 'description': 'wheat_data'},
                  {'model': ResNet(), 'config': configs.config_raw_Adam, 'data': utils.name_zt, 'description': 'zt_data'},
                  {'model': ResNet(), 'config': configs.config_raw_Adam, 'data': utils.name_hemp, 'description': 'hemp_data'},
                  {'model': ResNet(), 'config': configs.config_raw_Adam, 'data': utils.name_wheat, 'description': 'wheat_data'},
                  ]
    for e in experiment:
        model = e['model']; config = e['config']; data_names = e['data']; txt = e['description']
        print('Get data')
        splits = utils.get_train_test_filenames(0.8)
        df, l, df_test, l_test = datagenerator.generate_data(data_splits = splits, data_names = data_names, method = config.method, scale = config.scale,
                                                window_size = config.window_size, hop_length = config.hop_length, verbose = True)

        train = utils.get_one_level_split(df,l); test = utils.get_one_level_split(df_test,l_test)
        del(df); del(l); del(df_test); del(l_test)
        trainer = Trainer(model,config)
        trainer.fit(train, test); trainer.get_loader(); trainer.train(); trainer.test()
        trainer.plot_result(savefig=True, name = txt); trainer.write_log(description= txt); trainer.save_checkpoint(txt)
        del(trainer)

if __name__ == '__main__':
    print('Begin')
    main()
print('Finished')
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import torch
import pandas as pd
import numpy as np
import preprocessing
import visualization
import matplotlib.pyplot as plt
from copy import deepcopy as dc 

class Classifier():
    def __init__(self,models,device):
        self.models = models
        self.device = device
        self.configs_loaded = False
        self.models_loaded = False
        if len(self.models) == 1:
            print('Single level classifier')
            self.level = 1
        elif len(self.models) == 2:
            print('Two level classifier')
            self.level = 2
        else: 
            print('Please input a list of models!')
            return 
        
    def load_configs(self, configs):
        print('Configurations loaded.')
        self.configs = configs
        self.configs_loaded = True

    def load_checkpoint(self,path):
        for i in range(self.level):
            self.models[i] = torch.load(path[i])
            self.models[i].eval()
        self.models_loaded = True

    def predict(self, wave_array, ana, window_size=1024, hop_length=256, method = 'raw',model_type = 'mlp', scope: int = 1, verbose = False):
        
        for i in range(self.level):
            self.models[i].eval()

        if self.configs_loaded == True:
            window_size = self.configs.window_size
            hop_length = self.configs.hop_length
            method = self.configs.method
            model_type = self.configs.model_type
            scope = self.configs.scope
        else: 
            print('Configurations not loaded. Use specified.') 

        # Prepare data
        print('Preparing data...') if verbose == True else None
        self.wave_array = wave_array
        self.ana = ana
        self.input, self.true_label = preprocessing.generate_test_data(wave_array,ana,window_size=window_size,hop_length=hop_length, method = method)
        predicted_label = []
        self.level2_index = []

        print('Predicting...') if verbose == True else None
        for i in range(self.input.shape[0]): # level 1 prediction
            x = torch.from_numpy(self.input[i,:]).float().to(self.device)
            if model_type == 'mlp':
                x = x.unsqueeze(0)
            elif model_type == 'cnn':
                x = x.unsqueeze(0).unsqueeze(0)
                
            prediction = torch.argmax(self.models[0](x),dim=1).cpu().item()

            if self.level == 1: # 1-level prediction
                predicted_label.append(prediction)

            elif self.level == 2:# 2-level prediction
                if prediction == 0:
                    predicted_label.append(0)
                    self.level2_index.append(i)
                elif prediction == 1:
                    predicted_label.append(6)
        
        if self.level == 2: # level 2 prediction
            for i in self.level2_index:
                x = torch.from_numpy(self.input[i,:]).float().to(self.device)
                if model_type == 'mlp':
                    x = x.unsqueeze(0)
                elif model_type == 'cnn':
                    x = x.unsqueeze(0).unsqueeze(0)

                prediction = torch.argmax(self.models[1](x),dim=1).cpu().item()
                predicted_label[i] = prediction

        print('Aggregating predictions...') if verbose == True else None
        #map to original labels
        predicted_label = pd.Series(predicted_label).map({0: 1, 1: 2, 2: 4, 3: 6, 4: 7, 5: 8, 6: 5}).to_numpy() 

        self.predicted_label = predicted_label 
        self.acc = accuracy_score(self.true_label,predicted_label)
        self.b_acc = balanced_accuracy_score(self.true_label,predicted_label)
        self.cf = confusion_matrix(self.true_label,predicted_label)

        print('Accuracy: ', round(self.acc,3))
        print('Balanced accuracy: ', round(self.b_acc,3))
   
        # Write results in form of analysis files
        n_windows = len(predicted_label)//scope
        time = [0] # Make time marks
        for i in range(n_windows):
            time.append((window_size+i*scope*hop_length)/100)

        agg_pred = [] # aggregating consecutive predictions
        for i in range(n_windows):
            cl,c = np.unique(predicted_label[i*scope:(i+1)*scope], return_counts=True)
            agg_label = cl[np.argmax(c)]
            agg_pred.append(agg_label)

        predicted_label = np.append([agg_pred[0]], agg_pred) # merge the predicted labels

        ana_label = [] # analysis file
        ana_time = [time[0]]

        pin = 0 # Merge intervals
        for i in range(n_windows):
            if predicted_label[i] != predicted_label[pin]:
                ana_label.append(predicted_label[pin])
                ana_time.append(time[i])
                pin = i

        ana_label.append(predicted_label[n_windows-1])

        ana_time.append(time[i])
        ana_label += [12]

        self.predicted_analysis = pd.DataFrame({'label':ana_label,'time':ana_time})
        print('Finished.') if verbose == True else None
        return self.predicted_analysis

    def save_analysis(self,name: str = ''):
        self.predicted_analysis.to_csv(f'predicted_ana_{name}.ANA',sep = '\t',header = None,index=None)

    def plot(self,name: str = ''): 
        
        pred_ana = dc(self.predicted_analysis)
        pred_ana['time'] = pred_ana['time'].apply(lambda x: x*100).astype(int)

        visualization.visualize_wave(self.wave_array,self.ana)
        plt.title(name + ' Original segmentation')
        visualization.visualize_wave(self.wave_array,pred_ana)
        plt.title(name + ' Prediction result')

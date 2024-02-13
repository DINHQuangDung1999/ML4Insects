from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.functional as F
import pandas as pd
import numpy as np
import datagenerator
import visualization
import matplotlib.pyplot as plt
import os
from copy import deepcopy as dc 
import visualization 

wd = os.getcwd()

import warnings
warnings.filterwarnings("ignore")

class Classifier():
    def __init__(self, models, configs, device = None):
        try:
            len(models)
        except:
            self.models = [models]
        else: 
            self.models = models

        self.configs = configs
        self.window_size = configs.window_size
        self.scope = configs.scope
        self.hop_length = configs.window_size//configs.scope
        self.method = configs.method
        self.model_type = self.models[0].__type__
        self.model_arch = self.models[0].__arch__

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'     
               
        print('Configurations loaded.')
        
        try:
            len(self.models)
            if len(self.models) == 1:
                print('Single level classifier')
                self.level = 1
            elif len(self.models) == 2:
                print('Two level classifier')
                self.level = 2
        except: 
            raise RuntimeError('Please input a list of models!')

    def load_checkpoint(self,path):
        if isinstance(path,str):
            self.models[0] = torch.load(f'{wd}\\checkpoints\\{self.model_arch}\\{path}')
        else:
            for i in range(self.level):
                self.models[i] = torch.load(f'{wd}\\checkpoints\\{self.model_arch}\\{path[i]}')
                self.models[i].eval()

    def predict(self, wave_array, ana = None, to_ana = True, verbose = False):
        # Set models to eval mode 
        for i in range(self.level):
            self.models[i].eval()

        # Prepare data
        print('Preparing data...') if verbose == True else None

        # Scale the input amplitude to [0,1]
        scaler = MinMaxScaler()
        wave_array = scaler.fit_transform(wave_array.reshape(-1,1)).ravel()

        # Segment the signal
        self.wave_array = wave_array
        self.ana = ana
        data  = datagenerator.generate_test_data(wave_array, ana, window_size = self.window_size,
                                                                    hop_length = self.hop_length, 
                                                                    method = self.method)
        
        if ana is not None:
            self.input, self.true_label = data
        else: 
            self.input = data

        # Prediction
        predicted_label = []
        self.level2_index = []

        print('Predicting...') if verbose == True else None
        for i in range(self.input.shape[0]): # level 1 prediction
            x = torch.from_numpy(self.input[i,:]).float().to(self.device)
            if self.model_type == 'mlp':
                x = x.unsqueeze(0)
            elif self.model_type == 'cnn':
                x = x.unsqueeze(0).unsqueeze(0)
                
            prediction = torch.argmax(self.models[0](x),dim=-1).cpu().item()

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
                if self.model_type == 'mlp':
                    x = x.unsqueeze(0)
                elif self.model_type == 'cnn':
                    x = x.unsqueeze(0).unsqueeze(0)

                prediction = torch.argmax(self.models[1](x),dim=-1).cpu().item()
                predicted_label[i] = prediction

        #map to original labels  
        predicted_label = pd.Series(predicted_label).map({0: 1, 1: 2, 2: 4, 3: 6, 4: 7, 5: 8, 6: 5}).to_numpy() 
        self.predicted_label = predicted_label

        # Scoring
        if ana is not None: 
            self.acc = accuracy_score(self.true_label,predicted_label)
            self.b_acc = balanced_accuracy_score(self.true_label,predicted_label)
            self.cf = confusion_matrix(self.true_label, predicted_label)

            n_class_predictions = np.sum(self.cf,axis = 0)
            self.class_acc = [round(self.cf[i,i]/n_class_predictions[i],3) for i in range(len(n_class_predictions))]

            print(f'Accuracy: {round(self.acc,3)}, Balanced accuracy: {round(self.b_acc,3)}') if verbose == True else None

        if to_ana == True:
            return self.get_analysis(verbose)
        else:
            return self.predicted_label
        
    def get_analysis(self, verbose = True):
        print('Aggregating predictions...') if verbose == True else None
        predicted_label = dc(self.predicted_label)
        # Write results in form of analysis files
        n_windows = len(predicted_label)//self.scope
        time = [0] # Make time marks
        for i in range(n_windows):
            time.append((self.window_size+i*self.scope*self.hop_length)/100)

        agg_pred = [] # aggregating consecutive predictions
        for i in range(n_windows):
            cl,c = np.unique(predicted_label[i*self.scope : (i+1)*self.scope], return_counts=True)
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

    def save_analysis(self, name: str = ''):
        self.predicted_analysis.to_csv(f'./prediction/ANA/{name}_prediction.ANA',sep = '\t',header = None,index=None)

    def plot(self, result_only = False, save_plot = True, name: str = ''): 
        
        pred_ana = dc(self.predicted_analysis)
        if result_only == True:
            plt.figure(figsize=(16,3))
            visualization.visualize_wave(self.wave_array,pred_ana)
            plt.title(name + ' Prediction result')
            plt.tight_layout()
        else:
            plt.figure(figsize=(16,5))
            plt.subplot(2,1,1)
            visualization.visualize_wave(self.wave_array,self.ana)
            plt.title(name + ' Original segmentation')
            plt.xlabel('')
            plt.xticks([])
            plt.ylabel('')
            plt.yticks([])
            plt.subplot(2,1,2)
            visualization.visualize_wave(self.wave_array,pred_ana)
            plt.title(name + ' Prediction result')
            plt.tight_layout()

        if save_plot == True:
            if name == '':
                plt.savefig(os.path.join('./prediction/figures/','Untitled'))
            else:
                plt.savefig(os.path.join('./prediction/figures/', name))
                
    def interactive_plot(self, which = 'prediction', smoothen = False):
        pred_ana = dc(self.predicted_analysis)
        if which == 'prediction':
            visualization.interactive_visualization(self.wave_array, pred_ana, smoothen, title = which)
        elif which == 'original':
           visualization.interactive_visualization(self.wave_array, self.ana, smoothen, title = which)
        else:
            raise RuntimeError("Must input either 'prediction' or 'original' ")
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime
import utils
import seaborn as sns
import dataloader
import datagenerator
from sklearn.model_selection import StratifiedKFold
from easydict import EasyDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import os 
import torch.nn.functional as F

class Trainer():
    def __init__(self, model, config = None, device = None, seed = 28):
        # Trainer infos
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.seed = seed
        self.config = config

        self.model = model.to(self.device)
        
        self.result = {'training_loss': [], 'validation_loss': [], 'validation_accuracy': [], 
                       'test_class_accuracy': [], 'test_score': [], 'test_confusion_matrix': []}
        if self.model.init_weights == False:
            self.initialize_weights()   
        print(f'Date: {str(datetime.datetime.now())[:-7]} | Architecture: {self.model.__arch__} | Version: {self.model.__version__} | Device: {self.device} | Num epochs: {self.config.n_epochs} | lr: {self.config.lr}')        

        # Control variable
        self.data_loader_available = False
        self.data_available = False
        self.config_loaded = True if self.config is not None else False
        self.fit_config()   

    def initialize_weights(self, nonlinearity = 'relu'):
        torch.manual_seed(self.seed)
        for layer in self.model.children():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
                layer.weight = nn.init.kaiming_normal_(layer.weight, nonlinearity= nonlinearity)
                layer.bias = nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                layer.weight.data.fill_(1.0)
                layer.bias.data.fill_(0.0)
            
    def fit_config(self, config = None):
        if config is not None:
            self.config = config 

        self.criterion = nn.NLLLoss().to(self.device)
        
        if self.config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.lr)
        elif self.config.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.lr)
        else: 
            raise RuntimeError("Missing params 'optimizer' in input config.")
        
    def fit_data(self, train, test):
        self.X = train
        self.y = test
        self.data_available = True

    def get_data(self, name, ratio = 0.8):
        splits = utils.get_train_test_filenames(ratio)
        train, test = datagenerator.generate_data(data_names = name, data_splits = splits, config = self.config, verbose = True)
        self.fit_data(train, test)
        self.data_available = True

    def get_loader(self, r = 0.1, random_state = 28):
        print('Obtaning data loaders...')
        self.train_loader, self.validation_loader, self.test_loader = dataloader.get_loader(self.X, self.y, r = r, 
                                                                                 batch_size=self.config.batch_size, 
                                                                                 model_type = self.model.__type__,
                                                                                 random_state = random_state)
        self.data_loader_available = True

    def train_one_epoch(self):
        self.model.train()
        trainingloss = 0
        for x_batch,y_batch in self.train_loader:    
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            output = self.model(x_batch)
            loss = self.criterion(output,y_batch.ravel())
            trainingloss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        trainingloss = trainingloss/(len(self.train_loader))
        self.result['training_loss'].append(trainingloss)

    def validate_one_epoch(self):
        self.model.eval()
        validation_loss = 0
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for x_batch,y_batch in self.validation_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self.model(x_batch)
                loss = self.criterion(output,y_batch.ravel())
                validation_loss += loss.item()
                _,predicted = torch.max(output,1)

                n_samples += y_batch.size(0)
                n_correct += (predicted == y_batch.ravel()).sum().item()
        validation_loss = validation_loss/len(self.validation_loader)
        self.result['validation_loss'].append(validation_loss)
        self.result['validation_accuracy'].append(n_correct/n_samples)

    def test(self, verbose = True):
        print('Testing...') if verbose == True else None
        self.model.eval()
        true_label = self.y.label
        with torch.no_grad():
            predicted_label = []
            for x_batch,y_batch in self.test_loader:
                
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self.model(x_batch)
                _,predicted = torch.max(output,1)

                predicted_label.append(predicted.cpu().numpy())
        
        predicted_label = np.concatenate([p for p in predicted_label])
        self.predicted_label = predicted_label

        acc = round(accuracy_score(true_label,predicted_label),4)
        b_acc = round(balanced_accuracy_score(true_label,predicted_label),4)

        c = confusion_matrix(true_label,predicted_label)
        n_class_predictions = np.sum(c,axis = 0)
        f1 = round(f1_score(true_label, predicted_label,average='weighted',zero_division = 0),4)
        precision = round(precision_score(true_label, predicted_label,average='weighted',zero_division = 0),4)
        recall = round(recall_score(true_label, predicted_label,average='weighted',zero_division = 0),4) 

        self.result['test_score'] = EasyDict({'Acc':[acc], 
                                            'Balanced_acc': [b_acc], 
                                            'f1':[f1], 
                                            'precision': [precision], 
                                            'recall': [recall]})
        
        self.result['test_class_accuracy'] = [round(c[i,i]/n_class_predictions[i],3) for i in range(len(n_class_predictions))]
        self.result['test_confusion_matrix'] = c

        if verbose == True:
            print(f'Accuracy : {acc}, Balanced accuracy: {b_acc}') 
            print(f'Class accuracy: {self.result["test_class_accuracy"]}')
            print('Finished testing!')

    def train(self, early_stop = True, patience = 5, min_delta = 0.01, verbose = True):

        if self.data_available == False:
            raise RuntimeError('Please specify input data by Trainer.get_data().')
        if self.data_loader_available == False:
            self.get_loader()

        early_stopper = EarlyStopper(patience=patience, min_delta = min_delta) if early_stop == True else None
        print('Training...') if verbose == True else None
        for epoch in range(self.config.n_epochs):
            self.train_one_epoch()
            self.validate_one_epoch()
            if verbose == True:
                if (epoch %10 == 0) or (epoch == self.config.n_epochs - 1):
                    str1 = f"Epoch [{epoch+1}/{self.config.n_epochs}]"
                    str2 = f"Train loss: {self.result['training_loss'][epoch]:.4f}"
                    str3 = f"Validation loss: {self.result['validation_loss'][epoch]:.4f}"
                    str4 = f"Validation accuracy: {self.result['validation_accuracy'][epoch]:.4f}"
                    msg = str1 + ' | ' + str2 + ' | ' + str3 + ' | ' + str4
                    print(msg) 
            if early_stop == True:
                if early_stopper.early_stop(self.result['validation_loss'][-1]):      
                    print(f'Early stopped at epoch {epoch} after {patience} epochs of changes less than {min_delta}. Validation loss: {self.result["validation_loss"][-1]:.4f}')       
                    break        
        print('Finished training!') if verbose == True else None

    def reset(self):
        self.initialize_weights()
        self.result = {'training_loss': [], 'validation_loss': [],'validation_accuracy': [], 
                       'test_class_accuracy': [], 'test_score': [], 'test_confusion_matrix': []}
        
    def write_log(self, description):
        path = utils.path
        date = str(datetime.datetime.now())[:-7]
        if not os.path.exists('log'):
            os.makedirs('log')
        with open(os.path.join(path,'log',f'{self.model.__arch__}_session_result.txt'),'a') as f:
            f.writelines([
                        f'======================================================================================\n',
                        f'Date: {date} | Description: {description} | Model version: {self.model.__version__}\n',
                        f'Optimizer: {self.config.optimizer} | Device: {self.device} | Epochs: {self.config.n_epochs} | Learning rate: {self.config.lr} | Batch size: {self.config.batch_size}\n',
                        f"Test class accuracy: {self.result['test_class_accuracy']}\n",
                        ])
            f.writelines([f"{s}: {self.result['test_score'][s][0]}    " for s in self.result['test_score'].keys()])  
        with open(os.path.join(path,'log',f'{self.model.__arch__}_session_log.txt'),'a') as f:
            f.writelines([
                        f'======================================================================================\n',
                        f'Date: {date} | Description: {description} | Model version: {self.model.__version__}\n',
                        f'Optimizer: {self.config.optimizer} | Device: {self.device} | Epochs: {self.config.n_epochs} | Learning rate: {self.config.lr} | Batch size: {self.config.batch_size}\n',
                        f"Training loss: {np.round(self.result['training_loss'],2)}\n",
                        f"Validation loss: {np.round(self.result['validation_loss'],2)}\n",
                        f"Validation accuracy: {np.round(self.result['validation_accuracy'],2)}\n",
                        ])  
                            
    def plot_result(self, savefig = True, name = ''):
        # Learning curves
        train_loss = self.result['training_loss']
        val_loss = self.result['validation_loss']
        val_acc = self.result['validation_accuracy']

        # test scores
        test_score = self.result['test_score']
        scores = list(self.result['test_score'].keys())
        
        # Confusion matrices
        cf = dc(self.result['test_confusion_matrix'])
        cf = cf.astype(float)
        n_preds = np.sum(cf, axis = 0)
        for col in range(cf.shape[0]):
            for row in range(cf.shape[0]):
                try:
                    cf[row, col] = round(cf[row, col]/n_preds[col],2)
                except:
                    cf[row, col] = 0

        f, ax = plt.subplots(1,3,figsize = (12,4))
        ax[0].plot(train_loss,'r',label = 'train loss')
        ax[0].plot(val_loss,'b', label = 'validation_loss')
        ax[0].plot(val_acc,'b--', label = 'validation_accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_title('Traininng loss and validation accuracy')
        ax[0].grid()
        ax[0].legend()

        w = 0.3
        h = [test_score[k][0] for k in scores]
        plt.bar
        ax[1].bar(np.arange(len(scores)), h, width = w)
        ax[1].set_xticks(np.arange(len(scores)),scores)
        ax[1].set_title('Test accuracy and weighted scores')
        ax[1].set_ylim(0,1)

        sns.heatmap(cf, ax = ax[2], annot= True, cmap = 'YlGn')
        ax[2].set_title('Confusion matrix')     
        ax[2].xaxis.set_ticklabels(list(utils.reverse_labels_dict.values()))
        ax[2].yaxis.set_ticklabels(list(utils.reverse_labels_dict.values()))
        plt.tight_layout()
        if savefig == True:
            if not os.path.exists('log'):
                os.makedirs('log')
            d = str(datetime.date.today())
            p = os.path.join(os.getcwd(), 'log', f'{self.model.__arch__}_{d}_{name}')
            plt.savefig(p)

    def save_checkpoint(self, description):
        # description: Write some description about the data used or simply an empty string
        archi = self.model.__arch__
        date = str(datetime.date.today())
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists(f'checkpoints/{archi}'):
            os.makedirs(f'checkpoints/{archi}')
        saved_name = f'arch-{self.model.__arch__}.window-{self.config.window_size}.'
        saved_name += f'method-{self.config.method}.scale-{self.config.scale}.optimizer-{self.config.optimizer}.'
        saved_name += f'epochs-{self.config.n_epochs}.lr-{self.config.lr}.batchsize-{self.config.batch_size}.'
        dir = f'checkpoints/{archi}/' + saved_name + date + '.' + description + '.json'
        torch.save(self.model, dir)
        print(f'Parameters saved to "{dir}".')

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if np.abs(validation_loss - self.min_validation_loss) > self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False       







# ========================================================================================================================
class kfold_CV():
    def __init__(self, model, config, n_folds):
        self.trainer = Trainer(model,config)
        self.n_folds = n_folds

    def fit(self, X, y):
        self.trainer.fit(X,y)

    def CV(self, verbose = True):
        seed = [10*n for n in range(self.n_folds-1)] + [28]
        k_fold_result_ = []
        for i in range(self.n_folds):
            print(f'=================== Fold {i+1}, seed {seed[i]}===================') if verbose == True else None
            self.trainer.reset()
            self.trainer.get_loader(r = 0.1, random_state = seed[i])
            self.trainer.train(verbose = verbose)
            self.trainer.test(verbose = verbose)
            k_fold_result_.append(self.trainer.result)
        self.k_fold_result_ = k_fold_result_

    def write_log(self):
        summary = self.summarize()
        date = str(datetime.datetime.now())[:-7]
        cl = utils.reverse_labels_dict
        with open('kfold_log.txt','a') as f:
            f.writelines([
                        f'======================================================================================\n'
                        f'Date:{date}  |  Model version: {self.trainer.model.__version__}  |  Device:{self.trainer.device}  |  Epochs: {self.trainer.config.n_epochs}  |  Learning rate: {self.trainer.config.lr}  | Batch size: {self.trainer.config.batch_size}\n',
                        f'{self.n_folds}-fold max/Q1/Q2/Q3/min/mean/sd accuracy\n',
                        f"Accuracy: {summary['Global'].tolist()}\n",
                        f"Balanced accuracy: {summary['Balanced'].tolist()}\n"
                        ])
            for i in cl.keys():
                f.writelines(f"{cl[i]}: {summary[cl[i]].tolist()}\n")
    
    def summarize(self):
        index = ['max','Q1','Q2','Q3','min','mean','sd']
        summ = {}
        acc = [self.k_fold_result_[i]['test_score']['Acc'][0] for i in range(10)]
        b_acc = [self.k_fold_result_[i]['test_score']['Balanced_acc'][0] for i in range(10)]
        summ['Global'] = [np.max(acc), np.quantile(acc,0.25), np.quantile(acc,0.5), np.quantile(acc,0.75), np.min(acc),np.mean(acc), np.std(acc)]
        summ['Balanced'] = [np.max(b_acc), np.quantile(b_acc,0.25), np.quantile(b_acc,0.5), np.quantile(b_acc,0.75), np.min(b_acc),np.mean(b_acc), np.std(b_acc)]
        cl = utils.reverse_labels_dict
        class_acc_summary = np.array([self.k_fold_result_[i]['test_class_accuracy'] for i in range(10)])

        for i in cl.keys():
            class_acc = class_acc_summary[:,i]
            summ[cl[i]] = [np.max(class_acc), np.quantile(class_acc,0.25), np.quantile(class_acc,0.5), np.quantile(class_acc,0.75), np.min(class_acc),
                                       np.mean(class_acc), np.std(class_acc)]
        summ = pd.DataFrame(summ,index = index)
        summ = summ.apply(lambda x: round(x,3))
        return summ
    
    def plot_summary(self): #Trainer.result has train loss, valid acc, test score, test class accuracy and test confusion matrix
        # Learning curves
        train_loss = np.array([self.k_fold_result_[i]['training_loss'] for i in range(self.n_folds)])
        train_loss_mean = np.mean(train_loss, axis = 0)
        train_loss_sd = np.std(train_loss, axis = 0)

        val_acc = np.array([self.k_fold_result_[i]['validation_accuracy'] for i in range(self.n_folds)])
        val_acc_mean = np.mean(val_acc, axis = 0)
        val_acc_sd = np.std(val_acc, axis = 0)

        val_loss = np.array([self.k_fold_result_[i]['validation_loss'] for i in range(self.n_folds)])
        val_loss_mean = np.mean(val_loss, axis = 0)
        val_loss_sd = np.std(val_loss, axis = 0)

        # Test scores
        scores = pd.concat([pd.DataFrame(self.k_fold_result_[i]['test_score']) for i in range(self.n_folds)]).to_numpy()
        
        # class accuracy
        class_accuracy = np.array([self.k_fold_result_[i]['test_class_accuracy'] for i in range(self.n_folds)])
        
        # Sum of confusion matrix
        sum_cf = np.zeros(self.k_fold_result_[0]['test_confusion_matrix'].shape)
        for i in range(self.n_folds):
            sum_cf +=self.k_fold_result_[i]['test_confusion_matrix']

        sum_cf = sum_cf.astype(float)
        n_preds = np.sum(sum_cf, axis = 0)
        for col in range(sum_cf.shape[0]):
            for row in range(sum_cf.shape[0]):
                try:
                    sum_cf[row, col] = round(sum_cf[row, col]/n_preds[col],2)
                except:
                    sum_cf[row, col] = 0

        f, ax = plt.subplots(1,4,figsize = (20,4))
        ax[0].set_title('Loss & validation accuracy')
        ax[0].fill_between(np.arange(len(train_loss_mean)), train_loss_mean - train_loss_sd, train_loss_mean + train_loss_sd,alpha = 0.2, color = 'r')
        ax[0].fill_between(np.arange(len(val_loss_mean)), val_loss_mean - val_loss_sd, val_loss_mean + val_loss_sd,alpha = 0.2, color = 'b')
        ax[0].fill_between(np.arange(len(val_acc_mean)), val_acc_mean - val_acc_sd, val_acc_mean + val_acc_sd,alpha = 0.2, color = 'm')
        ax[0].plot(train_loss_mean, 'r', label = 'training_loss')
        ax[0].plot(val_loss_mean, 'b', label = 'validation_loss')
        ax[0].plot(val_acc_mean, 'm--', label = 'validation_accuracy')
        ax[0].grid()
        ax[0].legend()

        ax[1].boxplot(scores)
        ax[1].set_xticks(np.arange(1,6),['Accuracy','     B-accuracy','f1','precision','recall'])
        ax[1].set_title('Test scores')
        ax[1].set_ylim(0.4,1)

        ax[2].boxplot(class_accuracy)
        ax[2].set_xticks(np.arange(1,8), utils.original_labels.values())
        ax[2].set_title('Class accuracy')
        ax[2].set_ylim(0.4,1) 

        sns.heatmap(sum_cf, ax = ax[3], annot = True, cmap = 'YlGn', cbar = False, 
                    xticklabels= utils.labels_dict.keys(), yticklabels= utils.labels_dict.keys())        
        ax[3].set_title('Confusion matrix')     



from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score,recall_score, confusion_matrix
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle 
import preprocessing
import utils
import time
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from MLDataGenerator import get_feature_matrix
from easydict import EasyDict
import datetime 
import os
# ======================= MODEL ===========================
class Trainer():
    def __init__(self, clf):
        self.model = clf
        self.model_copy = dc(self.model)
        self.result_ = EasyDict({})
        
    def reset(self):
        self.model = dc(self.model_copy)
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def train(self, verbose = False):
        # Training model
        print('Training...') if verbose == True else None
        start = time.perf_counter()
        self.model.fit(self.X_train, self.y_train)
        end = time.perf_counter()
        training_time = round(end-start,3)
        self.result_['training_time'] = training_time
        print('Training time:', training_time) if verbose == True else None

    def predict(self, X_test, y_test, verbose = False):
        # Prediction
        print('Predicting ...') if verbose == True else None
        y_pred = self.model.predict(X_test)

        # Scoring
        cf = np.round(confusion_matrix(y_test, y_pred, normalize='pred'),2)
        acc = round(accuracy_score(y_test, y_pred),3)
        balanced_acc = round(balanced_accuracy_score(y_test, y_pred),3)
        f1 = round(f1_score(y_test, y_pred,average='weighted',zero_division = 0),3)
        precision = round(precision_score(y_test, y_pred, average='weighted',zero_division = 0),3)
        recall = round(recall_score(y_test, y_pred, average='weighted',zero_division = 0),3)
        
        self.result_['test_scores'] = EasyDict({'Acc':acc, 'Balanced_acc':balanced_acc, 'f1':f1, 'precision':precision, 'recall':recall})
        self.result_['test_confusion_matrix'] = cf
        print('Finished testing.') if verbose == True else None

    # def plot_result(self, savefig = True, name = ''):
    #     # Learning curves
    #     train_loss = self.result['training_loss']
    #     val_loss = self.result['validation_loss']
    #     val_acc = self.result['validation_accuracy']

    #     # test scores
    #     test_score = self.result['test_score']
    #     scores = list(self.result['test_score'].keys())
        
    #     # Confusion matrices
    #     cf = dc(self.result['test_confusion_matrix'])
    #     cf = cf.astype(float)
    #     n_preds = np.sum(cf, axis = 0)
    #     for col in range(cf.shape[0]):
    #         for row in range(cf.shape[0]):
    #             try:
    #                 cf[row, col] = round(cf[row, col]/n_preds[col],2)
    #             except:
    #                 cf[row, col] = 0

    #     f, ax = plt.subplots(1,3,figsize = (12,4))
    #     ax[0].plot(train_loss,'r',label = 'train loss')
    #     ax[0].plot(val_loss,'b', label = 'validation_loss')
    #     ax[0].plot(val_acc,'b--', label = 'validation_accuracy')
    #     ax[0].set_xlabel('Epoch')
    #     ax[0].set_title('Traininng loss and validation accuracy')
    #     ax[0].grid()
    #     ax[0].legend()

    #     w = 0.3
    #     h = [test_score[k][0] for k in scores]
    #     plt.bar
    #     ax[1].bar(np.arange(len(scores)), h, width = w)
    #     ax[1].set_xticks(np.arange(len(scores)),scores)
    #     ax[1].set_title('Test accuracy and weighted scores')
    #     ax[1].set_ylim(0,1)

    #     sns.heatmap(cf, ax = ax[2], annot= True, cmap = 'YlGn')
    #     ax[2].set_title('Confusion matrix')     
    #     ax[2].xaxis.set_ticklabels(list(utils.reverse_labels_dict.values()))
    #     ax[2].yaxis.set_ticklabels(list(utils.reverse_labels_dict.values()))
    #     plt.tight_layout()
    #     # if savefig == True:
    #     #     if not os.path.exists('log'):
    #     #         os.makedirs('log')
    #     #     d = str(datetime.date.today())
    #     #     p = os.path.join(os.getcwd(), 'log', f'{self.model.__type__}_{d}_{name}')
    #     #     plt.savefig(p)
    
    def kfold_CV(self, cv = 7, verbose = 0):
        scores = ['accuracy','balanced_accuracy','f1_weighted','precision_weighted','recall_weighted']
        self.cv_results = cross_validate(self.model, self.X_train, self.y_train, scoring = scores, cv=cv, n_jobs=-1,verbose = verbose, return_estimator= True)
        cv_scores = list(self.cv_results.keys())
        cv_scores.remove('estimator')
        summary = {}
        for key in cv_scores:
            summary[key] = [np.round(np.mean(self.cv_results[key]),2), np.round(np.std(self.cv_results[key]),2)]
        best_estimator_index = np.argmax(self.cv_results['test_accuracy'])
        cv_estimator = self.cv_results['estimator'][best_estimator_index]
        self.model = cv_estimator
        self.cv_summary = summary

    def write_log(self, description):
        path = utils.path
        date = str(datetime.datetime.now())[:-7]
        if not os.path.exists('log/ML'):
            os.makedirs('log/ML')
        with open(os.path.join(path,'log/ML',f'session_result.txt'),'a') as f:
            f.writelines([
                        f'======================================================================================\n',
                        f'Date: {date} | Description: {description}\n' 
                        f'===> CV result \n'
                        ]
                        + [f'{key}: {self.cv_summary[key][0]} +- {self.cv_summary[key][1]}\n' for key in self.cv_summary.keys()]
                        )
            f.writelines([f'====> Test result \n'] + [f"{s}: {self.result_['test_scores'][s]}    " for s in self.result_['test_scores'].keys()])  
            
default_configs = EasyDict({'window_size': 1024, 'scope': 4, 'method': 'raw'})
class Classifier():
    def __init__(self, clf, configs = default_configs):
        self.model = clf
        self.model_loaded_status = False
        
        self.configs = configs
        self.window_size = configs.window_size
        self.scope = configs.scope
        self.hop_length = configs.window_size//configs.scope
        self.method = configs.method
        print('Configurations loaded.')

    def predict(self, wave_array, ana, verbose = False):
        # Prepare data
        print('Preparing data...') if verbose == True else None
        self.wave_array = wave_array
        self.ana = ana
        df, self.true_label = preprocessing.generate_test_data(wave_array, ana, window_size = self.window_size, 
                                                                    hop_length = self.hop_length, method = self.method)
        self.input = get_feature_matrix(df)
        
        print('Predicting...') if verbose == True else None
        predicted_label = self.model.predict(self.input)

        #map to original labels  
        predicted_label = pd.Series(predicted_label).map({0: 1, 1: 2, 2: 4, 3: 6, 4: 7, 5: 8, 6: 5}).to_numpy() 

        self.predicted_label = predicted_label 

        self.cf = confusion_matrix(self.true_label, predicted_label)
        self.acc = accuracy_score(self.true_label,predicted_label)
        self.b_acc = balanced_accuracy_score(self.true_label,predicted_label)
        self.f1 = round(f1_score(self.true_label, predicted_label,average='weighted',zero_division = 0),3)
        self.precision = round(precision_score(self.true_label, predicted_label,average='weighted',zero_division = 0),3)
        self.recall = round(recall_score(self.true_label, predicted_label,average='weighted',zero_division = 0),3)

        n_class_predictions = np.sum(self.cf,axis = 0)
        self.class_acc = [round(self.cf[i,i]/n_class_predictions[i],3) for i in range(len(n_class_predictions))]
        print(f'Accuracy: {round(self.acc,3)}, Balanced accuracy: {round(self.b_acc,3)}')
        
        print('Aggregating predictions...') if verbose == True else None
            
        # Write results in form of analysis files
        n_windows = len(predicted_label)//self.scope
        time = [0] # Make time marks
        for i in range(n_windows):
            time.append((self.window_size+i*self.scope*self.hop_length)/100)

        agg_pred = [] # aggregating consecutive predictions
        for i in range(n_windows):
            cl,c = np.unique(predicted_label[i*self.scope:(i+1)*self.scope], return_counts=True)
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
    

# def plot_result(result_dict):
#     _,((ax1,ax2,ax3,ax4)) = plt.subplots(1,4,figsize = (16,3),sharex = True,sharey = True)

#     model_name = ['DecTree','LR','SVC','RF','GB','Ada','XGB']
#     n_model = len(model_name)

#     xtick = np.arange(0,n_model)
#     w = 0.2

#     r = result_dict
#     ax1.bar(xtick+w*np.ones(n_model),r['Balanced_acc'],width = w)
#     ax2.bar(xtick+w*np.ones(n_model),r['f1'],width = w)
#     ax3.bar(xtick+w*np.ones(n_model),r['precision'],width = w)
#     ax4.bar(xtick+w*np.ones(n_model),r['recall'],width = w)

#     ax1.set_title('Balanced Accuracy')
#     ax2.set_title('f1')
#     ax3.set_title('Precision')
#     ax4.set_title('Recall')
#     ax1.set_xticks(ticks = xtick,labels = model_name,rotation = 30)
#     ax2.set_xticks(ticks = xtick,labels = model_name,rotation = 30)
#     ax3.set_xticks(ticks = xtick,labels = model_name,rotation = 30)
#     ax4.set_xticks(ticks = xtick,labels = model_name,rotation = 30)

    
def save(model,path):
    # save the model to disk
    pickle.dump(model, open(path, 'wb'))

# ======================= VISUALIZATION ===========================
# classes = ['np','c','e1','e2','f','g','pd']
list_variables = ['mean', 'sd', 'sk', 'zcr', 'hurst', 'energy', 'sample_entropy', 
                'permutation_entropy', 'spectral_entropy', 'spectral_centroid', 'spectral_flatness']
original_labels = {1:'np', 2:'c', 4:'e1', 5:'e2', 6:'f', 7:'g', 8:'pd'}

def onevarplot(feature_matrix: np.ndarray, label: np.ndarray, var: str, n_obs: int = None):
    '''
        Input:
            feature_matrix: a matrix with 11 features calculated previously
            label: corresponding label to each row
            var: which variable (column to plot)
            n_obs: how many observation to plot
            xlab: label of axis x
            ylab: label of axis y
        Output:
            A colored plot of a column of feature_matrix
    '''
    plt.figure(figsize=(5,5))
    classes, counts = np.unique(label,return_counts=True)

    if not isinstance(n_obs,int):
        n_obs = min(counts)

    f1 = feature_matrix[:,list_variables.index(var)]
    split = []

    for i in range(len(classes)):
        split.append(f1[label == classes[i]])

    for i in range(len(classes)):
        plt.scatter(np.arange(0,n_obs),split[i][0:n_obs],label = classes[i])

    plt.xlabel('n-th Observation')
    plt.ylabel(var)
    plt.legend(ncols = 3)

def twovarplot(feature_matrix: np.ndarray, label: np.ndarray, var1: int, var2: int, n_obs: int = None):
    plt.figure(figsize=(5,5))
    classes = np.unique(label)

    f1 = feature_matrix[:,list_variables.index(var1)]; f2 = feature_matrix[:,list_variables.index(var2)]
    split1 = []; split2 = []

    for i in range(len(classes)):
        split1.append(f1[label == classes[i]])
        split2.append(f2[label == classes[i]])

    if n_obs == None:
        n_obs = 100
    
    for i in range(len(classes)):

        plt.scatter(split1[i][0:n_obs], split2[i][0:n_obs], label = classes[i])

    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend(loc = 'best', ncols = 2)

def twovarplot_multiple(feature_matrix: np.ndarray, label: np.ndarray, n_obs: int = None, n_pairs: int = 6):
    classes = np.unique(label)

    v = []
    for n in range(n_pairs):
        v.append(np.random.randint(0,11,2))

    fig, ax = plt.subplots(2,3,figsize = (5*3,5*2))

    n_ax = 0
    for var1, var2 in v:
        f1 = feature_matrix[:,var1]; f2 = feature_matrix[:,var2]
        split1 = []; split2 = []

        for i in range(len(classes)):
            split1.append(f1[label == classes[i]])
            split2.append(f2[label == classes[i]])

        for i in range(len(classes)):
            ax[n_ax//3,n_ax%3].scatter(split1[i][0:n_obs],split2[i][0:n_obs],label = original_labels[classes[i]])

        ax[n_ax//3,n_ax%3].set_xlabel(list_variables[var1])
        ax[n_ax//3,n_ax%3].set_ylabel(list_variables[var2])
        ax[n_ax//3,n_ax%3].legend()
        n_ax+=1 

        # kfold = StratifiedKFold(n_splits = n_fold)
        # kfold_idx = kfold.split(self.X_train, self.y_train)
        # self.kfold_results_ = EasyDict({})
        # self.kfold_results_['validation_scores'] = EasyDict({'Acc':[],
        #                                                     'Balanced_acc':[],
        #                                                     'f1':[],
        #                                                     'precision':[],
        #                                                     'recall':[]})
        # self.kfold_results_['validation_confusion_matrix'] = []        

        # for (train_idx, val_idx) in kfold_idx:
        #     clf = dc(self.model)
        #     X_train, X_val = self.X_train[train_idx], self.X_train[val_idx]
        #     y_train, y_val = self.y_train[train_idx], self.y_train[val_idx]

        #     clf.fit(X_train,y_train)
        #     y_pred = clf.predict(X_val)

        #     cf = confusion_matrix(y_val, y_pred)
        #     acc = round(accuracy_score(y_val,y_pred),3)
        #     balanced_acc = round(balanced_accuracy_score(y_val,y_pred),3)
        #     f1 = round(f1_score(y_val,y_pred,average='weighted',zero_division = 0),3)
        #     precision = round(precision_score(y_val,y_pred,average='weighted',zero_division = 0),3)
        #     recall = round(recall_score(y_val,y_pred,average='weighted',zero_division = 0),3)
        #     self.kfold_results_['validation_confusion_matrix'].append(cf)
        #     self.kfold_results_['validation_scores']['Acc'].append(acc)
        #     self.kfold_results_['validation_scores']['Balanced_acc'].append(balanced_acc)
        #     self.kfold_results_['validation_scores']['f1'].append(f1)
        #     self.kfold_results_['validation_scores']['precision'].append(precision)
        #     self.kfold_results_['validation_scores']['recall'].append(recall)

        # self.kfold_mean_scores_ = EasyDict({})
        # for score in self.kfold_results_['validation_scores'].keys():
        #     self.kfold_mean_scores_[score] = [np.mean(self.kfold_results_['validation_scores'][score])]
        # self.kfold_total_confusion_matrix_ = np.sum(self.kfold_results_['validation_confusion_matrix'])

        # print(pd.DataFrame(self.kfold_mean_scores_))
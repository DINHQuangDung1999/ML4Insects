import os
import numpy as np
import pandas as pd 
from easydict import EasyDict 
from copy import deepcopy as dc
from models import MLP, CNN1D, CNN2D, ResNet

# ============================= Label map =============================
path = dc(os.getcwd())
labels_dict = {'np':0,'c':1,'e1':2,'f':3,'g':4,'pd':5,'e2':6}
reverse_labels_dict = {0:'np',1:'c',2:'e1', 3:'f',4:'g',5:'pd', 6:'e2'}
original_labels = {1:'np', 2:'c', 4:'e1', 5:'e2', 6:'f', 7:'g', 8:'pd'}

# ============================= Format =============================
# Two levels split for 2-level classifier
def format_data(data, label, n_stage = 1):
    
    if n_stage == 1: 
        label = pd.Series(label).map({1: 0, 2: 1, 4: 2, 6: 3, 7: 4, 8: 5, 5: 6}).to_numpy()
        return EasyDict({'data':data,'label':label})
    
    elif n_stage == 2:
        binary_label = []
        for i in range(len(label)):
            if label[i] == 5:
                binary_label.append(1)
            else:
                binary_label.append(0)
        binary_label = np.array(binary_label)

        # Multilabels for level 2 classifier
        none2_label = label[label != 5]
        none2_label = pd.Series(none2_label).map({1: 0, 2: 1, 4: 2, 6: 3, 7: 4, 8: 5}).to_numpy()

        return EasyDict({'level1':{'data':data, 'label': binary_label},
                        'level2':{'data':data[label != 5],'label': none2_label}})      
    else:
        raise RuntimeError('Param "n_stage" should be 1 or 2.')
    
def get_one_level_split(df,l):
    label = pd.Series(l).map({1: 0, 2: 1, 4: 2, 6: 3, 7: 4, 8: 5, 5: 6}).to_numpy()
    print(f'Labels map (from/to): {{1: 0, 2: 1, 4: 2, 6: 3, 7: 4, 8: 5, 5: 6}}')
    return EasyDict({'data':df,'label':label})

def get_two_levels_split(df,l):
    '''
        Output: EasyDict of two level data and labels
    '''
    # Binary labels for level 1 classifier
    binary_label = []
    for i in range(len(l)):
        if l[i] == 5:
            binary_label.append(1)
        else:
            binary_label.append(0)
    binary_label = np.array(binary_label)

    # Multilabels for level 2 classifier
    none2_label = l[l != 5]
    none2_label = pd.Series(none2_label).map({1: 0, 2: 1, 4: 2, 6: 3, 7: 4, 8: 5}).to_numpy()

    return EasyDict({'level1':{'data':df, 'label': binary_label},
                     'level2':{'data':df[l != 5],'label': none2_label}})

# def get_two_levels_split(df,l):
#     '''
#         Output: EasyDict of two level data and labels
#     '''
#     # Binary labels for level 1 classifier
#     e_vs_rest_label = dc(l).tolist()
#     for i in range(len(l)):
#         if l[i] == 5 or l[i] == 4: #if e, then label 0
#             e_vs_rest_label[i] = 0
#         else: #otherwise label 1
#             e_vs_rest_label[i] = 1 
#     e_vs_rest_label = np.array(e_vs_rest_label)

#     e1_vs_e2_label = dc(l)
#     for i in range(len(l)):
#         if  l[i] != 5 or l[i] != 4: #if label e2 then label 1
#             e1_vs_e2_label[i] = 9
#     for i in range(len(l)):
#         if  l[i] == 5: #if label e2 then label 1
#             e1_vs_e2_label[i] = 1   
#         elif l[i] == 4: #
#             e1_vs_e2_label[i] = 0
#         else:
#             e1_vs_e2_label[i] = 9
#     e1_vs_e2_label = np.array(e1_vs_e2_label[e1_vs_e2_label != 9])
#     # Label of the rest
#     not_e_label = dc(l[(l!=5) & (l!=4)])
#     not_e_label = pd.Series(not_e_label).map({1: 0, 2: 1, 6: 2, 7: 3, 8: 4}).to_numpy()

#     return EasyDict({'level1_e_vs_rest':{'data':df, 'label': e_vs_rest_label},
#                      'level2_e1_vs_e2':{'data':df[(l == 5) | (l == 4)],'label': e1_vs_e2_label},
#                      'level2_not_e':{'data':df[(l!=5) & (l!=4)], 'label':not_e_label}})




# ============================= Read input =============================
def get_filename(name):
    list_files = os.listdir(os.path.join(path,'data',name))
    unique = []
    for name in list_files:
        name = name.split('.')[0]
        unique.append(name)
    unique = np.unique(unique)
    return unique

def read_signal(filename: str, signal_only = False) -> tuple:
    '''
        Input: 
            filename: name of the recording
        Output:
            tuple of data signal and analysis dataframe 
    '''
    # Read signals
    d = []
    source = filename.split('_')[0]
    readme = pd.read_csv(os.path.join(path,'data','readme.csv'),index_col='name')
    n_extension = readme.loc[source, 'recording_hour']
    extension = [f'.A0{n+1}' for n in range(n_extension)]
    
    # Read the signal (.A0x) files
    for i in range(len(extension)):
        file_path = os.path.join(path,'data', source, filename + extension[i])
        x = pd.read_csv(file_path,low_memory = False,delimiter=";",header = None, usecols=[1])
        d.append(x)
    data = pd.concat(d)
    data = data.to_numpy().squeeze(1)

    # Read the analysis (.ANA) files

    if signal_only == False:
        ana_file_path = os.path.join(path,'data',f'{source}_ANA', filename + '.ANA')
        ana = pd.read_csv(ana_file_path, encoding='utf-16', delimiter = '\t',header = None, usecols=[0,1])
        ana.columns = ['label','time']
        ana = ana[(ana['label'] != 9) & (ana['label'] != 10) & (ana['label'] != 11)]
        ana.drop_duplicates(subset='time',inplace=True)
        ana.index = np.arange(len(ana))
        return [data, ana.astype(int)]
    return data

def get_index(ana):
    '''
    Input.
        ana: analysis file
    Output.
        index: dictionary containing intervals of all wave types found in the analysis file 
            1 ~ np, 2 ~ c, 4 ~ e1, 5 ~ e2, 6 ~ f, 7 ~ g, 8 ~ pd
    '''
    index = {}
    n = len(ana)
    for i in range(0,n-1):
        start, end = ana.loc[i:i+1,'time'].tolist()
        if ana.loc[i,'label'] == 1:
            try:
                index['np'].append([int(start),int(end)])
            except:
                index['np'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 2:
            try:
                index['c'].append([int(start),int(end)])
            except:
                index['c'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 4:
            try:
                index['e1'].append([int(start),int(end)])
            except:
                index['e1'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 5:
            try:
                index['e2'].append([int(start),int(end)])
            except:
                index['e2'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 6:
            try:
                index['f'].append([int(start),int(end)])
            except:
                index['f'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 7:
            try:
                index['g'].append([int(start),int(end)])
            except:
                index['g'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 8:
            try:
                index['pd'].append([int(start),int(end)])
            except:
                index['pd'] = [[int(start),int(end)]]
    return index 

def extract_sample(wave_array,ana,wave_type,which):
    '''
        Extract one waveform sample from the whole signal
    '''
    wave_indices = get_index(ana)
    start,end = wave_indices[wave_type][which]
    return wave_array[start:end]

# ============================= Retrieve name =============================
#  train-test splits
def get_train_test_filenames(train_ratio = None, n_train = None, n_test = None, shuffle = False, name = None, seed = 10):
    np.random.seed(seed)
    if name is None:
        list_dir = [d for d in os.listdir('./data/') if not d.endswith('_ANA') if '.' not in d]
    else:
        if isinstance(name, list):
            list_dir = name
    splits = {}

    for name in list_dir: 
        recording_names = get_filename(name)
        if shuffle == True:
            np.random.shuffle(recording_names)
        if (train_ratio is not None):
            n = int(train_ratio*len(recording_names))
            train_name = recording_names[:n]
            test_name = recording_names[n:]
        elif (n_train is not None and n_test is not None):
            n = min(n_train,len(recording_names)-1)
            train_name = recording_names[:n]
            test_name = recording_names[n:n+n_test]       
        # print(n, name)     
        splits[name] = (train_name, test_name)
    return splits

# ============================= Retrieve models =============================
def get_model(architecture):
    if architecture == 'MLP':
        return MLP()
    elif architecture == 'CNN1D':
        return CNN1D()
    elif architecture == 'CNN2D':
        return CNN2D()
    elif architecture == 'ResNet':
        return ResNet()
    else:
        raise RuntimeError('Architecture must be one of MLP, CNN1D, CNN2D and ResNet.')

# ============================= Checkpoint =============================
import torch
import os
from datetime import date

def save_checkpoint(models, config, name = None):
    n_models = len(models)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists(f'checkpoints/{models[0].__type__}'):
        os.makedirs(f'checkpoints/{models[0].__type__}')
    dir = f'checkpoints/{models[0].__type__}'
    if n_models == 1:
        level = '1stage'
    elif n_models == 2:
        level = '2stage'
    for n in range(n_models):
        torch.save(models[n], dir + f'/{models[0].__type__}.{config.method}.{level}.{date.today()}.model{n+1}.pth')
        print(f'Parameters saved! "{models[0].__type__}.{config.method}.{level}.{date.today()}.model{n+1}.pth".')

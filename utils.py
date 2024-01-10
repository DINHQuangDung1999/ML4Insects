import os
import numpy as np
import preprocessing
import pandas as pd 
from easydict import EasyDict 
original_labels = {1:'np', 2:'c', 4:'e1', 5:'e2', 6:'f', 7:'g', 8:'pd'}

path = os.getcwd()
extension = ['.A01','.A02','.A03','.A04','.A05','.A06','.A07','.A08']
labels_dict = {'np':0,'c':1,'e1':2,'f':3,'pd':4,'g':5,'e2':6}
reverse_labels_dict = {0:'np',1:'c',2:'e1', 3:'f',4:'pd',5:'g', 6:'e2'}

# Two levels split for 2-level classifier
def get_one_level_split(df,l):
    label = pd.Series(l).map({1: 0, 2: 1, 4: 2, 6: 3, 7: 4, 8: 5, 5: 6}).to_numpy()
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

def get_filename(name):
    os.chdir(f'{path}\\{name}')
    files_list = os.listdir()
    unique = []
    for name in files_list:
        name = name.split('.')[0]
        unique.append(name)
    unique = np.unique(unique)
    os.chdir(path)
    return unique

def read_signal(filename: str) -> tuple:
    '''
        Input: 
            filename: name of the recording
        Output:
            tuple of data signal and analysis dataframe 
    '''
    s = filename.split('_')
    dir = s[0]
    
    os.chdir(os.path.join(path,dir))
    d = []
    for i in range(len(extension)):
        x = pd.read_csv(filename + extension[i],low_memory = False,delimiter=";",header = None,usecols=[1])
        d.append(x)
    data = pd.concat(d)
    data = data.to_numpy().squeeze(1)

    os.chdir(os.path.join(path,dir+'_ANA'))
    ana = pd.read_csv(filename + '.ANA',encoding='utf-16',delimiter = '\t',header = None,usecols=[0,1])
    ana.columns = ['label','time']
    ana.loc[:,'time'] = ana.loc[:,'time'].apply(lambda x: int(x*100))
    ana.drop_duplicates(subset='time',inplace=True)
    ana.index = np.arange(len(ana))
    
    os.chdir(path)
    return [data, ana.astype(int)]

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

#  train-test splits
import random
from random import sample
# ========================================================================
signal_16zt = get_filename('16zt')
signal_8zt = get_filename('8zt')
signal_0zt = get_filename('0zt')

random.seed(10)

# 16zt =====================================================================
test_idx_16zt = sample(range(1,len(signal_16zt)),5)
train_idx_16zt = list(set(np.arange(1,len(signal_16zt)))-set(test_idx_16zt))

# 8zt =====================================================================
test_idx_8zt = sample(range(1,len(signal_8zt)),5)
train_idx_8zt = list(set(np.arange(1,len(signal_8zt)))-set(test_idx_8zt))

# 0zt =====================================================================
test_idx_0zt = sample(range(1,len(signal_0zt)),5)
train_idx_0zt = list(set(np.arange(1,len(signal_0zt)))-set(test_idx_0zt))

splits = {'16zt': (signal_16zt[train_idx_16zt], signal_16zt[test_idx_16zt]),
             '8zt': (signal_8zt[train_idx_8zt], signal_8zt[test_idx_8zt]),
             '0zt': (signal_0zt[train_idx_0zt], signal_0zt[test_idx_0zt])}

small_splits = {'16zt': (signal_16zt[[0]], signal_16zt[[1]]),
             '8zt': (signal_8zt[[0]], signal_8zt[[1]]),
             '0zt': (signal_0zt[[0]], signal_0zt[[1]])}
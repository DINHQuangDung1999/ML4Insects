import pandas as pd
import numpy as np
import pywt 
import librosa
import os
from utils import read_signal

# PREPROCESSING TECHNIQUES
def downsampling(wave_array,ana = None,coef = 10):
    '''
        Downsample the signal by taking the average 
    '''
    new_length = len(wave_array)//coef

    downsampled_wave = []
    for i in range(new_length):
        downsampled_wave.append(np.mean(wave_array[i*10:(i+1)*10]))
        
    if ana is None:
        return np.array(downsampled_wave)
    else:
        downsampled_ana = pd.concat([ana.loc[:,'label'],ana.loc[:,'time'].apply(lambda x: x//10)],axis=1)
        return np.array(downsampled_wave), downsampled_ana

def outlier_filtering(wave_array,ana = {},option = 'whole'):
    '''
    Input:
        wave: wave signal of class np.ndarray
        ana: analysis file 
        option: 
            'whole' - apply outlier filter to each of the waves given by ana
            'indiv' - use when wave_array is a single wave
    Output:
        new signal whose outliers are set to zeros
    '''
    w = dc(wave_array)

    if option == 'whole':
        
        for i in range(len(ana)-1):

            start = ana.iloc[i,1].item()
            end = ana.iloc[i+1,1].item()

            wave_segment = w[start:end]

            Q1 = np.quantile(wave_segment,0.25)
            Q3 = np.quantile(wave_segment,0.75)
            IQR = abs(Q1-Q3)

            w_sub = wave_segment[(wave_segment > Q1-1.5*IQR) & (wave_segment< Q3+1.5*IQR)]
            median = np.median(w_sub)   

            for i in range(len(wave_segment)):
                if (wave_segment[i] < Q1-1.5*IQR) or (wave_segment[i]> Q3+1.5*IQR):
                    wave_segment[i] = median
        
        return [w,ana]
    
    elif option == 'indiv':
        
        Q1 = np.quantile(w,0.25)
        Q3 = np.quantile(w,0.75)
        IQR = abs(Q1-Q3) 
        w_sub = w[(w > Q1-1.5*IQR) & (w < Q3+1.5*IQR)]

        for i in range(len(w)):
            if (w[i] < Q1-1.5*IQR) or (w[i]> Q3+1.5*IQR):
                w[i] = np.median(w_sub)

        return w

### DATA RETRIEVERS
def generate_signal_dictionary(data = {},data_test = {}, verbose = False, outlier_filter = False, scale = False,
                               downsampling = False, denoising = False, data_splits: dict = None, name = '0zt'):

    train, test = data_splits[name]

    for i in range(len(train)):

        # Read data table and analysis file
        data[train[i]] = read_signal(train[i])

        # preprocessing option
        if outlier_filter == True:
            data[train[i]] = outlier_filtering(data[train[i]][0],data[train[i]][1])
        if downsampling == True:
            data[train[i]] = downsampling(data[train[i]][0],data[train[i]][1])
        if denoising == True:
            data[train[i]][0] = wavelet_denoising(data[train[i]][0],wavelet = 'sym4',n_level = 5)
        if scale == True:
            scaler = MinMaxScaler() # Scale the data to (0,1)
            data[train[i]][0] = scaler.fit_transform(data[train[i]][0].reshape(-1,1)).squeeze(1)

    for i in range(len(test)):

        # Read data table and analysis file
        data_test[test[i]] = read_signal(test[i])

        # preprocessing option
        if outlier_filter == True:
            data_test[test[i]] = outlier_filtering(data_test[test[i]][0],data_test[test[i]][1])
        if downsampling == True:
            data_test[test[i]] = downsampling(data_test[test[i]][0],data_test[test[i]][1])
        if denoising == True:
            data_test[test[i]][0] = wavelet_denoising(data_test[test[i]][0],wavelet = 'sym4',n_level = 5)
        if scale == True:
            scaler = MinMaxScaler() # Scale the data to (0,1)
            data_test[test[i]][0] = scaler.fit_transform(data_test[test[i]][0].reshape(-1,1)).squeeze(1)    
    if verbose == True:
        print(f'Outliers filtering: {str(outlier_filter)}')
        print(f'Scale: {str(scale)}')
    return data, data_test

def generate_data(wave_array, ana, window_size=1024, hop_length=256, method= 'raw'):
    '''
    Input:
        wave: wave signal of class np.ndarray (from func read_wave())
        ana: analysis file 
        window_size: size of fft window
        hop_length: the length of which the windows will be slided
    Output:
        data: np.ndarray containing windows of size 1024 of the input wave, shifted by hop_length, arranged row-wise
        label: labels corresponding to the rows
    '''

    d = []
    l = []

    tmp_hop = hop_length
    
    for i in range(0,len(ana)-1):

        start = ana['time'][i]
        end = ana['time'][i+1]
        wave_type  = ana['label'][i]

        if end-start > window_size: #if longer than window_size, segment into windows 
            pass

        else:#if shorter than window_size, pad both sides to get 2*window_size around the center hop length is reduced
                
            start, end = (start+end)//2 - window_size, (start+end)//2 + window_size
            hop_length = 128

        n_window = ((end-start)-window_size)//hop_length + 1

        for k in range(n_window):

            idx = np.arange(start + k*hop_length,start + k*hop_length + window_size)

            if method== 'fft':
                coef = np.abs(librosa.stft((wave_array[idx]),n_fft=window_size,center=False)).ravel()
                d.append(coef)

            elif method== 'spec_img':
                im = librosa.amplitude_to_db(np.abs(librosa.stft((wave_array[idx]) ,n_fft=128,center = False,hop_length=14)),ref=np.max)
                d.append(im)
            
            else:
                d.append((wave_array[idx]))

            l.append(wave_type)

        hop_length = tmp_hop

    data = np.stack([w for w in d])
    label = np.array(l)

    # if method== 'transformer':
    #     transfomer_data = []
    #     transformer_label = []

    #     n = len(data)//10
    #     for i in range(n):
    #         transfomer_data.append(data[i*10:(i+1)*10,:])
    #         transformer_label.append(label[i*10:(i+1)*10])
    #     transfomer_data = np.array(transfomer_data)
    #     data = transfomer_data
    #     label = transformer_label
        
    return data, label

from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc

def generate_model_data(data_dictionary,n = None,method = 'raw',window_size = 1024,hop_length = None,verbose = False):
    '''
    Input:
        :param data_dictionary: dict; dictionary of form [wave_array,wave analysis file] containing all the input files
        :param n: int; n first input files we would like to read
        :param scale: bool; scale the waveforms by MinMax or not
        :param window_size: int; size of fft/raw window
        :param hop_length: int; the length of which the windows will be slided
        :param verbose: bool; print description
    Output:
        data: concatenated array of each files data
        label: concatenated labels
    '''
    filenames = [*data_dictionary.keys()]
    d = []
    l = []

    if not isinstance(hop_length,int):
        hop_length = window_size//4      

    if not isinstance(n,int): #set n = number of input files by default
        n = len(filenames)

    for ind in range(n): 
        
        df = dc(data_dictionary[filenames[ind]][0])
        ana = data_dictionary[filenames[ind]][1] 

        features, labels = generate_data(df,ana,window_size=window_size,hop_length=hop_length, method = method)

        d.append(features)
        l.append(labels)

    data = np.concatenate([f for f in d])
    label = np.concatenate([lab for lab in l])

    if verbose == True:
        print(f'Signal processing method: {method}')
        print(f'Data shape: {data.shape}')
        cl, c = np.unique(label,return_counts=True)
        print('Class distribution (label/n_obs): '+ ', '.join(f'{cl[i]}: {c[i]}' for i in range(len(cl))))
            
    return data, label

def generate_test_data(wave_array,ana,window_size=1024,hop_length=256,method = 'raw'):
    '''
        Return: test data and true labeling 
    '''
    dense_labels = np.concatenate([[ana['label'].iloc[i]] * (ana['time'].iloc[i+1]-ana['time'].iloc[i])
                    for i in range(len(ana)-1)])
    
    n_windows = (len(wave_array)-window_size)//hop_length + 1
    d = []
    l = []
    for i in range(n_windows):
        idx = np.arange(i*hop_length, i*hop_length + window_size)
        
        if method == 'fft':
            coef = np.abs(librosa.stft((wave_array[idx]),n_fft=window_size,center=False)).ravel()
            d.append(coef)

        elif method == 'spec_img':
            im = librosa.amplitude_to_db(np.abs(librosa.stft((wave_array[idx]) ,n_fft=128,center = False,hop_length=14)),ref=np.max)
            d.append(im)
        
        else:
            d.append((wave_array[idx]))

        tmp_label = dense_labels[i*hop_length: i*hop_length+window_size]
        cl, c = np.unique(tmp_label,return_counts=True)
        l.append(cl[np.argmax(c)])
    d = np.array(d)
    l = np.array(l)
    return d, l

"""
Wavelet transforms
"""

def get_wavelet_coefficients(array, wavelet = 'sym4', n_level = 3):
    '''
        Input: 
            array: signal of the form np.ndarray
            wavelet: one of PyWavelet wavelets
            n_level: number of resolution
        Output: 
            tuple: (low_freq, high_freq) 
            low_freq: approximation coefficients of resolution 1 to n
            high_freq: detail coefficients of resolution 1 to n
    '''
    cA, cD = pywt.dwt(array,wavelet)
    low_freq = [cA]
    high_freq = [cD]
    for n in range(n_level-1):
        cA, cD = pywt.dwt(cA,wavelet)
        low_freq.append(cA)
        high_freq.append(cD)
        
    return low_freq, high_freq

def wavelet_denoising(wave, wavelet, n_level, threshold = 'global'):

    coeffs = pywt.wavedec(wave, wavelet, level=n_level)
    all_coeffs = np.concatenate([l for l in coeffs])

    if threshold == 'global':
        N = len(wave)
        sigma = np.median(np.abs(all_coeffs))/0.6745
        t = sigma*np.sqrt(2*np.log(N))

    elif threshold == 'std':
        std = np.std(all_coeffs,ddof = 1)
        t = 1.5*std
        
    elif threshold == 'ratio':
        t = 0.05*np.max(all_coeffs)
    
    else:
        t = threshold
        
    coeffs_thresholded = [pywt.threshold(c, t, mode='soft') for c in coeffs]
    denoised = pywt.waverec(coeffs_thresholded, wavelet)

    return denoised

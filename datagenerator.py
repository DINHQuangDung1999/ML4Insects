import numpy as np
import librosa
import pywt
from utils import read_signal, format_data
from preprocessing import quantile_outlier_filtering
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc

def generate_signal_dictionary(data_names, data_splits, outlier_filter = False, scale = False):
    '''
    :param scale: bool; scale the waveforms by MinMax or not
    '''
    data = {}; data_test = {}
    if isinstance(data_names, str):
        data_names = [data_names]
    for n in data_names:
        train, test = data_splits[n]
        for i in range(len(train)):
            # Read data table and analysis file
            data[train[i]] = read_signal(train[i])
            # preprocessing 
            if outlier_filter == True:
                data[train[i]][0] = quantile_outlier_filtering(data[train[i]][0])
            if scale == True:
                scaler = MinMaxScaler() # Scale the data to (0,1)
                data[train[i]][0] = scaler.fit_transform(data[train[i]][0].reshape(-1,1)).squeeze(1)

        for i in range(len(test)):
            # Read data table and analysis file
            data_test[test[i]] = read_signal(test[i])
            # preprocessing
            if outlier_filter == True:
                data_test[test[i]][0] = quantile_outlier_filtering(data_test[test[i]][0])
            if scale == True:
                scaler = MinMaxScaler() # Scale the data to (0,1)
                data_test[test[i]][0] = scaler.fit_transform(data_test[test[i]][0].reshape(-1,1)).squeeze(1)    
    print(f'There are {len(data)} recordings used for training and {len(data_test)} recordings used for testing')
    return data, data_test

def generate_input(wave_array, ana_file, window_size=1024, hop_length=256, method= 'raw'):
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
    dummy_hop = hop_length
    ana = dc(ana_file)
    ana.loc[:,'time'] = ana.loc[:,'time'].apply(lambda x: int(x*100))
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
            if start + k*hop_length + window_size > len(wave_array):
                break
            idx = np.arange(start + k*hop_length, start + k*hop_length + window_size)
            slice = wave_array[idx]
            if method == 'fft':
                fft_coef = (np.abs(librosa.stft(slice, n_fft = window_size, center=False))/np.sqrt(window_size)).ravel()
                d.append(fft_coef)
            elif method == 'dwt':
                dwt_coef = pywt.wavedec(slice, 'sym8', level = 2)[0]
                d.append(dwt_coef)
            elif method == 'spec_img':
                im = librosa.amplitude_to_db(np.abs(librosa.stft(slice, n_fft = 128, hop_length = 14, center = False))/np.sqrt(window_size), ref=np.max)
                d.append(im)
            elif method == 'raw':
                d.append(slice)
            else:
                raise RuntimeError ("Param 'method' should be one of 'fft', 'dwt', 'spec_img' or 'raw'.")
            
            l.append(wave_type)
        hop_length = dummy_hop

    data = np.stack([w for w in d])
    label = np.array(l)
        
    return data, label

def generate_data(data_names, data_splits, config, verbose = False):
    '''
    Input:
        :param data_dictionary: dict; dictionary of form [wave_array,wave analysis file] containing all the input files
        :param n: int; n first input files we would like to read
        
        :param window_size: int; size of fft/raw window
        :param hop_length: int; the length of which the windows will be slided 
        :param verbose: bool; print description
    Output:
        data: concatenated array of each files data
        label: concatenated labels
    '''
    method = config.method
    scale = config.scale
    window_size = config.window_size
    outlier_filter = config.outlier_filter
    hop_length = window_size//4

    dict_train, dict_test = generate_signal_dictionary(data_names, data_splits, outlier_filter, scale)
    d = []; l = []    
    for filename in dict_train.keys(): 
        df = dict_train[filename][0]; ana = dict_train[filename][1] 
        features, labels = generate_input(df, ana, window_size = window_size, hop_length= hop_length, method = method)
        d.append(features); l.append(labels)
    df_train = np.concatenate([f for f in d])
    label_train = np.concatenate([lab for lab in l])

    d = []; l = []    
    for filename in dict_test.keys(): 
        df = dict_test[filename][0]; ana = dict_test[filename][1] 
        features, labels = generate_input(df, ana, window_size = window_size, hop_length= hop_length, method = method)
        d.append(features); l.append(labels)
    df_test = np.concatenate([f for f in d])
    label_test = np.concatenate([lab for lab in l])

    train = format_data(df_train,label_train)
    test = format_data(df_test,label_test)
    
    if verbose == True:
        print(f'Signal processing method: {method} | Outliers filtering: {str(outlier_filter)} | Scale: {str(scale)}')
        print(f'Train/test shape: {df_train.shape}/{df_test.shape}')
        cl, c = np.unique(label_train,return_counts=True)
        print('Train distribution (label/ratio): '+ ', '.join(f'{cl[i]}: {round(c[i]/len(label_train),2)}' for i in range(len(cl))))
        cl, c = np.unique(label_test,return_counts=True)
        print('Test distribution (label/ratio): '+ ', '.join(f'{cl[i]}: {round(c[i]/len(label_test),2)}' for i in range(len(cl))))
        print(f'Labels map (from/to): {{1: 0, 2: 1, 4: 2, 6: 3, 7: 4, 8: 5, 5: 6}}')
            
    return train, test

##################################################################################################
def generate_test_data(wave_array, ana_file = None, window_size=1024, hop_length=256, method = 'raw'):
    '''
        Return: test data and true labeling 
    '''
    ana = dc(ana_file)
    if ana is not None:
        ana.loc[:,'time'] = ana.loc[:,'time'].apply(lambda x: int(x*100))
        dense_labels = np.concatenate([[ana['label'].iloc[i]] * (ana['time'].iloc[i+1]-ana['time'].iloc[i])
                        for i in range(len(ana)-1)])
    
    n_windows = (len(wave_array)-window_size)//hop_length + 1
    d = []
    l = []
    for i in range(n_windows):
        idx = np.arange(i*hop_length, i*hop_length + window_size)
        slice = wave_array[idx]
        if method == 'fft':
            fft_coef = (np.abs(librosa.stft(slice,n_fft=window_size,center=False))/np.sqrt(window_size)).ravel()
            d.append(fft_coef)
        elif method == 'dwt':
            dwt_coef = pywt.wavedec(slice, 'sym8', level = 2)[0]
            d.append(dwt_coef)
        elif method == 'spec_img':
            im = librosa.amplitude_to_db(np.abs(librosa.stft(slice, n_fft=128, center = False, hop_length=14))/np.sqrt(window_size),ref=np.max)
            d.append(im)
        elif method == 'raw':
            d.append(slice)
        else:
            raise RuntimeError ("Param 'method' should be one of 'fft', 'dwt', 'spec_img' or 'raw'.")
        if ana is not None:
            tmp_label = dense_labels[i*hop_length: i*hop_length+window_size]
            cl, c = np.unique(tmp_label, return_counts=True)
            l.append(cl[np.argmax(c)])
    d = np.array(d)
    
    if ana is not None:
        l = np.array(l) 
        return d, l
    else: 
        return d
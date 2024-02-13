from copy import deepcopy as dc
from scipy.stats import skew
import librosa
from stats import hurst_dfa
import preprocessing
import numpy as np
import pandas as pd
import pywt
import time
import entropy 
var = ['mean', 'sd', 'sk', 'lowpassenergy', 'lowpassentropy', 'spec_entropy', 'centroid', 'flatness']
window_size = 1024
fft_freq = librosa.fft_frequencies(sr=100, n_fft = 1024)

def spectral_centroid(spectrum):
    return np.sum(spectrum*fft_freq) / np.sum(spectrum)

def spectral_flatness(spectrum):
    if 0. in spectrum:
        return 0
    else: 
        n = len(spectrum)
        return np.exp(np.mean(np.log(spectrum))/n)/(np.mean(spectrum)/n)
    
def shannon_entropy(array):
    max = array.max()
    min = array.min()
    if max == min:
        return np.log(2)
    else:
        normalized_array = (array - min)/(max - min)
        entropy = 0
        for i in range(len(array)):
            if normalized_array[i] != 0:
                entropy -= normalized_array[i]*np.log(normalized_array[i])
        return entropy

def get_bins_features(array, n_bins = 4):
    bin_length = len(array)//n_bins
    mean = []
    std = []
    slope = []
    for i in range(n_bins):
        slice = array[i*bin_length: (i+1)*bin_length]
        mean.append(np.mean(slice))
        std.append(np.std(slice))
        slope.append(np.polyfit(np.arange(bin_length),slice,deg = 1)[0])
    return mean + std + slope

def get_bins_slopes(array, n_bins = 8):
    bin_length = len(array)//n_bins
    # mean = []
    # std = []
    slope = []
    for i in range(n_bins):
        slice = array[i*bin_length: (i+1)*bin_length]
        # mean.append(np.mean(slice))
        # std.append(np.std(slice))
        slope.append(np.polyfit(np.arange(bin_length),slice,deg = 1)[0])
    return slope

def get_feature_matrix(df):
    d = []
    ten_percent = len(df)//10
    c = 0
    for i in range(len(df)):
        if i%ten_percent == 0:
            print(f' {c*10}% done. ') 
            c +=1

        # fft  
        spectrum = np.abs(librosa.stft(df[i,:], n_fft = window_size,center = False)).ravel()/window_size
        spec_centroid = spectral_centroid(spectrum)

        power_spectrum = spectrum**2
        spec_entropy = shannon_entropy(power_spectrum)
        spec_flatness = spectral_flatness(power_spectrum)
        permutation_entropy = entropy.permutation_entropy(df[i,:])

        # Wavelet
        coef = pywt.wavedec(df[i,:],'db4',3)
        lowpass_coef = coef[1]
        highpass_coef = coef[0]
        
        # low_freq
        lowpass_energy = np.mean(lowpass_coef**2)
        lowpass_entropy = shannon_entropy(lowpass_coef)
        # bin_features = get_bins_features(lowpass_coef, 4)
        bin_slopes = get_bins_slopes(lowpass_coef, 4)

        # high_freq
        sk = skew(highpass_coef)
        if np.isnan(sk):
            sk = 0
        mean = np.mean(highpass_coef)
        sd = np.std(highpass_coef)
        zcr = librosa.feature.zero_crossing_rate(y = highpass_coef,frame_length = len(highpass_coef),center = False).item()

        # concat
        f = [lowpass_energy, lowpass_entropy, permutation_entropy, spec_entropy, spec_centroid, spec_flatness, zcr,
             mean, sd, sk]
        f += bin_slopes
        d.append(f)
        
    print('Complete.')
    d = np.stack([f for f in d])
    return d

from sklearn.preprocessing import MinMaxScaler

def preprocess(feature_matrix, label):
    fm = pd.DataFrame(feature_matrix)
    fm.fillna(0, inplace = True)
    fm = fm.to_numpy()

    scaler = MinMaxScaler()
    fm = scaler.fit_transform(fm)
    label = pd.Series(label).map({1: 0, 2: 1, 4: 2, 5: 6, 6: 3, 7: 4, 8: 5}).to_numpy()

    return fm, label
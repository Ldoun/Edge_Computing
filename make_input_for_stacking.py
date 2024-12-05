import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from functools import partial
from sklearn.preprocessing import MinMaxScaler

from data import load_audio_mfcc
from config import args_for_audio, args_for_data

train_files = glob('result/HuggingFace*/for_stacking_input.csv') #validation set prediction
test_files = glob('result/HuggingFace*/sum.csv') #summation of test set prediction proba
model_list = ['wav2vec2-base', 'ast', 'wav2vec2-conformer-rope', 'hubert-xlarge', 'unispeech-sat-large', 'wavlm-large', 'wav2vec2-conformer-rel', 'unispeech-large-1500h', 'data2vec-audio-large', 'sew-mid']

train_df_list = []
test_df_list = []
for train_file, test_file, model in zip(train_files, test_files, model_list):
    train_df = pd.read_csv(train_file)
    train_df = train_df.rename(columns={str(i):model+str(i) for i in range(6)})
    train_df_list.append(train_df)

    test_df = pd.read_csv(test_file)
    test_df = test_df[[str(i) for i in range(6)]] #drop id and path to file
    test_df = test_df.rename(columns={str(i):model+str(i) for i in range(6)})
    test_df = test_df.applymap(lambda x: x/10) #because of 10 fold cross validation
    test_df_list.append(test_df)

train_stacking = pd.concat(train_df_list, axis=1)
train_stacking.to_csv('result/train_stacking_input.csv', index=False)
test_stacking = pd.concat(test_df_list, axis=1)
test_stacking.to_csv('result/test_stacking_input.csv', index=False)

parser = argparse.ArgumentParser()
args_for_audio(parser) #reuse the argument made by config.py
args_for_data(parser) #reuse the argument made by config.py
args = parser.parse_args()

mfcc_func = partial(load_audio_mfcc,
            sr=16000, n_fft=args.n_fft, win_length=args.win_length, hop_length=args.hop_length, n_mels=args.n_mels, n_mfcc=args.n_mfcc)
process_func = lambda x: np.mean(mfcc_func(x).T, axis=1)

train_data = pd.read_csv(args.train)
train_data['path'] = train_data['path'].apply(lambda x: os.path.join(args.path, x))
test_data = pd.read_csv(args.test)
test_data['path'] = test_data['path'].apply(lambda x: os.path.join(args.path, x))

train_features = [process_func(file) for file in train_data['path']]
test_features = [process_func(file) for file in test_data['path']]

min_max_scaler = MinMaxScaler().fit(train_features) #use min-max scaler to scale MFCC features to a range of 0~1
train_features = min_max_scaler.transform(train_features)
test_features = min_max_scaler.transform(test_features)

train_mfcc_df = pd.DataFrame(train_features, columns=['mfcc_'+str(x) for x in range(1,args.n_mfcc+1)])
test_mfcc_df = pd.DataFrame(test_features, columns=['mfcc_'+str(x) for x in range(1,args.n_mfcc+1)])

pd.concat([train_stacking, train_mfcc_df], axis=1).to_csv('result/train_stacking_with_mfcc.csv', index=False)
pd.concat([test_stacking, test_mfcc_df], axis=1).to_csv('result/test_stacking_with_mfcc.csv', index=False)
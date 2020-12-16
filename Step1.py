# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:05:07 2020

@author: quang
"""
from pyvi import ViTokenizer 
from tqdm import tqdm
import gensim 
import os 
import pickle
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'Data')
#đọc dữ liệu từ file để xử lí 
def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                #Loai bo ky tu dac biet, lowcase
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                #Tách từ Tiếng Việt
                lines = ViTokenizer.tokenize(lines)
                X.append(lines)
                y.append(path)
    return X, y
#import dữ liệu huấn luyện
train_path = os.path.join(dir_path, r'C:\Users\quang\OneDrive\Desktop\NLP 02\new train')
X_data, y_data = get_data(train_path)
#import dữ liệu test
test_path = os.path.join(dir_path, r'C:\Users\quang\OneDrive\Desktop\NLP 02\new test')
X_data_test, y_data_test = get_data(test_path)
#Xuất data sau khi xử lí
pickle.dump(X_data, open(r'C:\Users\quang\OneDrive\Desktop\NLP 02\X_data.pkl', 'wb'))
pickle.dump(y_data, open(r'C:\Users\quang\OneDrive\Desktop\NLP 02\y_data.pkl', 'wb'))
pickle.dump(X_data_test, open(r'C:\Users\quang\OneDrive\Desktop\NLP 02\X_data_test.pkl', 'wb'))
pickle.dump(y_data_test, open(r'C:\Users\quang\OneDrive\Desktop\NLP 02\y_data_test.pkl', 'wb'))


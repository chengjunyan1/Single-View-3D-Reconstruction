import os,random
import numpy as np

def get_train_ids(data_path):
    ids_train_path=os.path.join(data_path,'ids_train.txt')
    if not os.path.exists(ids_train_path):
        print('ids file not found, generating ids...')
        generate_ids(data_path)
    ids_train=load_ids(ids_train_path)
    return ids_train

def get_test_ids(data_path):
    ids_test_path=os.path.join(data_path,'ids_test.txt')
    if os.path.exists(ids_test_path):
        ids_test=load_ids(ids_test_path)
        return ids_test
    else:
        print('ids file not found, generating ids based on ids_train...')
        ids_train_path=os.path.join(data_path,'ids_train.txt')
        if os.path.exists(ids_train_path):
            ids_train=load_ids(ids_train_path)
            ids_test=[]
            gt_path=os.path.join(data_path,'gt')
            all_ids=[i.split('.')[0] for i in os.listdir(gt_path)]
            for i in all_ids:
                if i not in ids_train:
                    ids_test.append(i)
            return ids_test
        else:
            print('ids_train not fount')
            return

def generate_ids(data_path,ratio=0.2):
    gt_path=os.path.join(data_path,'gt')
    all_ids=[i.split('.')[0] for i in os.listdir(gt_path)]
    categories=get_categories(data_path)
    counter={}
    category={}
    for i in categories:
        counter[i]=0
        category[i]=[]
    for i in all_ids:
        c=i.split('+')[0]
        counter[c]+=1
        category[c].append(i)
    ids_test=[]
    for i in categories:
        test_num=int(counter[i]*ratio)
        testi=random.sample(category[i],test_num)
        ids_test+=testi
    ids_train=[]
    for i in all_ids:
        if i not in ids_test:
            ids_train.append(i)
    ids_train_path=os.path.join(data_path,'ids_train.txt')
    ids_test_path=os.path.join(data_path,'ids_test.txt')
    save_ids(ids_train,ids_train_path)
    save_ids(ids_test,ids_test_path)

def save_ids(ids,path):
    file = open(path,'w')
    for i in ids:
        file.write(i); 
        file.write('\n')
    file.close()

def load_ids(path):
    file = open(path,'r')
    ids=[]
    temp=file.readlines()
    for i in temp:
        ids.append(i.strip('\n'))
    file.close()
    return ids

def get_categories(data_path):
    gt_path=os.path.join(data_path,'gt')
    all_ids=[i.split('.')[0] for i in os.listdir(gt_path)]
    cat=[]
    for i in all_ids:
        c=i.split('+')[0]
        if c not in cat:
            cat.append(c)
    return cat



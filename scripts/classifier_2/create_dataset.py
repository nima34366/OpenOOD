import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from tqdm import tqdm

need_plot = False

dataset_ID = 'cifar10'
num_methods = 0
conf_size = 0
for i in tqdm(os.listdir('/home/nima/OpenOOD/results'), desc = 'Initial'):
    if  (('test_ood' in i) and (dataset_ID in i)):
        if num_methods == 0:
            for j in os.listdir('/home/nima/OpenOOD/results/'+i+'/scores'):
                conf_size+=np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
                print(j,np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size)
            num_id_samples = np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+dataset_ID+'.npz')['conf'].size
            num_ood_datasets = len(os.listdir('/home/nima/OpenOOD/results/'+i+'/scores'))-1
            num_ood_samples_needed_from_1_OOD_dataset = int(num_id_samples/num_ood_datasets)
        num_methods+=1
print('Number of OOD detecting methods:',num_methods)
print('Total number of samples =',conf_size)

datatype = np.dtype([('f_value',float,num_methods),('f_method','U13',num_methods),('dataset_name','U13'),('ood',int),('filename','U300'),('prob',float)])
dataset = np.empty(conf_size,dtype = datatype)  

fromm = 0
too = 0
method_num = 0
for i in tqdm(os.listdir('/home/nima/OpenOOD/results'), desc='dataset'):
    if (('test_ood' in i) and (dataset_ID in i)):
        method = i.split('_')[6]
        for j in os.listdir('/home/nima/OpenOOD/results/'+i+'/scores'):
            d = j.split('.')[0]
            if d == 'place365':
                d = 'places365'
            too += np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
            dataset[fromm:too]['f_value'][:,method_num] = np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'] 
            dataset[fromm:too]['filename'] = genfromtxt('/home/nima/OpenOOD/data/benchmark_imglist/'+dataset_ID+'/test_'+d+'.txt',dtype=object)[:,0] 
            dataset[fromm:too]['f_method'][:,method_num] = method
            dataset[fromm:too]['dataset_name'] = d
            if d == dataset_ID:
                dataset[fromm:too]['ood'] = 0
                dataset[fromm:too]['prob'] = 0.5/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
            else:
                dataset[fromm:too]['ood'] = 1
                dataset[fromm:too]['prob'] = (0.5/num_ood_datasets)/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
            fromm += np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
        method_num+=1
        if (fromm!=conf_size):
            print('PROBLEM!!!')
            exit()
        fromm = 0
        too = 0

# print(np.reshape(np.unique(dataset, axis=4),(conf_size,num_methods,)))
method_num = 0

if need_plot ==  True:
    for i in tqdm(os.listdir('/home/nima/OpenOOD/results'), desc='plots'):
        if (('test_ood' in i) and (dataset_ID in i)):
            method = i.split('_')[6]
            for j in os.listdir('/home/nima/OpenOOD/results/'+i+'/scores'):
                d = j.split('.')[0]
                if d == 'place365':
                    d = 'places365'
                if ((d == 'cifar100') or (d == 'tin')):
                    color = 'r'
                elif ((d == 'mnist') or (d == 'svhn') or (d == 'texture') or (d == 'places365')):
                    color = 'g'
                else:
                    color = 'b'
                req_data = dataset[dataset['dataset_name']==d]['f_value'][:,method_num]
                plt.hist(req_data, label=d, alpha=0.5, bins = 100, stacked=True, density=True, histtype='bar', color=color)
            plt.legend()
            plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/'+method)
            plt.clf() 
            method_num+=1
# dataset['prob'] = dataset['prob']/np.sum(dataset['prob'])
x = np.empty((num_id_samples*2,num_methods))
y = np.empty(num_id_samples*2)
selected_sample_indices = np.zeros(conf_size, dtype=bool)
for i in tqdm(range(num_id_samples*2), desc='sampling'):
    p = np.random.choice(conf_size, 1, replace=True, p=dataset['prob'])
    selected_sample_indices[p] = 1
    prob = dataset['prob'][p]
    dataset['prob'][p] = 0
    sim = np.argwhere((dataset['dataset_name']==dataset['dataset_name'][p]) & (dataset['prob'] == prob))
    if sim.size == 0:
        dataset['prob'] = dataset['prob']/np.sum(dataset['prob'])
        continue
    newprob = (sim.size*prob+prob)/sim.size
    dataset['prob'][sim]=newprob
    
x = dataset['f_value'][selected_sample_indices]
y = dataset['ood'][selected_sample_indices]
key = np.unique(dataset[selected_sample_indices]['dataset_name'], return_counts=True)[0]
val = np.unique(dataset[selected_sample_indices]['dataset_name'], return_counts=True)[1]
for i in range(len(key)):
    print(key[i], val[i])

np.save('/home/nima/OpenOOD/scripts/data/classifier_2/dataset.npy', dataset)
np.save('/home/nima/OpenOOD/scripts/data/classifier_2/x.npy', x)
np.save('/home/nima/OpenOOD/scripts/data/classifier_2/y.npy', y)
np.save('/home/nima/OpenOOD/scripts/data/classifier_2/selected_sample_indices.npy', selected_sample_indices)



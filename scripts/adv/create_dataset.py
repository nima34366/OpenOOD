import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from tqdm import tqdm
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

need_plot = True

f = open('./scripts/classifier_2/log.txt','w')

def print_write(*text):
    print(*text)
    print(*text, file=f)

dataset_ID = 'cifar10'
dataset_nearood = ['cifar100','tin']
dataset_adv = 'pgd'
datasets_adv = ['fgsm']
num_near_ood_datasets = len(dataset_nearood)
num_methods = 0
conf_size = 0

for i in tqdm(os.listdir('/home/nima/OpenOOD/results'), desc = 'Initial'):
    if  (('test_ood' in i) and (dataset_ID in i)):
        if num_methods == 0:
            for j in os.listdir('/home/nima/OpenOOD/results/'+i+'/scores'):
                if j.split('.')[0] in datasets_adv:
                    continue
                conf_size+=np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
                print_write(j,np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size)
            num_id_samples = np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+dataset_ID+'.npz')['conf'].size
            num_ood_datasets = len(os.listdir('/home/nima/OpenOOD/results/'+i+'/scores'))-len(datasets_adv)-2
            # num_ood_samples_needed_from_1_OOD_dataset = int(num_id_samples/num_ood_datasets)
        num_methods+=1
print_write('Number of OOD detecting methods:',num_methods)
print_write('Total number of samples =',conf_size)

num_far_ood_datasets = np.abs(num_ood_datasets-len(dataset_nearood))
datatype = np.dtype([('f_value',float,num_methods),('f_method','U13',num_methods),('dataset_name','U13'),('ood',int),('nearood',int),('filename','U300'),('prob',float),('nearprob',float),('advprob',float)])
dataset = np.empty(conf_size,dtype = datatype)  

fromm = 0
too = 0
method_num = 0

for i in tqdm(os.listdir('/home/nima/OpenOOD/results'), desc='dataset'):
    if (('test_ood' in i) and (dataset_ID in i)):
        method = i.split('_')[6]
        for j in os.listdir('/home/nima/OpenOOD/results/'+i+'/scores'):
            d = j.split('.')[0]
            if d in datasets_adv:
                continue
            too += np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
            dataset[fromm:too]['f_value'][:,method_num] = np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'] 
            if d == dataset_adv:
                dataset[fromm:too]['filename'] = genfromtxt('/home/nima/OpenOOD/data/benchmark_imglist/cifar10/adv.txt',dtype=object)[:,0] 
            else:
                dataset[fromm:too]['filename'] = genfromtxt('/home/nima/OpenOOD/data/benchmark_imglist/'+dataset_ID+'/test_'+d+'.txt',dtype=object)[:,0] 
                
            dataset[fromm:too]['f_method'][:,method_num] = method
            dataset[fromm:too]['dataset_name'] = d
            if d == dataset_ID:
                dataset[fromm:too]['ood'] = 0
                dataset[fromm:too]['nearood'] = 0
                dataset[fromm:too]['prob'] = 0.5/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
                dataset[fromm:too]['nearprob'] = 1/3/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
                dataset[fromm:too]['advprob'] = 1/4/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
            elif d in dataset_nearood:
                dataset[fromm:too]['ood'] = 1
                dataset[fromm:too]['nearood'] = 1
                dataset[fromm:too]['prob'] = (0.5/num_ood_datasets)/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
                dataset[fromm:too]['nearprob'] = (1/3/num_near_ood_datasets)/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
                dataset[fromm:too]['advprob'] = (1/4/num_near_ood_datasets)/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
            elif d == dataset_adv:
                dataset[fromm:too]['ood'] = -1
                dataset[fromm:too]['nearood'] = 3
                dataset[fromm:too]['prob'] = 0
                dataset[fromm:too]['nearprob'] = 0
                dataset[fromm:too]['advprob'] = 1/4/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
            else:
                dataset[fromm:too]['ood'] = 1
                dataset[fromm:too]['nearood'] = 2
                dataset[fromm:too]['prob'] = (0.5/num_ood_datasets)/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
                dataset[fromm:too]['nearprob'] = (1/3/num_far_ood_datasets)/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
                dataset[fromm:too]['advprob'] = (1/4/num_far_ood_datasets)/np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
            fromm += np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
        method_num+=1
        if (fromm!=conf_size):
            print_write('PROBLEM!!!')
            exit()
        fromm = 0
        too = 0
if not os.path.exists('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv):
    os.mkdir('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv)
np.save('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/dataset.npy', dataset)

# print_write(np.reshape(np.unique(dataset, axis=4),(conf_size,num_methods,)))
method_num = 0

if need_plot ==  True:
    for i in tqdm(os.listdir('/home/nima/OpenOOD/results'), desc='plots'):
        if (('test_ood' in i) and (dataset_ID in i)):
            method = i.split('_')[6]
            for j in os.listdir('/home/nima/OpenOOD/results/'+i+'/scores'):
                d = j.split('.')[0]
                if d in datasets_adv:
                    continue
                if ((d == 'cifar100') or (d == 'tin')):
                    color = 'r'
                elif ((d == 'mnist') or (d == 'svhn') or (d == 'texture') or (d == 'places365')):
                    color = 'g'
                elif d == dataset_adv:
                    color = 'b'
                else:
                    color = 'm'
                req_data = dataset[dataset['dataset_name']==d]['f_value'][:,method_num]
                plt.hist(req_data, label=d, alpha=0.5, bins = 100, stacked=True, density=True, histtype='bar', color=color)
            plt.legend()
            if not os.path.exists('/home/nima/OpenOOD/scripts/classifier_2/images/'+dataset_adv):
                os.makedirs('/home/nima/OpenOOD/scripts/classifier_2/images/'+dataset_adv)
            plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/'+dataset_adv+'/'+method)
            plt.clf() 
            method_num+=1

# selected_sample_indices = np.zeros(conf_size, dtype=bool)
# for i in tqdm(range(num_id_samples*2), desc='sampling'):
#     p = np.random.choice(conf_size, 1, replace=True, p=dataset['prob'])
#     selected_sample_indices[p] = 1
#     prob = dataset['prob'][p]
#     dataset['prob'][p] = 0
#     sim = np.argwhere((dataset['dataset_name']==dataset['dataset_name'][p]) & (dataset['prob'] == prob))
#     if sim.size == 0:
#         dataset['prob'] = dataset['prob']/np.sum(dataset['prob'])
#         continue
#     newprob = (sim.size*prob+prob)/sim.size
#     dataset['prob'][sim]=newprob
    
# x = dataset['f_value'][selected_sample_indices]
# y = dataset['ood'][selected_sample_indices]
# key = np.unique(dataset[selected_sample_indices]['dataset_name'], return_counts=True)[0]
# val = np.unique(dataset[selected_sample_indices]['dataset_name'], return_counts=True)[1]
# for i in range(len(key)):
#     print_write(key[i], val[i])
# np.save('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/x.npy', x)
# np.save('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/y.npy', y)
# np.save('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/selected_sample_indices.npy', selected_sample_indices)

# print_write('NEAROOD SAMPLING')

# selected_sample_indices_near = np.zeros(conf_size, dtype=bool)
# for i in tqdm(range(num_id_samples*3), desc='near sampling'):
#     p = np.random.choice(conf_size, 1, replace=True, p=dataset['nearprob'])
#     selected_sample_indices_near[p] = 1
#     prob = dataset['nearprob'][p]
#     dataset['nearprob'][p] = 0
#     sim = np.argwhere((dataset['dataset_name']==dataset['dataset_name'][p]) & (dataset['nearprob'] == prob))
#     if sim.size == 0:
#         dataset['nearprob'] = dataset['nearprob']/np.sum(dataset['nearprob'])
#         continue
#     newprob = (sim.size*prob+prob)/sim.size
#     dataset['nearprob'][sim]=newprob
    
# x_near = dataset['f_value'][selected_sample_indices_near]
# y_near = dataset['nearood'][selected_sample_indices_near]
# key = np.unique(dataset[selected_sample_indices_near]['dataset_name'], return_counts=True)[0]
# val = np.unique(dataset[selected_sample_indices_near]['dataset_name'], return_counts=True)[1]
# for i in range(len(key)):
#     print_write(key[i], val[i])
# np.save('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/x_near.npy', x_near)
# np.save('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/y_near.npy', y_near)
# np.save('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/selected_sample_indices_near.npy', selected_sample_indices_near)


print_write('ADVERSERIAL SAMPLING')

selected_sample_indices_adv = np.zeros(conf_size, dtype=bool)
for i in tqdm(range(num_id_samples*4), desc='adv sampling'):
    p = np.random.choice(conf_size, 1, replace=True, p=dataset['advprob'])
    selected_sample_indices_adv[p] = 1
    prob = dataset['advprob'][p]
    dataset['advprob'][p] = 0
    sim = np.argwhere((dataset['dataset_name']==dataset['dataset_name'][p]) & (dataset['advprob'] == prob))
    if sim.size == 0:
        dataset['advprob'] = dataset['advprob']/np.sum(dataset['advprob'])
        continue
    newprob = (sim.size*prob+prob)/sim.size
    dataset['advprob'][sim]=newprob
    
x_adv = dataset['f_value'][selected_sample_indices_adv]
y_adv = dataset['nearood'][selected_sample_indices_adv]
key = np.unique(dataset[selected_sample_indices_adv]['dataset_name'], return_counts=True)[0]
val = np.unique(dataset[selected_sample_indices_adv]['dataset_name'], return_counts=True)[1]
for i in range(len(key)):
    print_write(key[i], val[i])
np.save('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/x_adv.npy', x_adv)
np.save('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/y_adv.npy', y_adv)
np.save('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/selected_sample_indices_adv.npy', selected_sample_indices_adv)

# print_write('STATS')

# dataset = np.load('./scripts/classifier_2/data/dataset.npy')

# for i,method in enumerate(dataset[0]['f_method']):
#     req_data = dataset['f_value'][:,i]
#     print_write(method, 'mean:', round(np.mean(req_data),2), 'std:',round(np.std(req_data),2), 'min:',round(np.min(req_data),2), 'max:',round(np.max(req_data),2))
#     # print_write(method, 'mean:', np.mean(req_data), 'std:',np.std(req_data), 'min:',np.min(req_data), 'max:',np.max(req_data))

f.close()




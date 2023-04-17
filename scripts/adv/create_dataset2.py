import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from tqdm import tqdm
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

id_datasets=['cifar10','mnist']
nood_datasets={'cifar10':['cifar100','tin'],
               'mnist':['notmnist', 'fashionmnist']}
food_datasets={'cifar10': ['mnist', 'svhn', 'texture', 'places365'],
               'mnist':['texture', 'cifar10', 'tin', 'places365']}
adv_datasets=['fgsm', 'pgd']
methods=['dice','ebo','godin','gradnorm','gram','klm','knn','mds','mls','msp','odin','react','vim']

f = open('./scripts/classifier_2/log.txt','w')

def print_write(*text):
    print(*text)
    print(*text, file=f)

def create_dataset(id_dataset):
    conf_size=0
    all_datasets = id_dataset+nood_datasets[id_dataset] + food_datasets[id_dataset]+adv_datasets
    for i in os.listdir('/home/nima/OpenOOD/results'):
        if  (('test_ood' in i) and (id_dataset in i)):
            for j in os.listdir('/home/nima/OpenOOD/results/'+i+'/scores'):
                conf_size+=np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
                print_write(j,np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size)
            num_id_samples = np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+id_dataset+'.npz')['conf'].size
            break
    
    print_write('ID dataset:',id_dataset)
    print_write('Number of OOD detecting methods:',len(methods))
    print_write('Total number of samples =',conf_size)
    print_write('All datasets:',all_datasets)

    scores_dataset=np.empty(len(methods),conf_size,2)

    fromm = 0
    too = 0
    for i in os.listdir('/home/nima/OpenOOD/results'):
        for method_num, method in enumerate(methods):
            if  (('test_ood' in i) and (id_dataset in i) and (method in i)):
                for j in os.listdir('/home/nima/OpenOOD/results/'+i+'/scores'):
                    for dataset_num, dataset in enumerate(all_datasets):
                        if dataset in j:
                            too += np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf'].size
                            scores_dataset[method_num,fromm:too,0]=np.load('/home/nima/OpenOOD/results/'+i+'/scores/'+j)['conf']
                            scores_dataset[method_num,fromm:too,1]=dataset_num
                            fromm = too
                            break
            if (fromm!=conf_size):
                print_write('FROMM not equal CONFSIZE!!!')
                exit()
            fromm = 0
            too = 0
    np.save('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/dataset.npy', dataset)

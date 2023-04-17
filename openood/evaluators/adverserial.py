import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.preprocessors.transform import normalization_dict
from openood.postprocessors import BasePostprocessor
from openood.utils import Config
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from torchvision.utils import save_image
from autoattack import AutoAttack

def to_np(x):
    return x.data.cpu().numpy()

class Adv:
    def __init__(self, config: Config):
        self.config = config
        normalization_type = config.dataset.normalization_type
        self.attacks = ['fgsm', 'pgd']
        self.supported_attacks = ['fgsm', 'pgd', 'aa', 'cw']
        for attack in self.attacks:
            if attack not in self.supported_attacks:
                raise ValueError('Attack {} not supported'.format(attack))
        if normalization_type in normalization_dict.keys():
            self.mean = torch.Tensor(normalization_dict[normalization_type][0]).cuda()
            self.std = torch.Tensor(normalization_dict[normalization_type][1]).cuda()
        else:
            self.mean = torch.Tensor([0.5, 0.5, 0.5]).cuda()
            self.std = torch.Tensor([0.5, 0.5, 0.5]).cuda()
        
        self.epsilon = [{'cifar10': 0.01, 'mnist': 0.1, 'cifar100': 0.01},
                        {'cifar10': 0.03, 'mnist': 0.2, 'cifar100': 0.03},
                        {'cifar10': 0.09, 'mnist': 0.3, 'cifar100': 0.09},
                        {'cifar10': 0.27, 'mnist': 0.4, 'cifar100': 0.27}]

        if self.config.dataset.name=='imagenet':
            self.dataset_name='imagenet_1k'
        else:
            self.dataset_name=self.config.dataset.name

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()

        for epsilon in tqdm(self.epsilon, leave=False):
            loss_avg = {'orig': 0.0}
            correct = {'orig': 0}
            for attack in self.attacks:
                loss_avg[attack] = 0.0
                correct[attack] = 0

            count = 0
            for attack in self.attacks:
                if attack not in os.listdir("/home/nima/OpenOOD/data/images_classic/"+self.dataset_name):
                    os.mkdir("/home/nima/OpenOOD/data/images_classic/"+self.dataset_name+'/'+attack+'_'+str(epsilon[self.dataset_name]))
            if 'aa' in self.attacks:
                adversary = AutoAttack(net, norm='Linf', eps=epsilon[self.dataset_name], version='standard')
                adversary.attacks_to_run = ['fab-t']
            for batch in tqdm(data_loader,
                                desc='Eval: ',
                                position=0,
                                leave=False):
                data = {'orig': batch['data'].cuda()}
                target = batch['label'].cuda()
                if 'fgsm' in self.attacks:
                    data['fgsm'] = fast_gradient_method(net, data['orig'], epsilon[self.dataset_name], np.inf)
                if 'pgd' in self.attacks:
                    data['pgd'] = projected_gradient_descent(net, data['orig'], epsilon[self.dataset_name], 0.01, 40, np.inf)
                if 'aa' in self.attacks:
                    data['aa'] = adversary.run_standard_evaluation(data['orig'], target, bs=len(data)//8)
                if 'cw' in self.attacks:
                    data['cw'] = carlini_wagner_l2(net, data['orig'], self.config.dataset.num_classes)
                for i in range(len(data['orig'])):
                    for attack in self.attacks:
                        save_image(torch.add(torch.mul(data[attack][i].transpose(0,-1),self.std),self.mean).transpose(0,-1), "/home/nima/OpenOOD/data/images_classic/"+self.dataset_name+"/"+attack+'_'+str(epsilon[self.dataset_name])+"/"+str(count)+".png")
                    count+=1

                output = {'orig': net(data['orig'])}
                loss = {'orig': F.cross_entropy(output['orig'], target)}
                pred = {'orig': output['orig'].data.max(1)[1]}
                correct['orig'] += pred['orig'].eq(target.data).sum().item()
                loss_avg['orig'] += float(loss['orig'].data)
                for attack in self.attacks:
                    output[attack] = net(data[attack])
                    loss[attack] = F.cross_entropy(output[attack], target)
                    pred[attack] = output[attack].data.max(1)[1]
                    correct[attack] += pred[attack].eq(target.data).sum().item()
                    loss_avg[attack] += float(loss[attack].data)

            print(self.dataset_name,'epsilon',epsilon[self.dataset_name])
            loss = {'orig': loss_avg['orig'] / len(data_loader)}
            acc = {'orig': correct['orig'] / len(data_loader.dataset)}
            for attack in self.attacks:
                loss[attack] = loss_avg[attack] / len(data_loader)
                acc[attack] = correct[attack] / len(data_loader.dataset)
            for keys in acc.keys():
                print(keys, 'acc', acc[keys])
                print(keys, 'loss', loss[keys])
            print()
from xgboost import XGBClassifier
# read data
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

dataset_adv = 'pgd'

f = open('./scripts/classifier_2/data/'+dataset_adv+'/log_shap.txt','w')
def print_write(*text):
    print(*text)
    print(*text, file=f)

dataset = np.load('./scripts/classifier_2/data/'+dataset_adv+'/dataset.npy')

# dirx='/home/nima/OpenOOD/scripts/classifier_2/data/x.npy'
# diry='/home/nima/OpenOOD/scripts/classifier_2/data/y.npy'
# x = np.load(dirx)
# y = np.load(diry)

# #all 13
# _x = x
# X_train, X_test, y_train, y_test = train_test_split(_x, y, test_size=.2)
# bst = XGBClassifier(n_estimators=2, max_depth=3, learning_rate=1, objective='binary:logistic')
# bst.fit(X_train, y_train)
# preds = bst.predict(X_test)
# print_write('all together acc:',(preds == y_test).sum()/len(preds))
# bst.save_model('/home/nima/OpenOOD/scripts/classifier_2/2class.json')
# explainer = shap.Explainer(bst)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=dataset[0]['f_method'])
# plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/shap_bar.png')
# plt.clf()
# shap_values = explainer(X_test)
# shap_values.feature_names = dataset[0]['f_method']
# shap.plots.beeswarm(shap_values)
# plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/shap.png')
# plt.clf()

# data_id_ood = {}
# indexes = []
# for i,method in enumerate(dataset[0]['f_method']):
#     val = []
#     for j,method2 in enumerate(dataset[0]['f_method']):
#         X_train_, X_test_ = X_train[:,[i,j]], X_test[:,[i,j]]
#         bst = XGBClassifier(n_estimators=2, max_depth=3, learning_rate=1, objective='binary:logistic')
#         bst.fit(X_train_, y_train)
#         preds = bst.predict(X_test_)
#         val.append((preds == y_test).sum()/len(preds))
#         # print_write(i,j,method, method2,'acc:',(preds == y_test).sum()/len(preds))
#     data_id_ood[method] = val
#     indexes.append(method)
# df = pd.DataFrame(data_id_ood, index = indexes).to_csv('./scripts/classifier_2/data_id_ood.csv')


# print_write('################# NEAR FAR OOD ##################')

# dirx='/home/nima/OpenOOD/scripts/classifier_2/data/x_near.npy'
# diry='/home/nima/OpenOOD/scripts/classifier_2/data/y_near.npy'
# x = np.load(dirx)
# y = np.load(diry)

# #all 13
# _x = x
# X_train, X_test, y_train, y_test = train_test_split(_x, y, test_size=.2)

# bst = XGBClassifier(n_estimators=2, max_depth=3, learning_rate=1, objective='multi:softmax', num_class = 3)
# bst.fit(X_train, y_train)
# preds = bst.predict(X_test)
# print_write('all together acc:',(preds == y_test).sum()/len(preds))
# bst.save_model('/home/nima/OpenOOD/scripts/classifier_2/3class.json')
# explainer = shap.Explainer(bst)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=['ID','Near OOD','Far OOD'], class_inds=[0,1,2], feature_names=dataset[0]['f_method'])
# plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/shap_near_bar.png')
# plt.clf() 
# shap_values = explainer(X_test)
# shap_values.feature_names = dataset[0]['f_method']
# shap.plots.beeswarm(shap_values[:,:,0])
# plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/shap_near_id.png')
# plt.clf()
# shap.plots.beeswarm(shap_values[:,:,1])
# plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/shap_near_nod.png')
# plt.clf()
# shap.plots.beeswarm(shap_values[:,:,2])
# plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/shap_near_fod.png')
# plt.clf()


# data_near_far_ood = {}
# indexes = []
# for i,method in enumerate(dataset[0]['f_method']):
#     val = []
#     for j,method2 in enumerate(dataset[0]['f_method']):
#         X_train_, X_test_ = X_train[:,[i,j]], X_test[:,[i,j]]
#         bst = XGBClassifier(n_estimators=2, max_depth=3, learning_rate=1, objective='multi:softmax', num_class = 3)
#         bst.fit(X_train_, y_train)
#         preds = bst.predict(X_test_)
#         accs = str(round((preds == y_test).sum()/len(preds),2))+'/'
#         for i in range(3):
#             accs += str(round((preds[y_test==i] == y_test[y_test==i]).sum()/len(y_test[y_test==i]),2))+'/'
#         val.append(accs)
#         # print_write(i,j,method, method2,'acc:',(preds == y_test).sum()/len(preds))
#     data_near_far_ood[method] = val
#     indexes.append(method)
# df = pd.DataFrame(data_near_far_ood, index = indexes).to_csv('./scripts/classifier_2/data_near_far_ood.csv')


print_write('################# ADV OOD ##################')

dirx='/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/x_adv.npy'
diry='/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/y_adv.npy'
x = np.load(dirx)
y = np.load(diry)

#all 13
_x = x
X_train, X_test, y_train, y_test = train_test_split(_x, y, test_size=.2)

bst = XGBClassifier(n_estimators=2, max_depth=3, learning_rate=1, objective='multi:softmax', num_class = 4)
bst.fit(X_train, y_train)
preds = bst.predict(X_test)
print_write('all together acc:',(preds == y_test).sum()/len(preds))
bst.save_model('/home/nima/OpenOOD/scripts/classifier_2/data/'+dataset_adv+'/4class.json')
explainer = shap.Explainer(bst)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=['ID','Near OOD','Far OOD', 'Adversarial'], class_inds=[0,1,2,3], feature_names=dataset[0]['f_method'])
plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/'+dataset_adv+'/shap_adv_bar.png')
plt.clf() 
shap_values = explainer(X_test)
shap_values.feature_names = dataset[0]['f_method']
shap.plots.beeswarm(shap_values[:,:,0])
plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/'+dataset_adv+'/shap_adv_id.png')
plt.clf()
shap.plots.beeswarm(shap_values[:,:,1])
plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/'+dataset_adv+'/shap_adv_nod.png')
plt.clf()
shap.plots.beeswarm(shap_values[:,:,2])
plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/'+dataset_adv+'/shap_adv_fod.png')
plt.clf()
shap.plots.beeswarm(shap_values[:,:,3])
plt.savefig('/home/nima/OpenOOD/scripts/classifier_2/images/'+dataset_adv+'/shap_adv_adv.png')
plt.clf()


data_adv_ood = {}
indexes = []
for i,method in enumerate(dataset[0]['f_method']):
    val = []
    for j,method2 in enumerate(dataset[0]['f_method']):
        X_train_, X_test_ = X_train[:,[i,j]], X_test[:,[i,j]]
        bst = XGBClassifier(n_estimators=2, max_depth=3, learning_rate=1, objective='multi:softmax', num_class = 4)
        bst.fit(X_train_, y_train)
        preds = bst.predict(X_test_)
        accs = str(round((preds == y_test).sum()/len(preds),2))+'/'
        for i in range(4):
            accs += str(round((preds[y_test==i] == y_test[y_test==i]).sum()/len(y_test[y_test==i]),2))+'/'
        val.append(accs)
        # print_write(i,j,method, method2,'acc:',(preds == y_test).sum()/len(preds))
    data_adv_ood[method] = val
    indexes.append(method)
df = pd.DataFrame(data_adv_ood, index = indexes).to_csv('./scripts/classifier_2/data/'+dataset_adv+'/data_adv_ood.csv')

f.close()
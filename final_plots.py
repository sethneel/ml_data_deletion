from clean_data import *
from sklearn import model_selection
from alg_desc_to_del import *
from model_log_reg_l2 import *
import matplotlib.pyplot as plt
import pickle
from random import sample
import pandas as pd

adult_results = pickle.load( open("pickles/adult_10_25_10.p", "rb") )
adult_unlearning_acc = adult_results[0]
adult_retrain_acc = adult_results[1]
lawschool_results = pickle.load( open("pickles/lawschool_5_25_10.p", "rb") )
lawschool_unlearning_acc = lawschool_results[0]
lawschool_retrain_acc = lawschool_results[1]
L=100

adult_unlearning = sum(adult_unlearning_acc)/float(L)
adult_retrain = sum(adult_retrain_acc)/float(L)
lawschool_unlearning = sum(lawschool_unlearning_acc)/float(L)
lawschool_retrain = sum(lawschool_retrain_acc)/float(L)

plt.plot(adult_unlearning, color='blue', label='unlearning')
plt.plot(adult_retrain, '--', color='blue', label='retraining')
plt.plot(lawschool_unlearning, color='red', label='unlearning')
plt.plot(lawschool_retrain, '--', color='red', label='retraining')
#plt.xticks(range(n_deletions+1))
plt.xlabel('update number')
plt.ylabel('accuracy')
#plt.legend(loc = 'best', prop={'size': 8})
#UPDATE
plt.title(r'blue: adult ($\epsilon = 10$, $\mathcal{I} = 25$), red: lawschool ($\epsilon = 5$, $\mathcal{I} = 25$)')
plt.savefig('figures/fig2.png', dpi = 300)
plt.show()

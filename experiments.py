from clean_data import *
from sklearn import model_selection
from alg_desc_to_del import *
from model_log_reg_l2 import *
import matplotlib.pyplot as plt
import pickle
from random import sample
import pandas as pd

#things to change/explore:
#dataset, epsilon, I

X, y = clean_adult(scale_and_center=True, normalize=True, intercept=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
n_deletions = 25
del_indices = np.random.randint(0, X_train.shape[0], size=n_deletions)
u_seq = [('-', ind,  X_train.iloc[ind], y_train.iloc[ind]) for ind in del_indices]

#X, y = clean_adult(scale_and_center=True, normalize=True, intercept=True)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2)
#X_train = X_train.reset_index(drop=True)
#y_train = y_train.reset_index(drop=True)
#n_additions = 10
#n_deletions = 40
#add_indices = np.random.randint(0, X_train.shape[0], size=n_additions)
#add_seq = [('+', X_train.iloc[ind], pd.Series(y_train.iloc[ind])) for ind in add_indices]
#X_train = X_train.drop(index = add_indices)
#y_train = y_train.drop(index = add_indices)
#del_indices = np.random.randint(0, X_train.shape[0], size=n_deletions)
#del_seq = [('-', ind,  X_train.iloc[ind], y_train.iloc[ind]) for ind in del_indices]
#u_seq = sample(add_seq+del_seq, len(add_seq+del_seq,))

#X_train = X_train.reset_index(drop=True)
#y_trian = y_train.reset_index(drop=True)
#desc_del_algorithm = DescDel(X_train, X_test, y_train, y_test, epsilon=1.0, delta=1.0/np.power(len(y_train), 2),
#                        update_grad_iter=50, model_class=LogisticReg, start_grad_iter=1000, update_sequence=u_seq,
#                        l2_penalty=.01)

unlearning_acc = []
retrain_acc = []
L = 1
for i in range(L):
    print(i)
    desc_del_algorithm = DescDel(X_train, X_test, y_train, y_test, epsilon=10.0, delta=1.0/np.power(len(y_train), 2),
                        update_grad_iter=25, model_class=LogisticReg, start_grad_iter=1000, update_sequence=u_seq,
                        l2_penalty=0.05)
    desc_del_algorithm.run()
    unlearning_acc.append( np.array(desc_del_algorithm.model_accuracies) )
    retrain_acc.append( np.array(desc_del_algorithm.scratch_model_accuracies) )

#print('unlearning algorithm accuracies:')
#print(desc_del_algorithm.model_accuracies)
#print('retraining accuracies:')
#print(desc_del_algorithm.scratch_model_accuracies)
#print(unlearning_acc)
#print(retrain_acc)

#UPDATE
saved_list = [unlearning_acc, retrain_acc]
pickle.dump(saved_list, open('pickles/adult_10_25_25.p', 'wb'))

unlearning = sum(unlearning_acc)/float(L)
retrain = sum(retrain_acc)/float(L)
plt.plot(unlearning, color='blue', label='unlearning')
plt.plot(retrain, color='red', label='retraining')
#plt.xticks(range(n_deletions+1))
plt.xlabel('update number')
plt.ylabel('accuracy')
plt.legend(loc='best', prop={'size': 10})
#UPDATE
plt.title(r'adult dataset, $\epsilon = 10$, $\mathcal{I} = 25$')
plt.savefig('figures/adult_10_25_25.png', dpi=300)
plt.show()

#thetas = [m.theta for m in desc_del_algorithm.models]

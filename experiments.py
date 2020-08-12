from clean_data import *
from sklearn import model_selection
from alg_desc_to_del import *
from model_log_reg_l2 import *
X, y = clean_adult(scale_and_center=True, normalize=True, intercept=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2)
n_deletions = 5
del_indices = np.random.randint(0, X_train.shape[0], size=n_deletions)
u_seq = [('-', ind,  X_train.iloc[ind], y_train.iloc[ind]) for ind in del_indices]
X_train.reset_index(drop=True)
y_train.reset_index(drop=True)
desc_del_algorithm = DescDel(X_train, X_test, y_train, y_test, epsilon=1.0, delta=1.0/np.power(len(y_train), 2),
                        update_grad_iter=50, model_class=LogisticReg, start_grad_iter=1000, update_sequence=u_seq,
                        l2_penalty=.01)

desc_del_algorithm.run()

print('unlearning algorithm accuracies:')
print(desc_del_algorithm.model_accuracies)
print('retraining accuracies:')
print(desc_del_algorithm.scratch_model_accuracies)
#thetas = [m.theta for m in desc_del_algorithm.models]

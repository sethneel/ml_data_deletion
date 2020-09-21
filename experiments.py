from alg_desc_to_del import *
from fed_desc_del import *
from model_log_reg_l2 import *
import matplotlib.pyplot as plt
import pickle


X, y = clean_adult(scale_and_center=True, normalize=True, intercept=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
n = X_train.shape[0]

# set hyper parameters
# set bootstrap size sample in [n, n^4/3] per theory
B = np.power(n, 7.0 / 6.0)
# scale of the data (x is normalized)
data_scale = 1.0
update_grad_iter = 25
epsilon = 10.0
delta = 1.0/np.power(len(y_train), 2)
l2 = 0.05

# compute starting gradient iterations for training desc-del (Thm. 3.1 in the paper)
# since the condition is T >= start_grad_iter_desc below, we set T = max(start_grad_iter_dist, start_grad_iter_dist)
# since the latter is generally larger and we want the two algorithms to start on equal footing
e = np.log(B) / np.log(n)
par = np.random.normal(0, 1, X_train.shape[1])
par = par / (np.sqrt(np.sum(np.power(par, 2))))
model = model_log_reg_l2.LogisticReg(par, l2_penalty=l2)
loss_fn_constants = model.get_constants()
gamma = (loss_fn_constants['smooth'] - loss_fn_constants['strong']) / (loss_fn_constants['strong'] +
                                                                            loss_fn_constants['smooth'])
t_part_1 = update_grad_iter * np.power(n, (3 * e - 4) / 2)
t_part_2 = np.log(data_scale * loss_fn_constants['strong'] / loss_fn_constants['lip'] *
                          np.power(n, e) * (1 + 10 * np.log(2.0 / delta))) / np.log(1 / gamma)
start_grad_iter_distributed = int(np.round(t_part_1 + t_part_2))
start_grad_iter_desc = int(np.round(update_grad_iter + np.log(data_scale*loss_fn_constants['strong']*n/2 *
                                                              loss_fn_constants['lip'])/np.log(1/gamma)))
start_grad_iter = int(np.max([start_grad_iter_distributed, start_grad_iter_desc]))



# create deletion sequence
n_deletions = 25
del_indices = np.random.randint(0, X_train.shape[0], size=n_deletions)
u_seq = [('-', ind,  X_train.iloc[ind], y_train.iloc[ind]) for ind in del_indices]

# instantiate algorithms
desc_del_algorithm = DescDel(X_train, X_test, y_train, y_test, epsilon=epsilon, delta=delta,
                        update_grad_iter=update_grad_iter, model_class=LogisticReg, start_grad_iter=start_grad_iter,
                             update_sequence=u_seq,l2_penalty=l2)


fed_algorithm = FedDescDel(X_train, X_test, y_train, y_test, epsilon=10.0,
                               delta=1.0 / np.power(len(y_train), 2), update_grad_iter=update_grad_iter,
                               model_class=model_log_reg_l2.LogisticReg,
                               update_sequence=u_seq, B=B, data_scale=data_scale, l2_penalty=0.05)

# run the algorithms
desc_del_algorithm.run()
fed_algorithm.run()

# plot accuracies as a function of iteration
desc_del_acc = (np.array(desc_del_algorithm.model_accuracies))
retrain_acc = np.array(desc_del_algorithm.scratch_model_accuracies)
dist_del_acc = np.array(fed_algorithm.model_accuracies)

#UPDATE
saved_list = [desc_del_acc, retrain_acc, dist_del_acc]
pickle.dump(saved_list, open(f'pickles/adult_I_{update_grad_iter}_updates_{n_deletions}_eps_{epsilon}_delta_{delta}.p', 'wb'))


plt.plot(desc_del_acc, color='blue', label='descent-to-delete unlearning')
plt.plot(retrain_acc, color='red', label='full retraining')
plt.plot(dist_del_acc, color='orange', label='federated unlearning')
plt.xlabel('update number')
plt.ylabel('accuracy')
plt.legend(loc='best', prop={'size': 10})
plt.title(f'adult dataset, (epsilon, delta) = {epsilon}, I = {update_grad_iter}')
plt.savefig('figures/adult_I_{update_grad_iter}_updates_{n_deletions}_eps_{epsilon}_delta_{delta}.png', dpi=300)
plt.show()


from clean_data import *
from sklearn import model_selection
from alg_desc_to_del import *
from model_log_reg_l2 import *
X, y = clean_communities(scale_and_center=True, normalize=True, intercept=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.1)
u_seq = [('-', 50,  X_train.iloc[50], y_train.iloc[50])]
desc_del_algo = DescDel(X_train, X_test, y_train, y_test, epsilon=1.0, delta=1.0/np.power(len(y_train), 2),
                        update_grad_iter=10, model_class=LogisticReg,
             start_grad_iter=100, update_sequence=u_seq, l2_penalty=1.0)

desc_del_algo.run()


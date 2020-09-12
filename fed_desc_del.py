import pdb
from clean_data import *
from sklearn import model_selection
import model_log_reg_l2


class FedDescDel:
    """Implement Algorithm 1 from Descent-to-Delete"""
    def __init__(self, X_train, X_test, y_train, y_test, epsilon, delta, update_grad_iter, model_class, start_grad_iter,
                 update_sequence, B,  l2_penalty=0.0):
        """
        sigma: noise added to guarantee (eps, delta) deletion
        loss_fn_constants: smoothness, strong convexity, lipschitz constant
        loss_fn_gradient: fn that given X, y, theta returns grad(f(X,y, theta))
        update_sequence: list of tuples [(x_1, y_1, +), (x_2, y_2, -), etc]
        update_grad_iter: number of allowed gradient iterations per round
        """
        self.X_train = X_train
        self.X_u = X_train
        self.B = B
        # dictionary where entry k is the indices in the kth partition
        self.K = int(np.ceil(np.sqrt(B)))
        self.bootstrap_size = int(float(B)/float(self.K))
        n = X_train.shape[0]
        m = int(np.round(float(B)/float(self.K)))
        self.bootstrap = {f'{x}': np.random.choice(size=m, a=n, replace=True) for x in range(self.K)}
        self.X_test = X_test
        self.y_train = y_train
        self.y_u = y_train
        self.y_test = y_test
        # list of dictionaries
        self.models = []
        # list of dicts
        self.noisy_models = []
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = 0
        self.gamma = 0
        self.model_class = model_class
        self.update_sequence = update_sequence
        self.update_grad_iter = update_grad_iter
        self.start_grad_iter = start_grad_iter
        self.datadim = X_train.shape[1]
        self.model_accuracies = []
        self.scratch_model_accuracies = []
        self.l2_penalty = l2_penalty

    def update(self, update):
        """Given update, output retrained model, noisy and secret state"""
        # update bootstrap sample
        updated_partitions = self.reservoir_sampling_update(update)
        # retrain on updated partitions - update models
        new_model_dict = self.train_partitions(self.update_grad_iter, updated_partitions, init_dict=self.models[-1])
        # calculate noisy theta (STOPPED HERE NEED TO IMPLEMENT)
        noisy_model = self.publish(new_model_dict)
        self.model_accuracies.append(self.get_test_accuracy(noisy_model))
        #self.scratch_model_accuracies.append(self.get_test_accuracy(new_model_scratch))

    def set_sigma(self):
        """Compute the noise level as a fn of (eps, delta)."""

        loss_fn_constants = self.models[-1]['0'].get_constants()
        gamma = (loss_fn_constants['smooth']-loss_fn_constants['strong'])/(loss_fn_constants['strong'] +
                                                                                     loss_fn_constants['smooth'])
        n = self.X_u.shape[0]
        e = np.log(self.B)/np.log(n)
        if e < 1 or e > 4/3:
            raise Exception("Bootstrap sample size must be in [n, n^4/3]")

        intermediate_calc = np.power(gamma, self.update_grad_iter*np.power(n, (4-3*e)/2))
        sigma_numerator = 4*np.sqrt(2)*loss_fn_constants['lip'] * intermediate_calc

        sigma_denominator = loss_fn_constants['strong']*n*(1-intermediate_calc) * \
                            (np.sqrt(np.log(2/self.delta + self.epsilon)) - np.sqrt(np.log(2/self.delta)))
        return sigma_numerator/sigma_denominator

    def train(self, X_u, y_u, iters, init=None):
        """In initial round """
        if init:
            model = self.model_class(init.theta, l2_penalty=self.l2_penalty)
        else:
            par = np.random.normal(0,1, self.datadim)
            par = par/(np.sqrt(np.sum(np.power(par, 2))))
            model = self.model_class(par, l2_penalty=self.l2_penalty)
        for _ in range(iters):
            model.proj_gradient_step(X_u, y_u)
        return model

    def train_partitions(self, iters, updated_partitions, init_dict):
        # if all partitions are being updated initialize with empty
        if len(updated_partitions) != self.K:
            new_model_dict = self.models[-1]
        else:
            new_model_dict = {}
        for part in updated_partitions:
            # use .loc instead of .iloc since selecting by index not row #
            x_part = self.X_u.loc[self.bootstrap[part], :]
            y_part = self.y_u.loc[self.bootstrap[part]]
            new_model_dict[part] = self.train(x_part, y_part, iters, init=init_dict[part])
        self.models.append(new_model_dict)
        return new_model_dict

    def publish(self, model):
        noise = np.random.normal(0, self.sigma, self.datadim)
        average_theta = np.average([m.theta for m in model.values()], axis=0)
        theta = average_theta + noise
        noisy_model = self.model_class(theta, l2_penalty=self.l2_penalty)
        self.noisy_models.append(noisy_model)
        return noisy_model

    def reservoir_sampling_update(self, update):
        """Update the bootstrap sample. First update X_u, y_u."""
        # update X_u, y_u
        self.update_data_set(update)

        # track and output partitions updated during this step
        updated_partitions = []
        if update[0] == '-':
            for key, value in self.bootstrap.items():
                if update[1] in value:
                    updated_partitions.append(key)
                    # sample indices is from X_u since its already been updated
                    self.bootstrap[key] = remove(update[1], value, sample_indices=self.X_u.index)
        else:
            # number of indices to be replaced
            N = np.random.binomial(self.B, 1.0/self.X_u.shape[0])
            replacement_indices = np.random.choice(self.B, N, replace=False)
            for key, value in self.bootstrap.items():
                index_interval = range(np.int(key)*len(self.bootstrap_size), (np.int(key) + 1)*len(self.bootstrap_size))
                adjusted_ind = [r - np.int(key)*len(self.bootstrap_size) for r in replacement_indices
                                if r in index_interval]
                if len(adjusted_ind) > 0:
                    updated_partitions.append(key)
                    self.bootstrap[key][adjusted_ind] = update[1]
        return updated_partitions

    def update_data_set(self, update):
        # Implement Reservoir Sampling:
        # update the underlying data set X_u
        """Update X_u, y_u with update (+, x, y) or (-, index, x, y)."""
        if update[0] == '-':
            self.X_u.drop(update[1], inplace=True)
            self.y_u.drop(update[1], inplace=True)

        if update[0] == '+':
            self.X_u.append(update[1], inplace=True)
            self.y_u.append(update[2], inplace=True)

    def run(self):
        # train on all partitions and return initial model dict (also append to models)
        init_model_dict = self.train_partitions(iters=self.start_grad_iter,
                                                updated_partitions=[f'{x}' for x in range(self.K)],
                                                init_dict={f'{x}': None for x in range(self.K)})
        self.set_sigma()
        initial_noisy_model = self.publish(init_model_dict)
        self.noisy_models.append(initial_noisy_model)
        self.model_accuracies.append(self.get_test_accuracy(initial_noisy_model))
        for update in self.update_sequence:
            self.update(update)

    def get_test_accuracy(self, model):
        y_hat = model.predict(self.X_test)
        return np.float(np.sum([np.array(y_hat) == np.array(self.y_test)]))/np.float(len(self.y_test))


def remove(index_to_remove, index_array, sample_indices):
    """remove index_to_remove from index_array and replace by a sample from sample indices """
    return [ind if ind != index_to_remove else np.random.choice(sample_indices) for ind in index_array]


if __name__ == "__main__":

    X, y = clean_adult(scale_and_center=True, normalize=True, intercept=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    n_deletions = 25
    del_indices = np.random.randint(0, X_train.shape[0], size=n_deletions)
    u_seq = [('-', ind, X_train.iloc[ind], y_train.iloc[ind]) for ind in del_indices]
    fed_algorithm = FedDescDel(X_train, X_test, y_train, y_test, epsilon=10.0,
                               delta=1.0 / np.power(len(y_train), 2), update_grad_iter=25,
                               model_class=model_log_reg_l2.LogisticReg, start_grad_iter=1000,
                               update_sequence=u_seq, B=5000, l2_penalty=0.05)
    fed_algorithm.run()

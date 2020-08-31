import numpy as np
import pdb

class FedDescDel:
    """Implement Algorithm 1 from Descent-to-Delete"""
    def __init__(self, X_train, X_test, y_train, y_test, epsilon, delta, update_grad_iter, model_class, start_grad_iter,
                 update_sequence, K, B,  l2_penalty=0):
        """
        sigma: noise added to guarantee (eps, delta) deletion
        loss_fn_constants: smoothness, strong convexity, lipschitz constant
        loss_fn_gradient: fn that given X, y, theta returns grad(f(X,y, theta))
        update_sequence: list of tuples [(x_1, y_1, +), (x_2, y_2, -), etc]
        update_grad_iter: number of allowed gradient iterations per round
        """
        self.X_train = X_train
        self.X_u = X_train
        # dictionary where entry k is the indices in the kth partition
        self.K = K
        self.B = B
        self.bootstrap_size = int(float(B)/float(K))
        n = X_train.shape[0]
        m = int(np.round(float(B)/float(K)))
        self.bootstrap = {f'{x}': np.random.choice(size=m, a=n, replace=True) for x in range(K)}
        self.X_test = X_test
        self.y_train = y_train
        self.y_u = y_train
        self.y_test = y_test
        # list of dictionaries
        self.models = []
        # list of dictionriers
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

        # update X_u
        self.update_data_set(update)
        # update boostrap sample
        updated_partitions = self.reservoir_sampling_update(update)
        # retrain on updated partitions - update models
        self.train_partitions(self.update_grad_iter, updated_partitions, init_dict=self.models[-1])
        # calculate noisy theta (STOPPED HERE NEED TO IMPLEMENT)
        self.publish(self.models[-1])
        self.model_accuracies.append(self.get_test_accuracy(noisy_model))
        self.scratch_model_accuracies.append(self.get_test_accuracy(new_model_scratch))

    def set_sigma(self):
        """Compute the noise level as a fn of (eps, delta)."""
        eta = 0.5
        loss_fn_constants = self.models[-1].get_constants()
        self.gamma = (loss_fn_constants['smooth']-loss_fn_constants['strong'])/(loss_fn_constants['strong'] +
                                                                                     loss_fn_constants['smooth'])
        #self.gamma = 1 - eta*loss_fn_constants['strong']
        sigma_numerator = 4*np.sqrt(2)*loss_fn_constants['lip']*np.power(self.gamma, self.update_grad_iter)
        sigma_denominator = (loss_fn_constants['strong'] * len(self.y_train) *
                             (1-np.power(self.gamma, self.update_grad_iter)))*((np.sqrt(np.log(1/self.delta) + self.epsilon)) -
                                                                     np.sqrt(np.log(1/self.delta)))
        self.sigma = sigma_numerator/sigma_denominator

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
        new_model_dict = self.models[-1]
        for part in updated_partitions:
            X_part = self.X_u.iloc[self.bootstrap[part], :]
            y_part = self.y_u.iloc[self.bootstrap[part]]
            new_model_dict[part] = self.train(X_part, y_part, iters, init_dict[part])
        self.models.append(new_model_dict)

    def publish(self, model):
        noise = np.random.normal(0, self.sigma, self.datadim)
        theta = model.theta + noise
        return self.model_class(theta, l2_penalty=self.l2_penalty)

    def reservoir_sampling_update(self, update):
        """Update the bootstrap sample. Assumes self.X_u already updated."""
        # track and output partitions updated during this step
        updated_partitions = []
        if update[0] == '-':
            for key, value in self.bootstrap:
                if update[1] in value:
                    updated_partitions.append(key)
                    self.bootstrap[key] = remove(update[1], value, self.X_u.index)
        else:
            # number of indices to be replaced
            N = np.random.binomial(self.B, 1.0/self.X_u.shape[0])
            replacement_indices = np.random.choice(self.B, N, replace=False)
            for key, value in self.bootstrap:
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
        self.X_u = self.X_u.reset_index(drop=True)
        self.y_u = self.y_u.reset_index(drop=True)
        if update[0] == '-':
            try:
                self.X_u = self.X_u.drop(update[1])
                self.y_u = self.y_u.drop(update[1])
            except:
                pdb.set_trace()
        if update[0] == '+':
            self.X_u = self.X_u.append(update[1])
            self.y_u = self.y_u.append(update[2])

    def run(self):
        # initialize noise level
        initial_model = self.train(iters=self.start_grad_iter, init=None)
        initial_scratch_model = self.train(iters=self.update_grad_iter, init=None)
        self.models.append(initial_model)
        self.set_sigma()
        initial_noisy_model = self.publish(initial_model)
        self.noisy_models.append(initial_noisy_model)
        self.scratch_models.append(initial_scratch_model)
        self.model_accuracies.append(self.get_test_accuracy(initial_noisy_model))
        self.scratch_model_accuracies.append(self.get_test_accuracy(initial_scratch_model))
        for update in self.update_sequence:
            self.update(update)

    def get_test_accuracy(self, model):
        y_hat = model.predict(self.X_test)
        return np.float(np.sum([np.array(y_hat) == np.array(self.y_test)]))/np.float(len(self.y_test))

def remove(index_to_remove, index_array, sample_indices):
    """remove index_to_remove from index_array and replace by a sample from sample indices """
    return [ind if ind != index_to_remove else np.random.choice(sample_indices) for ind in index_array]

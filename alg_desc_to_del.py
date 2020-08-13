import numpy as np
import pdb

class DescDel:
    """Implement Algorithm 1 from Descent-to-Delete"""
    def __init__(self, X_train, X_test, y_train, y_test, epsilon, delta, update_grad_iter, model_class, start_grad_iter,
                 update_sequence, l2_penalty=0):
        """
        sigma: noise added to guarantee (eps, delta) deletion
        loss_fn_constants: smoothness, strong convexity, lipschitz constant
        loss_fn_gradient: fn that given X, y, theta returns grad(f(X,y, theta))
        update_sequence: list of tuples [(x_1, y_1, +), (x_2, y_2, -), etc]
        update_grad_iter: number of allowed gradient iterations per round
        """
        self.X_train = X_train
        self.X_u = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_u = y_train
        self.y_test = y_test
        self.models = []
        self.noisy_models = []
        self.scratch_models = []
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
        self.update_data_set(update)
        new_model = self.train(iters=self.update_grad_iter, init=self.models[-1])
        new_model_scratch = self.train(iters=self.update_grad_iter, init=None)
        noisy_model = self.publish(new_model)
        self.models.append(new_model)
        self.noisy_models.append(noisy_model)
        self.scratch_models.append(new_model_scratch)
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

    def train(self, iters, init=None):
        """In initial round """
        if init:
            model = self.model_class(init.theta, l2_penalty=self.l2_penalty)
        else:
            par = np.random.normal(0,1, self.datadim)
            par = par/(np.sqrt(np.sum(np.power(par, 2))))
            model = self.model_class(par, l2_penalty=self.l2_penalty)
        for _ in range(iters):
            model.proj_gradient_step(self.X_u, self.y_u)
        return model

    def publish(self, model):
        noise = np.random.normal(0, self.sigma, self.datadim)
        theta = model.theta + noise
        return self.model_class(theta, l2_penalty=self.l2_penalty)

    def update_data_set(self, update):
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















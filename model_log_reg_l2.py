import numpy as np
import pandas as pd
import clean_data
import pdb
import copy

class LogisticReg:
    """Implement Algorithm 1 from Descent-to-Delete"""

    def __init__(self, theta, l2_penalty=.1):
        self.l2_penalty = l2_penalty
        self.theta = theta
        self.constants_dict = {'strong': self.l2_penalty, 'smooth': .25 + self.l2_penalty, 'diameter': 2.0,
                               'lip': 1.0 + 2.0*self.l2_penalty}

    def gradient_loss_fn(self, X, y):
        n = X.shape[0]
        log_grad = np.dot(np.diag(-y/(1 + np.exp(y*np.dot(X, self.theta)))), X)
        log_grad_sum = np.dot(np.ones(n), log_grad)
        reg_grad = 2*self.l2_penalty*self.theta
        return (reg_grad + (1/n)*log_grad_sum)

    def get_constants(self):
        # must have ||theta|| <= 1
        return self.constants_dict

    def proj_gradient_step(self, X, y):

        #eta = 2.0/(self.constants_dict['strong'] + self.constants_dict['smooth'])
        eta = 0.5
        current_theta = self.theta
        grad = self.gradient_loss_fn(X, y)
        # gradient update
        #next_theta = copy.deepcopy(self.theta)-eta*grad
        next_theta = current_theta - eta*grad
        if np.sum(np.power(next_theta, 2)) > 1:
            next_theta = next_theta/(clean_data.l2_norm(next_theta))
        #if np.sum(self.theta == next_theta) == len(next_theta):
            #pdb.set_trace()
        #    print('equal')
        self.theta = next_theta

    def predict(self, X):
        probs = 1.0/(1+np.exp(-np.dot(X, self.theta)))
        return pd.Series([1 if p >= .5 else -1 for p in probs])

if __name__ == "__main__":
    X, y = clean_data.clean_communities(scale_and_center=True, intercept=True, normalize=True)
    par = np.ones(X.shape[1])
    par = par/clean_data.l2_norm(par)
    model = LogisticReg(theta=par, l2_penalty=1.0)
    model.gradient_loss_fn(par, X, y)
    yhat = model.predict(X)


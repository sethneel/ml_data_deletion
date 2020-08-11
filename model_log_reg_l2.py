import numpy as np
import pandas as pd
import clean_data
import pdb

class LogisticReg:
    """Implement Algorithm 1 from Descent-to-Delete"""

    def __init__(self, theta, l2_penalty=.1):
        self.l2_penalty = l2_penalty
        self.theta = theta
        self.constants_dict = {'strong': self.l2_penalty, 'smooth': 1/4 + self.l2_penalty, 'diameter': 2,
                               'lip': 1 + 2*self.l2_penalty}

    def gradient_loss_fn(self, X, y):
        n = X.shape[0]
        log_grad = np.dot(np.diag(-y/(1 + np.exp(y*np.dot(X, self.theta)))), X)
        log_grad_sum = np.dot(np.ones(n), log_grad)
        reg_grad = 2*self.l2_penalty*self.theta
        return reg_grad + log_grad_sum

    def get_constants(self):
        # must have ||theta|| <= 1
        return self.constants_dict

    def proj_gradient_step(self, X, y):
        """Project onto norm <= 1"""
        eta = 2/(self.constants_dict['strong'] + self.constants_dict['smooth'])
        grad = self.gradient_loss_fn(X, y)
        self.theta = self.theta-eta*grad
        if np.sum(np.power(self.theta, 2)) > 1:
            self.theta = self.theta/(clean_data.l2_norm(self.theta))

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


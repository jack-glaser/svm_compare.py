import numpy as np
from qp import solve_QP

def linear_kernel(xj, xk):
    """
    Kernel Function, linear kernel (ie: regular dot product)

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :return: float32
    """
    return(np.dot(xj, xk))

def rbf_kernel(xj, xk, gamma = 0.1):
    """
    Kernel Function, radial basis function kernel or gaussian kernel

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :param gamma: parameter of the RBF kernel.
    :return: float32
    """
    return(np.exp(-gamma*((abs(xj-xk))^2)))
    

def polynomial_kernel(xj, xk, c = 0, d = 2):
    """
    Kernel Function, polynomial kernel

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :param c: mean of the polynomial kernel (np array)
    :param d: exponent of the polynomial (np array)
    :return: float32
    """
    return((np.dot(xj, xk) + c)**d)


class SVM(object):

    def __init__(self, kernel_func=linear_kernel, lambda_param=1):
        self.kernel_func = kernel_func
        self.lambda_param = lambda_param

    def train(self, inputs, labels):
        """
        train the model with the input data (inputs and labels),
        find the coefficients and constaints for the quadratic program and
        calculate the alphas

        :param inputs: inputs of data, a numpy array
        :param labels: labels of data, a numpy array
        :return: None
        """
        self.train_inputs = inputs
        self.train_labels = labels
        Q, c = self._objective_function()
        A, b = self._inequality_constraints()
        E, d = self._equality_constraints()
        self.alphas = solve_QP(Q, c, A, b, E, d)
        if np.isnan(self.alphas).any():
            print('WARNING: alphas contain at least one nan.')

        self.b = 0
        n = self.train_inputs.shape[0]
        for i in range(inputs.shape[0]):
            if not np.isclose(self.alphas[i], 0.0, rtol=0, atol=1e-3) and not np.isclose(self.alphas[i], 1/(2*n*self.lambda_param), rtol=0, atol=1e-3):
                for k in range(n):
                    self.b += self.alphas[k]*(2*self.train_labels[k]-1)*self.kernel_func(self.train_inputs[i, :], self.train_inputs[k, :])
                self.b -= (2*self.train_labels[i] - 1)
                break


    def _objective_function(self):
        """
        Generate the coefficients on the variables in the objective function for the
        SVM quadratic program.

        Recall the objective function is:
        minimize (1/2)x^T Q x + c^T x

        :return: two numpy arrays, Q and c which fully specify the objective function.
        """

        n = self.train_inputs.shape[0]
        P = np.zeros((n, n))
        for row in range(n):
            for col in range(n):
                P[row,col] = (2*self.train_labels[row]-1) * (2*self.train_labels[col]-1) *\
                             self.kernel_func(self.train_inputs[row,:], self.train_inputs[col,:])


        q = -1*np.ones((n,), dtype=np.float64)
        return P, q

    def _equality_constraints(self):
        """
        Generate the equality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ex = d.

        :return: two numpy arrays, E, the coefficients, and d, the values
        """

        n = self.train_inputs.shape[0]
        E = (2*self.train_labels-1).reshape((1,n))
        d = np.array([0.0], dtype=np.float64)
        return E, d

    def _inequality_constraints(self):
        """
        Generate the inequality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ax <= b.

        :return: two numpy arrays, A, the coefficients, and b, the values
        """
        n = self.train_inputs.shape[0]

        # x_{i} > 0 for all i
        A = -1.0 * np.identity(n)
        b = np.zeros((n,))

        # x_{i} < 1/(2*n*gamma)
        A = np.vstack([A, np.identity(n)])
        b = np.hstack([b, (1/(2*n*self.lambda_param))*np.ones((n,))])
        return A, b.T

    def predict(self, input):
        """
        Generate predictions given input.

        :param input: 2D Numpy array. Each row is a vector for which we output a prediction.
        :return: A 1D numpy array of predictions.
        """

        predictions = []
        n_training = self.train_inputs.shape[0]
        for test_idx in range(input.shape[0]):
            summation = 0
            for train_idx in range(n_training):
                summation += (2*self.train_labels[train_idx] - 1) * self.alphas[train_idx] *\
                             self.kernel_func(self.train_inputs[train_idx,:], input[test_idx, :])
            summation -= self.b
            guess = 1 if summation > 0 else 0
            predictions.append(guess)

        return np.array(predictions)

    def accuracy(self, inputs, labels):
        """
        Calculate the accuracy of the classifer given inputs and their true labels.

        :param inputs: 2D Numpy array which we are testing calculating the accuracy of.
        :param labels: 1D Numpy array with the inputs corresponding true labels.
        :return: A float indicating the accuracy (between 0.0 and 1.0)
        """
        predictions = self.predict(inputs)
        return  np.mean(predictions == labels)

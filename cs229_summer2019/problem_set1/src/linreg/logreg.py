import numpy as np
import util

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_val, y_val, clf.theta, plot_path)
    # Use np.savetxt to save predictions on eval set to save_path
    probs = clf.predict(x_val)
    np.savetxt(save_path, probs)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if not self.theta:
            self.theta = np.zeros(n, dtype=np.float32)

        for i in range(self.max_iter):
            grad = self._gradient(x, y)
            hess = self._hessian(x)

            old_theta = self.theta.copy()
            self.theta -= self.step_size * np.linalg.inv(hess).dot(grad)

            if self.verbose:
                print(f'Iter {i+1}: loss={self._loss(x, y)}')

            if np.sum(np.abs(self.theta - old_theta)) < self.eps:
                break
        if self.verbose:
            print('Final theta (logreg): {}'.format(self.theta))
        # *** END CODE HERE ***
    
    def _loss(self, x, y):
        probs = self._sigmoid(x.dot(self.theta))
        return -np.mean(y * np.log(probs + self.eps) + (1 - y) * np.log(1 - probs + self.eps))

    def _gradient(self, x, y):
        m, _ = x.shape
        probs = self._sigmoid(x.dot(self.theta))
        return -1 / m * x.T.dot(y - probs)

    def _hessian(self, x):
        m, _ = x.shape
        probs = self._sigmoid(x.dot(self.theta))
        diag = np.diag(probs * (1. - probs))
        return 1 / m * x.T.dot(diag).dot(x)

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_h = self._sigmoid(x.dot(self.theta))
        return y_h
        # *** END CODE HERE ***

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')

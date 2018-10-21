"""
The plot show the sequence of observations generated with the transitions
between them. We can see that, as specified by our transition matrix,
there are no transition between component 1 and 3.
"""
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt


def sampling(plot=False):
    startprob = np.array([0.6, 0.3, 0.1, 0.0])
    # The transition matrix, note that there are no transitions possible
    # between component 1 and 3
    transmat = np.array([
        [0.7, 0.2, 0.0, 0.1],
        [0.3, 0.5, 0.2, 0.0],
        [0.0, 0.3, 0.5, 0.2],
        [0.2, 0.0, 0.2, 0.6],
    ])
    # The means of each component
    means = np.array([
        [0.0,  0.0],
        [0.0, 11.0],
        [9.0, 10.0],
        [11.0, -1.0],
    ])
    # The covariance of each component
    covars = .5 * np.tile(np.identity(2), (4, 1, 1))

    # Build an HMM instance and set parameters
    model = hmm.GaussianHMM(n_components=4, covariance_type="full")

    # Instead of fitting it from the data, we directly set the estimated
    # parameters, the means and covariance of the components
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars

    # Generate samples
    X, Z = model.sample(500)

    # Plot the sampled data
    plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
             mfc="orange", alpha=0.7)

    # Indicate the component numbers
    for i, m in enumerate(means):
        plt.text(m[0], m[1], 'Component %i' % (i + 1),
                 size=17, horizontalalignment='center',
                 bbox=dict(alpha=.7, facecolor='w'))

    # Show results
    print('Transmission matrix')
    print(model.transmat_)
    if plot:
        plt.legend(loc='best')
        plt.show()
    return X, Z


def train(X):
    remodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    remodel.fit(X)
    Z2 = remodel.predict(X)
    print(remodel.monitor_)
    assert remodel.monitor_.converged
    return Z2


if __name__ == '__main__':
    X, Z = sampling()
    Z2 = train(X)
    print(Z)
    print(Z2)
    print('\nNumber of matching predictions')
    print(len([z1 == z2 for z1, z2 in zip(Z, Z2) if z1 == z2]))

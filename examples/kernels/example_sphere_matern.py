import numpy as np
import functools
from mayavi import mlab
from scipy.special import sph_harm

from geometric_kernels.kernels.geometric_kernels import MaternIntegratedKernel
from geometric_kernels.spaces.sphere import Sphere


def spherical_harmonic(x, m, n):
    """
    This function computes spherical harmonics on the sphere manifold.

    Parameters
    ----------
    :param x: point on the sphere                               (torch tensor)
    :param m: spherical harmonic number
    :param n: spherical harmonic number

    Returns
    -------
    :return: value of the spherical harmonic at x              (torch [1,1] array)
    """
    if np.ndim(x) < 2:
        x = x[None]
    x = x[0]

    # To spherical coordinates
    theta = np.arccos(x[2])
    phi = np.arctan2(x[1], x[0])

    y = sph_harm(m, n, phi, theta).real
    return y


if __name__ == "__main__":
    # Create the manifold
    dimension = 2
    sphere_manifold = Sphere(dimension)

    # Test function
    test_function = functools.partial(spherical_harmonic, m=1, n=2)

    # Define training observations
    nb_data_train = 10
    train_x = np.random.randn(nb_data_train, dimension + 1)
    train_x = train_x / np.linalg.norm(train_x, axis=1)[:, None]
    train_y = np.zeros(nb_data_train)
    for n in range(nb_data_train):
        train_y[n] = test_function(train_x[n])

    # Define the kernel function
    matern_kernel = MaternIntegratedKernel(sphere_manifold, nu=2.5, num_points_t=100)
    lengthscale = 1.

    # Generate a grid on the sphere
    radius = 1.
    n_elems = 30
    u = np.linspace(0, 2 * np.pi, n_elems)
    v = np.linspace(0, np.pi, n_elems)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    x_grid = np.vstack((np.reshape(x, (n_elems*n_elems)),
                        np.reshape(y, (n_elems*n_elems)),
                        np.reshape(z, (n_elems*n_elems)))).T

    # Generate prior and posterior samples
    nb_samples = 3
    # Prior samples
    prior_mean = np.zeros(n_elems**2)
    prior_covariance = matern_kernel.K(x_grid, x_grid, lengthscale=lengthscale)
    # model.covar_module(x_grid, x_grid).detach().numpy()
    prior_samples = np.random.multivariate_normal(prior_mean, prior_covariance, size=nb_samples)

    # Posterior mean and variance
    k_train = matern_kernel.K(train_x, train_x, lengthscale=lengthscale) + 1e-6 * np.eye(nb_data_train)
    inv_k_train = np.linalg.inv(k_train)
    k_train_test = matern_kernel.K(train_x, x_grid, lengthscale=lengthscale)
    mean_preds = np.dot(k_train_test.T, np.dot(inv_k_train, train_y))
    covariance_preds = prior_covariance - np.dot(k_train_test.T, np.dot(inv_k_train, k_train_test))
    # Posterior samples
    posterior_samples = np.random.multivariate_normal(mean_preds, covariance_preds, size=nb_samples)

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
    fig = mlab.gcf()
    mlab.clf()
    # Plot function on sphere
    colors = np.zeros(n_elems ** 2)
    for n in range(n_elems * n_elems):
        colors[n] = test_function(x_grid[n])
    colors = colors.reshape(n_elems, n_elems)
    max_val = np.max(colors)
    min_val = np.min(colors)
    colors = (colors - min_val) / (max_val - min_val)
    mlab.mesh(x + 2.5, y, z, scalars=colors, colormap='inferno')
    # Plot prior mean on sphere
    colors_mean = prior_mean.reshape((n_elems, n_elems))
    colors_mean = (colors_mean - min_val) / (max_val - min_val)
    mlab.mesh(x, y, z, scalars=colors_mean, colormap='inferno')
    # Plot prior variance on sphere
    colors_var = np.sqrt(np.diag(prior_covariance)).reshape(n_elems, n_elems)
    mlab.mesh(x - 2.5, y, z, scalars=colors_var, colormap='viridis')
    mlab.view(30, 120)
    mlab.show()

    # Plot prior samples
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
    fig = mlab.gcf()
    mlab.clf()
    for i in range(nb_samples):
        colors_mean = prior_samples[i].reshape((n_elems, n_elems))
        colors_mean = (colors_mean - min_val) / (max_val - min_val)
        mlab.mesh(x + 2.5 * i, y, z, scalars=colors_mean, colormap='inferno')
    mlab.view(30, 120)
    mlab.show()

    # Plot function on sphere, posterior mean, and posterior variance
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
    fig = mlab.gcf()
    mlab.clf()
    mlab.mesh(x + 2.5, y, z, scalars=colors, colormap='inferno')
    # Plot estimated mean on sphere
    colors_mean = mean_preds.reshape((n_elems, n_elems))
    colors_mean = (colors_mean - min_val) / (max_val - min_val)
    mlab.mesh(x, y, z, scalars=colors_mean, colormap='inferno')
    for n in range(nb_data_train):
        # Plot mean
        mlab.points3d(train_x[n, 0], train_x[n, 1], train_x[n, 2], color=(0., 0., 0.), scale_factor=0.06)
    # Plot estimated variance on sphere
    colors_var = np.sqrt(np.diag(covariance_preds)).reshape(n_elems, n_elems)
    mlab.mesh(x - 2.5, y, z, scalars=colors_var, colormap='viridis')
    for n in range(nb_data_train):
        # Plot mean
        mlab.points3d(train_x[n, 0] - 2.5, train_x[n, 1], train_x[n, 2], color=(0., 0., 0.), scale_factor=0.06)
    mlab.view(30, 120)
    mlab.show()

    # Plot posterior samples
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
    fig = mlab.gcf()
    mlab.clf()
    for i in range(nb_samples):
        colors_mean = posterior_samples[i].reshape((n_elems, n_elems))
        colors_mean = (colors_mean - min_val) / (max_val - min_val)
        mlab.mesh(x + 2.5 * i, y, z, scalars=colors_mean, colormap='inferno')
        for n in range(nb_data_train):
            mlab.points3d(train_x[n, 0] + 2.5 * i, train_x[n, 1], train_x[n, 2], color=(0., 0., 0.), scale_factor=0.06)
    mlab.view(30, 120)
    mlab.show()



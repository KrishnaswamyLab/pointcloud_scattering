import pickle
import numpy as np
from sklearn.impute import SimpleImputer
from scattering_utility import *
from utility import *

# The following rotated MNIS  code from:
# github.com/FengGmsu/manifold


NORTHPOLE_EPSILON = 1e-3



def meshgrid(b, grid_type='Driscoll-Healy'):
    return np.meshgrid(*linspace(b, grid_type), indexing='ij')



def linspace(b, grid_type='Driscoll-Healy'):
    if grid_type == 'Driscoll-Healy':
        beta = np.arange(2 * b) * np.pi / (2. * b)
        alpha = np.arange(2 * b) * np.pi / b
    elif grid_type == 'SOFT':
        beta = np.pi * (2 * np.arange(2 * b) + 1) / (4. * b)
        alpha = np.arange(2 * b) * np.pi / b
    elif grid_type == 'Clenshaw-Curtis':
        # beta = np.arange(2 * b + 1) * np.pi / (2 * b)
        # alpha = np.arange(2 * b + 2) * np.pi / (b + 1)
        # Must use np.linspace to prevent numerical errors that cause beta > pi
        beta = np.linspace(0, np.pi, 2 * b + 1)
        alpha = np.linspace(0, 2 * np.pi, 2 * b + 2, endpoint=False)
    elif grid_type == 'Gauss-Legendre':
        x, _ = leggauss(b + 1)  # TODO: leggauss docs state that this may not be only stable for orders > 100
        beta = np.arccos(x)
        alpha = np.arange(2 * b + 2) * np.pi / (b + 1)
    elif grid_type == 'HEALPix':
        #TODO: implement this here so that we don't need the dependency on healpy / healpix_compat
        from healpix_compat import healpy_sphere_meshgrid
        return healpy_sphere_meshgrid(b)
    elif grid_type == 'equidistribution':
        raise NotImplementedError('Not implemented yet; see Fast evaluation of quadrature formulae on the sphere.')
    else:
        raise ValueError('Unknown grid_type:' + grid_type)
    return beta, alpha



def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    # http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    """

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M



def rotate_grid(rot, grid):
    x, y, z = grid
    xyz = np.array((x, y, z))
    x_r, y_r, z_r = np.einsum('ij,jab->iab', rot, xyz)
    return x_r, y_r, z_r


def get_projection_grid(b, grid_type="Driscoll-Healy"):
    ''' returns the spherical grid in euclidean
    coordinates, where the sphere's center is moved
    to (0, 0, 1)'''
    theta, phi = meshgrid(b=b, grid_type=grid_type)
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return x_, y_, z_


def project_sphere_on_xy_plane(grid, projection_origin):
    ''' returns xy coordinates on the plane
    obtained from projecting each point of
    the spherical grid along the ray from
    the projection origin through the sphere '''

    sx, sy, sz = projection_origin
    x, y, z = grid
    z = z.copy() + 1

    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 1/2 * (-1 - sx) + -1
    ymin = 1/2 * (-1 - sy) + -1

    # ensure that plane projection
    # ends up on southern hemisphere
    rx = (qx - xmin) / (2 * np.abs(xmin))
    ry = (qy - ymin) / (2 * np.abs(ymin))

    return rx, ry




def sample_within_bounds(signal, x, y, bounds):
    ''' '''
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        sample = np.zeros((x.shape[0], x.shape[1]))
        sample[idxs] = signal[x[idxs], y[idxs]]
    return sample


def sample_bilinear(signal, rx, ry):
    ''' '''

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

    rx *= signal_dim_x
    ry *= signal_dim_y

    # discretize sample position
    ix = rx.astype(int)
    iy = ry.astype(int)

    # obtain four sample coordinates
    ix0 = ix - 1
    iy0 = iy - 1
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    # linear interpolation in x-direction
    fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
    fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry) * fx1 + (ry - iy0) * fx2


def project_2d_on_sphere(signal, grid, projection_origin=None):
    if projection_origin is None:
        projection_origin = (0, 0, 2 + NORTHPOLE_EPSILON)
    #project sphere grid which is in Euclidean space to a 2d grid though projection origin which is the noth pole
    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
    #sample and interpolation in x,y direction
    sample = sample_bilinear(signal, rx, ry)

    # ensure that only south hemisphere gets projected
    sample *= (grid[2] <= 1).astype(np.float64)

    # rescale signal to [0,1]
    sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
    sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)

    sample = (sample - sample_min) / (sample_max - sample_min)
    sample *= 255
    sample = sample.astype(np.uint8)

    return sample



def xyz2latlong(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    long = np.arctan2(y, x)
    xy2 = x**2 + y**2
    lat = np.arctan2(z, np.sqrt(xy2))
    return lat, long



def interp_r2tos2(sig_r2, V, method="linear", dtype=np.float32):
    """
    sig_r2: rectangular shape of (lat, long, n_channels)
    V: array of spherical coordinates of shape (n_vertex, 3)
    method: interpolation method. "linear" or "nearest"
    """
    ele, azi = xyz2latlong(V)
    nlat, nlong = sig_r2.shape[0], sig_r2.shape[1]
    dlat, dlong = np.pi/(nlat-1), 2*np.pi/nlong
    lat = np.linspace(-np.pi/2, np.pi/2, nlat)
    long = np.linspace(-np.pi, np.pi, nlong+1)
    sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1)
    intp = RegularGridInterpolator((lat, long), sig_r2, method=method)
    s2 = np.array([ele, azi]).T
    sig_s2 = intp(s2).astype(dtype)
    return sig_s2



trainset = datasets.MNIST(root ='MNIST',train=True, download=True)
testset = datasets.MNIST(root ='MNIST',train=False, download=True)



mnist_train = {}
mnist_train['images'] = trainset.data.numpy()
mnist_train['labels'] = trainset.targets.numpy()
mnist_test = {}
mnist_test['images'] = testset.data.numpy()
mnist_test['labels'] = testset.targets.numpy()




grid = get_projection_grid(b=30)
#below we rotate the grid
rot = rand_rotation_matrix(deflection=1.0)
new_grid = rotate_grid(rot, grid)

dataset = {}
for label, data in zip(["train", "test"], [mnist_train, mnist_test]):

    print("projecting {0} data set".format(label))
    current = 0
    signals = data['images'].reshape(-1, 28, 28).astype(np.float64)
    n_signals = signals.shape[0]
    projections = np.ndarray(
        (signals.shape[0], 2 * 30, 2 * 30),
        dtype=np.uint8)

    while current < n_signals:
        #-----------------------------------------------------------------------------------------------------
        #below select roatted grid or non-rotated grid
        #rotated_grid = grid is non rotated
        #rotated_grid = new_grid is rotated
        rotated_grid = new_grid
        idxs = np.arange(current, min(n_signals,
                                      current + 500))
        chunk = signals[idxs]
        projections[idxs] = project_2d_on_sphere(chunk, rotated_grid)
        current += 500
        print("\r{0}/{1}".format(current, n_signals), end="")
    print("")
    dataset[label] = {
        'images': projections,
        'labels': data['labels']
    }

x_train = dataset['train']['images']
x_test = dataset['test']['images']
y_train = dataset['train']['labels']
y_test = dataset['test']['labels']

eps_list = [3,4,6,7]
norm_list = [1, 2, 3, 4]
q = 0.5
K_list = [200]
for K in K_list:
    N_list = [1200]
    for N in N_list:
        X = np.random.normal(size=(N, 3))
        X = np.divide(X, np.linalg.norm(X, axis=1, keepdims=True))


        x_train_s2 = []
        print("Converting training set...")
        for i in range(x_train.shape[0]):
            x_train_s2.append(interp_r2tos2(x_train[i], X))

        x_test_s2 = []
        print("Converting test set...")
        for i in range(x_test.shape[0]):
            x_test_s2.append(interp_r2tos2(x_test[i], X))

        x_train_s2 = np.stack(x_train_s2, axis=0)
        x_test_s2 = np.stack(x_test_s2, axis=0)



        d = {"train_inputs": x_train_s2,"train_labels": y_train,"test_inputs": x_test_s2,"test_labels": y_test}
            

        training_signal = d['train_inputs']
        test_signal = d['test_inputs']
        training_Y = d['train_labels']
        test_Y = d['test_labels']
        # remove the 6s from the dataset, since they are indistinguishable from the 9s under random rotation
	train_indices = [i for i, x in enumerate(training_Y) if x == 6]
        test_indices = [i for i, x in enumerate(test_Y) if x == 6]
        training_signal_ = [x for i, x in enumerate(training_signal) if i not in train_indices]
        test_signal_ = [x for i, x in enumerate(test_signal) if i not in test_indices]
        training_Y_ = [x for i, x in enumerate(training_Y) if i not in train_indices]
        test_Y_ = [x for i, x in enumerate(test_Y) if i not in test_indices]
        

        for this_eps in eps_list:
            vals, vecs, eps  = compute_eigen(X, this_eps, K, d=2,eps_quantile=q)
            J = 8
            training_feature, test_feature = compute_all_features(vals, vecs, training_signal_, test_signal_, training_Y_, test_Y_, eps, N, norm_list, J)
            
            G_pool = [0.00001,0.0001,0.001,0.005,0.01,0.02,0.04]
            C_pool = [50,100,250,650]

            training_feature_z = scipy.stats.mstats.zscore(training_feature,0)
            this_max = np.amax(training_feature_z[~np.isinf(training_feature_z)])
            training_feature_z[np.isinf(training_feature_z)] = this_max            
            test_feature_z = scipy.stats.mstats.zscore(test_feature,0)
            this_max = np.amaz(test_feature_z[~np.isinf(test_feature_z)])
            test_feature_z[np.isinf(test_feature_z)] = this_max
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp = imp.fit(training_feature_z)
            training_feature_z = imp.transform(training_feature_z)
            test_feature_z = imp.trainsform(test_feature_z)

            # change the first argument here to change the number of folds used to validate the hyperparameters
            result,prediction_acc = cross_validate(5,training_feature_z,training_Y_,test_feature_z,test_Y_,G_pool,C_pool)
            print("Cross-validation result: %s" % result)
            print('prediction accuracy',prediction_acc)
            print("N = %d, K = %d, eps = %f, J = %d" % (N, K, eps, J))

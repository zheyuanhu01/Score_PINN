import haiku as hk
import jax
import optax
import jax.numpy as jnp
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from scipy.stats import multivariate_t
from scipy.special import gamma

parser = argparse.ArgumentParser(description='PINN Training')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--epochs', type=int, default=100001)
parser.add_argument('--N_SM', type=int, default=1000)
parser.add_argument('--N_SSM', type=int, default=1000)
parser.add_argument('--N_PINN', type=int, default=1000)
parser.add_argument('--N_PINN_LL', type=int, default=1000)
parser.add_argument('--PINN_L', type=int, default=4)
parser.add_argument('--PINN_h', type=int, default=128)
parser.add_argument('--method', type=int, default=0)
parser.add_argument('--V', type=int, default=1)

parser.add_argument('--cauchy_gamma', type=float, default=1)
args = parser.parse_args()
print(args)

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)

keys = jax.random.split(key, 3)
Gamma = jax.random.uniform(keys[0], shape=(args.dim // 2, )) * 0.1 + 1
Gamma = jnp.concatenate([Gamma, 1 / Gamma])
key = keys[2]

constant = np.log(gamma((1 + args.dim) / 2)) - args.dim * np.log(args.cauchy_gamma) - \
    0.5 * np.log(np.pi) - args.dim / 2 * np.log(np.pi)
constant /= args.dim

def sample_test(batch_size, MCMC_size):
    d = args.dim
    mean, cov = np.zeros(d), np.eye(d)
    cauchy = multivariate_t(loc=mean, shape=cov, df=1)
    x0 = cauchy.rvs(size=batch_size) * args.cauchy_gamma
    t = np.random.rand(batch_size, ) + 1e-2
    x = np.random.randn(batch_size, d) * np.sqrt(Gamma).reshape(1, d) * np.sqrt(t).reshape(-1, 1) + x0

    def initial_logpdf(x):
        return constant - (args.dim + 1) / (2 * args.dim) * jnp.log(args.cauchy_gamma**2 + jnp.sum(x**2))

    def func(x, t, y):
        y = y * jnp.sqrt(t)
        return initial_logpdf(x - y)
    
    y = np.random.randn(MCMC_size, d) * np.sqrt(Gamma).reshape(1, d) # (MCMC_size, d)
    logpdf = jax.vmap(jax.vmap(func, in_axes=(None, None, 0)), (0, 0, None))(x, t, y) # batch_size, MCMC_size
    logpdf *= d
    return x, t, logpdf

X, T, Q = [], [], []
for _ in tqdm(range(100)):
    x, t, q = sample_test(1, int(1e7))
    q_max = q.max(1)
    q = q - q_max.reshape(-1, 1)
    q = jnp.exp(q)
    q = jnp.mean(q, 1)
    q = jnp.log(q) + q_max
    q = q / args.dim
    X.append(x); T.append(t); Q.append(q)
X, T, Q = jnp.concatenate(X, 0), jnp.concatenate(T, 0), jnp.concatenate(Q, 0)
print(Q.min(), Q.max())
np.savetxt("BC_Data_X_" + str(args.dim) + ".txt", X)
np.savetxt("BC_Data_T_" + str(args.dim) + ".txt", T)
np.savetxt("BC_Data_Q_" + str(args.dim) + ".txt", Q)

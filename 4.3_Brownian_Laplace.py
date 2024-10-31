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

parser.add_argument('--laplace_b', type=float, default=1)
args = parser.parse_args()
print(args)

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)

keys = jax.random.split(key, 3)
Gamma = jax.random.uniform(keys[0], shape=(args.dim // 2, )) * 0.1 + 1
Gamma = jnp.concatenate([Gamma, 1 / Gamma])
A = jax.random.normal(keys[1], shape=(args.dim, args.dim))
A, _ = jnp.linalg.qr(A)
key = keys[2]

class MLP(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x, t):
        X = jnp.hstack([x, t])
        for dim in self.layers[:-1]:
            X = hk.Linear(dim)(X)
            X = jnp.tanh(X)
        X = hk.Linear(self.layers[-1])(X)
        return X

class MLP_Q(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x, t):
        X = jnp.hstack([x, t])
        for dim in self.layers[:-1]:
            X = hk.Linear(dim)(X)
            X = jnp.tanh(X)
        X = hk.Linear(self.layers[-1])(X)
        return X[0]

def sample_test(batch_size, MCMC_size):
    d = args.dim
    x0 = np.random.laplace(size=(batch_size, d))
    t = np.random.rand(batch_size, ) + 1e-2
    x = A.T @ np.sqrt(np.diag(Gamma)) @ jax.random.normal(keys[0], shape=(d, batch_size))
    x = x.T * np.sqrt(t.reshape(-1, 1))
    x += x0

    def initial_logpdf(x):
        return -jnp.log(2 * args.laplace_b) - jnp.mean(jnp.abs(x)) / args.laplace_b

    def func(x, t, y):
        y = y * jnp.sqrt(t)
        return initial_logpdf(x - y)
    
    y = np.random.randn(MCMC_size, d) @ np.sqrt(jnp.diag(Gamma)) @ A # (MCMC_size, d)
    logpdf = jax.vmap(jax.vmap(func, in_axes=(None, None, 0)), (0, 0, None))(x, t, y) # batch_size, MCMC_size
    logpdf *= d
    return x, t, logpdf

X, T, Q = [], [], []
for _ in tqdm(range(20)):
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

class PINN:
    def __init__(self, args):
        self.dim = args.dim; self.epoch = args.epochs; self.SEED = args.SEED; self.method = args.method; self.V = args.V;
        self.N_SM, self.N_SSM, self.N_PINN, self.N_PINN_LL = args.N_SM, args.N_SSM, args.N_PINN, args.N_PINN_LL
        self.laplace_b = args.laplace_b
        layers = [args.PINN_h] * (args.PINN_L - 1) + [self.dim]
        @hk.transform
        def network(x, t): return MLP(layers=layers)(x, t)
        self.net = hk.without_apply_rng(network)
        self.pred_fn = jax.vmap(self.net.apply, (None, 0, 0))
        self.X, self.T, self.Q = X, T, Q
        self.params = self.net.init(key, self.X[0], self.T[0])

        layers_Q = [args.PINN_h] * (args.PINN_L - 1) + [1]
        @hk.transform
        def network_Q(x, t): return MLP_Q(layers=layers_Q)(x, t)
        self.net_Q = hk.without_apply_rng(network_Q)
        self.pred_fn_Q = jax.vmap(self.net_Q.apply, (None, 0, 0))
        self.params_Q = self.net_Q.init(key, self.X[0], self.T[0])

    def resample(self, N, rng):
        keys = jax.random.split(rng, 4)
        tf = jax.random.uniform(keys[1], shape=(N, )) + 1e-2
        x0 = jax.random.laplace(keys[0], shape=(N, self.dim))
        xf = A.T @ jnp.sqrt(jnp.diag(Gamma)) @ jax.random.normal(keys[2], shape=(self.dim, N))
        xf = xf.T * jnp.sqrt(tf).reshape(-1, 1) + x0
        return x0, xf, tf, keys[3]

    def score_matching(self, params, x0, x, t):
        s = self.net.apply(params, x, t) # score prediction
        residual = s * jnp.sqrt(t) + (x - x0) / jnp.sqrt(t) / Gamma
        return residual
    def get_loss_score_matching(self, params, x0, x, t):
        pred = jax.vmap(self.score_matching, in_axes=(None, 0, 0, 0))(params, x0, x, t)
        loss = jnp.mean(pred ** 2)
        return loss

    def sliced_score_matching(self, params, x, t):
        s = 0.5 * jnp.sum(self.net.apply(params, x, t)**2)
        fn = lambda x: self.net.apply(params, x, t)
        nabla_s = jnp.sum(jnp.diag(jax.jacfwd(fn)(x)))
        return nabla_s + s
    def get_loss_sliced_score_matching(self, params, x, t):
        pred = jax.vmap(self.sliced_score_matching, in_axes=(None, 0, 0))(params, x, t)
        loss = jnp.mean(pred)
        return loss

    def sliced_score_matching_hte(self, params, x, t, v):
        s = 0.5 * jnp.sum(self.net.apply(params, x, t)**2)
        fn = lambda x: self.net.apply(params, x, t)
        nabla_s = jnp.dot(v, jax.jvp(fn, (x,), (v,))[1])
        return nabla_s + s
    def get_loss_sliced_score_matching_hte(self, params, x, t, v):
        pred = jax.vmap(jax.vmap(self.sliced_score_matching_hte, in_axes=(None, 0, 0, None)), in_axes=(None, None, None, 0))(params, x, t, v)
        pred = pred.mean(0)
        loss = jnp.mean(pred)
        return loss

    def residual_pred(self, params, x, t):
        gn = lambda x: Gamma * self.net.apply(params, x, t)
        fn = lambda x: 0.5 * jnp.sum(Gamma * self.net.apply(params, x, t)**2) + \
            0.5 * jnp.sum(jnp.diag(jax.jacrev(gn)(x)))
        s_t = jax.jacrev(self.net.apply, argnums=2)(params, x, t)
        pred_x = jax.jacfwd(fn)(x)
        return s_t - pred_x
    def get_loss_pinn(self, params, x, t):
        pred = jax.vmap(self.residual_pred, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(pred**2)

    def residual_pred_hte(self, params, x, t, v):
        gn = lambda x: Gamma * self.net.apply(params, x, t)
        fn = lambda x: 0.5 * jnp.sum(Gamma * self.net.apply(params, x, t)**2) + \
            0.5 * jnp.dot(v, jax.jvp(gn, (x,), (v,))[1])
        s_t = jax.jacrev(self.net.apply, argnums=2)(params, x, t)
        pred_x = jax.jacfwd(fn)(x)
        return s_t - pred_x
    def get_loss_pinn_hte(self, params, x, t, v):
        pred = jax.vmap(jax.vmap(self.residual_pred_hte, in_axes=(None, 0, 0, None)), in_axes=(None, None, None, 0))(params, x, t, v)
        pred = pred.mean(0)
        return jnp.mean(pred**2)
    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state, rng):
        x0, xf, tf, rng = self.resample(self.N_PINN, rng)
        keys = jax.random.split(rng, 2)
        v = 2 * (jax.random.randint(keys[0], shape=(self.V, self.dim), minval=0, maxval=2) - 0.5)
        if self.method == 0: # vanilla score matching
            current_loss, gradients = jax.value_and_grad(self.get_loss_score_matching)(params, x0, xf, tf)
        elif self.method == 1: # sliced score matching
            current_loss, gradients = jax.value_and_grad(self.get_loss_sliced_score_matching)(params, xf, tf)
        elif self.method == 2: # sliced score matching + HTE
            current_loss, gradients = jax.value_and_grad(self.get_loss_sliced_score_matching_hte)(params, xf, tf, v)
        elif self.method == 3: # vanilla PINN
            current_loss, gradients = jax.value_and_grad(self.get_loss_pinn)(params, xf, tf)
        elif self.method == 4: # PINN + HTE
            current_loss, gradients = jax.value_and_grad(self.get_loss_pinn_hte)(params, xf, tf, v)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, keys[1]
    
    def train_score(self):
        lr = optax.exponential_decay(init_value=1e-3, transition_steps=10000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)
        self.rng = jax.random.PRNGKey(self.SEED)
        for n in tqdm(range(self.epoch)):
            current_loss, self.params, self.opt_state, self.rng = self.step(self.params, self.opt_state, self.rng)
            if n % 1000 == 0: print('epoch %d, loss: %e'%(n, current_loss))

    def residual_pred_ll(self, params, x, t):
        fn = lambda x: Gamma * self.net.apply(self.params, x, t)
        pred_x = 0.5 * jnp.mean(Gamma * self.net.apply(self.params, x, t)**2) + \
            0.5 * jnp.mean(jnp.diag(jax.jacrev(fn)(x)))
        q_t = jax.jacrev(self.net_Q.apply, argnums=2)(params, x, t)
        return self.dim * (q_t - pred_x)
    def boundary_pred_ll(self, params, x, t):
        q = self.net_Q.apply(params, x, t)
        q_ref = - jnp.log(2 * self.laplace_b) - jnp.mean(jnp.abs(x)) / self.laplace_b
        return self.dim * (q - q_ref)
    def get_loss_pinn_ll(self, params, x, t, xb, tb):
        pred = jax.vmap(self.residual_pred_ll, in_axes=(None, 0, 0))(params, x, t)
        pred_b = jax.vmap(self.boundary_pred_ll, in_axes=(None, 0, 0))(params, xb, tb)
        return jnp.mean(pred**2) + 20 * jnp.mean(pred_b**2)
    @partial(jax.jit, static_argnums=(0,))
    def step_Q(self, params, opt_state, rng):
        x0, xf, tf, rng = self.resample(self.N_PINN_LL, rng)
        t0 = jnp.zeros((self.N_PINN_LL, ))
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn_ll)(params, xf, tf, x0, t0)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng
    def train_ll(self):
        lr = optax.exponential_decay(init_value=1e-3, transition_steps=10000, decay_rate=0.9)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params_Q)
        self.rng = jax.random.PRNGKey(self.SEED)
        for n in tqdm(range(self.epoch)):
            current_loss, self.params_Q, self.opt_state, self.rng = self.step_Q(self.params_Q, self.opt_state, self.rng)
            if n % 1000 == 0: print('epoch %d, loss: %e, LL L2/Linf %e/%e, PDF L2/Linf: %e/%e'%(n, current_loss, *self.L2_pinn_Q(self.params_Q, self.X, self.T, self.Q)))
        self.final_error_ll = self.L2_pinn_Q(self.params_Q, self.X, self.T, self.Q)
    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn_Q(self, params, x, t, q):
        pred = self.pred_fn_Q(params, x, t)
        rel_error = jnp.linalg.norm(q - pred, 2) / jnp.linalg.norm(q, 2)
        return rel_error
    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn_Q(self, params, x, t, q):
        d = self.dim
        pred = self.pred_fn_Q(params, x, t)
        LL_L2_error = jnp.linalg.norm(q - pred, 2) / jnp.linalg.norm(q, 2)
        LL_Linf_error = jnp.max(jnp.abs(q - pred)) / jnp.max(jnp.abs(q))
        q, pred = q - q.max(), pred - q.max()
        PDF_L2_error = jnp.linalg.norm(jnp.exp(d * q) - jnp.exp(d * pred), 2) / jnp.linalg.norm(jnp.exp(d * q), 2)
        PDF_Linf_error = jnp.max(jnp.abs(jnp.exp(d * q) - jnp.exp(d * pred))) / jnp.max(jnp.exp(d * q))
        return LL_L2_error, LL_Linf_error, PDF_L2_error, PDF_Linf_error

model = PINN(args)
model.train_score()
model.train_ll()
print(model.method, model.final_error_ll)
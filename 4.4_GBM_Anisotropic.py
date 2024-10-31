import haiku as hk
import jax
import optax
import jax.numpy as jnp
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
import os

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
parser.add_argument('--T', type=float, default=0.3)
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

Sigma = A.T @ jnp.diag(Gamma) @ A
Sigma_sqrt = A.T @ jnp.diag(jnp.sqrt(Gamma)) @ A

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
        X = X * t - 1 / x - (A.T @ jnp.diag(1 / (Gamma)) @ A) @ jnp.log(x) / x
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
        X = X[0] * t - 1 / 2 * jnp.log(2 * np.pi) - 0.5 * jnp.mean(jnp.log(Gamma)) - \
             jnp.log(x) @ (A.T @ jnp.diag(1 / (Gamma)) @ A) @ jnp.log(x) / 2 / args.dim - jnp.mean(jnp.log(x))
        return X

class PINN:
    def __init__(self, args):
        self.dim = args.dim; self.epoch = args.epochs; self.SEED = args.SEED; self.method = args.method; self.V = args.V;
        self.N_SM, self.N_SSM, self.N_PINN, self.N_PINN_LL = args.N_SM, args.N_SSM, args.N_PINN, args.N_PINN_LL
        self.T_end = args.T
        layers = [args.PINN_h] * (args.PINN_L - 1) + [self.dim]
        @hk.transform
        def network(x, t): return MLP(layers=layers)(x, t)
        self.net = hk.without_apply_rng(network)
        self.pred_fn = jax.vmap(self.net.apply, (None, 0, 0))
        self.X, self.T, self.U, self.Q = self.sample_test(jax.random.PRNGKey(args.SEED))
        self.params = self.net.init(key, self.X[0], self.T[0])

        layers_Q = [128] * 3 + [1]
        @hk.transform
        def network_Q(x, t): return MLP_Q(layers=layers_Q)(x, t)
        self.net_Q = hk.without_apply_rng(network_Q)
        self.pred_fn_Q = jax.vmap(self.net_Q.apply, (None, 0, 0))
        self.params_Q = self.net_Q.init(key, self.X[0], self.T[0])

    def sample_test(self, rng):
        keys = jax.random.split(rng, 3)
        t = jax.random.uniform(keys[1], shape=(int(1e4), )) * self.T_end + 1e-2
        def transform_x(x, t): return (A.T @ jnp.diag(jnp.sqrt(1 - jnp.exp(-t) + Gamma)) @ A) @ x
        x = jax.random.normal(keys[0], shape=(int(1e4), self.dim))
        x = jax.vmap(transform_x, in_axes=(0, 0))(x, t)
        x = jnp.exp(x)
        def score(x, t): return -1 / x - (A.T @ jnp.diag(1 / (Gamma + 1 - jnp.exp(-t))) @ A) @ jnp.log(x) / x
        exact_score = jax.vmap(score, in_axes=(0, 0))(x, t)
        def LL(x, t): return - 1 / 2 * jnp.log(2 * np.pi) - \
            0.5 * jnp.mean(jnp.log(Gamma + 1 - jnp.exp(-t))) - \
            jnp.log(x) @ (A.T @ jnp.diag(1 / (Gamma + 1 - jnp.exp(-t))) @ A) @ jnp.log(x) / 2 / self.dim - \
            jnp.mean(jnp.log(x))
        exact_ll = jax.vmap(LL, in_axes=(0, 0))(x, t)
        return x, t, exact_score, exact_ll
    def resample(self, N, rng):
        keys = jax.random.split(rng, 4)
        x0 = jax.random.normal(keys[0], shape=(N, self.dim))
        tf = jax.random.uniform(keys[1], shape=(N, )) * self.T_end + 1e-2
        xf = jax.random.normal(keys[2], shape=(N, self.dim)) * jnp.sqrt((1 - jnp.exp(-tf)).reshape(N, 1))
        def transform_x(x): return (A.T @ jnp.diag(jnp.sqrt(Gamma)) @ A) @ x
        x0 = jax.vmap(transform_x, in_axes=(0))(x0)
        xf += x0
        x0 = jnp.exp(x0)
        xf = jnp.exp(xf)
        return x0, xf, tf, keys[3]
    
    def score_matching(self, params, x0, x, t):
        s = self.net.apply(params, x, t) # score prediction
        residual = s * jnp.sqrt(1 - jnp.exp(-t)) + (x0 / x) * jnp.sqrt(1 - jnp.exp(-t)) + \
            (jnp.log(x) - jnp.log(x0)) / x * x0 / jnp.sqrt(1 - jnp.exp(-t))
        return residual
    def get_loss_score_matching(self, params, x0, x, t):
        pred = jax.vmap(self.score_matching, in_axes=(None, 0, 0, 0))(params, x0, x, t)
        loss = jnp.mean(pred ** 2)
        return loss

    def sliced_score_matching(self, params, x, t):
        s = 0.5 * jnp.sum(self.net.apply(params, x, t)**2)
        fn = lambda x: self.net.apply(params, x, t)
        nabla_s = jnp.sum(jnp.diag(jax.jacfwd(fn)(x)))
        return (nabla_s + s)
    def get_loss_sliced_score_matching(self, params, x, t):
        pred = jax.vmap(self.sliced_score_matching, in_axes=(None, 0, 0))(params, x, t)
        loss = jnp.mean(pred)
        return loss

    def residual_pred(self, params, x, t):
        gn = lambda x: jnp.exp(-t) * self.net.apply(params, x, t) * x**2
        fn = lambda x: 0.5 * jnp.sum(jnp.exp(-t) * (x * self.net.apply(params, x, t))**2) + \
            0.5 * jnp.sum(jnp.diag(jax.jacrev(gn)(x))) + 0.5 * jnp.exp(-t) * jnp.dot(x, self.net.apply(params, x, t))
        s_t = jax.jacrev(self.net.apply, argnums=2)(params, x, t)
        pred_x = jax.jacfwd(fn)(x)
        return (s_t - pred_x)
    def get_loss_pinn(self, params, x, t):
        pred = jax.vmap(self.residual_pred, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(pred**2)
    
    def residual_pred_hte(self, params, x, t, v):
        gn = lambda x: jnp.exp(-t) * self.net.apply(params, x, t) * x**2
        fn = lambda x: 0.5 * jnp.sum(jnp.exp(-t) * (x * self.net.apply(params, x, t))**2) + \
            0.5 * jnp.dot(v, jax.jvp(gn, (x,), (v,))[1]) + 0.5 * jnp.exp(-t) * jnp.dot(x, self.net.apply(params, x, t))
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
            if n % 1000 == 0: print('epoch %d, loss: %e, L1_s: %e, L2_s: %e'%(n, \
                current_loss, *self.L2_pinn(self.params)))
        self.final_error_score = self.L2_pinn(self.params)

    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn(self, params):
        x, t, u = self.X, self.T, self.U
        pred = self.pred_fn(params, x, t)
        mae_error, mse_error = jnp.mean(jnp.abs(pred - u)), jnp.sqrt(jnp.mean((pred - u)**2))
        return mae_error, mse_error

    def residual_pred_ll(self, params, x, t):
        gn = lambda x: jnp.exp(-t) * self.net.apply(self.params, x, t) * x**2
        pred_x = 0.5 * jnp.sum(jnp.exp(-t) * (x * self.net.apply(self.params, x, t))**2) + \
            0.5 * jnp.sum(jnp.diag(jax.jacrev(gn)(x))) + 0.5 * jnp.exp(-t) * jnp.dot(x, self.net.apply(self.params, x, t)) + \
            0.5 * self.dim * jnp.exp(-t)
        q_t = jax.jacrev(self.net_Q.apply, argnums=2)(params, x, t)
        return (self.dim * q_t - pred_x)
    def get_loss_pinn_ll(self, params, x, t):
        pred = jax.vmap(self.residual_pred_ll, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(pred**2)
    @partial(jax.jit, static_argnums=(0,))
    def step_Q(self, params, opt_state, rng):
        _, xf, tf, rng = self.resample(self.N_PINN_LL, rng)
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn_ll)(params, xf, tf)
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
            if n % 1000 == 0: print('epoch %d, loss %e, LL L2/Linf %e %e, PDF L2/Linf %e/%e'%(n, current_loss, *self.L2_pinn_Q(self.params_Q, self.X, self.T, self.Q)))
        self.final_error_ll = self.L2_pinn_Q(self.params_Q, self.X, self.T, self.Q)
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
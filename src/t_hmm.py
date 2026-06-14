import jax.numpy as jnp
import jax.random as jr
from jax import jit, grad
import dynamax.hidden_markov_model as hmm

def t_log_prob(x, loc, scale, df):
    """Log-probability of the location-scale t-distribution."""
    z = (x - loc) / scale
    log_prob = (jnp.lgamma((df + 1) / 2) 
                - jnp.lgamma(df / 2) 
                - 0.5 * jnp.log(df * jnp.pi) 
                - jnp.log(scale) 
                - (df + 1) / 2 * jnp.log1p(z**2 / df))
    return log_prob.sum()


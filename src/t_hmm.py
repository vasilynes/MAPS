import jax.numpy as jnp
import jax.random as jr
from jax import jit, grad
import tensorflow_probability.substrates.jax.distributions as tfd
from typing import NamedTuple
from jaxtyping import Array, Float
from dynamax.parameters import ParameterProperties

class StudentTDistribution(tfd.Distribution):
    """
    Multivariate location-scale Student distribution p(x | loc, scale, df).
    The emission dims are assumed to be independent, univariate, t-distributed.
    """
    def __init__(
            self,
            loc, 
            scale,
            df, 
            validate_args=False,
            name='StudentTDistribution'
    ):
        self._loc = loc
        self._scale = scale
        self._df = df
        super().__init__(
            dtype=jnp.float32,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=True,
            name=name
        )

    def _log_prob(self, x):
        z = (x - self._loc) / self._scale
        lp = (
            jnp.lgamma((self._df + 1) / 2)
            - jnp.lgamma(self._df / 2)
            - 0.5 * jnp.log(self._df * jnp.pi)
            - jnp.log(self._scale)
            - (self._df + 1) / 2 * jnp.log1p(z ** 2 / self._df)
        )
        return lp.sum(axis=-1) # Sum over emission dimensions
    
    def _sample_n(self, n, seed):
        """
        Sample via Gaussian-chi2:
        x = loc + scale * Z / sqrt(V/df)
        """
        key1, key2 = jr.split(seed)
        shape = (n,) + self._loc.shape
        z = jr.normal(key1, shape)
        v = jr.gamma(key2, self._df / 2, shape) * 2 
        return self._loc + self._scale * z / jnp.sqrt(v / self._df)
    
    def _event_shape(self):
        return self._loc_shape
    
    def _event_shape_tensor(self):
        return jnp.array(self._loc.shape, dtype=jnp.int32)
    
    def _batch_shape(self):
        return ()
    
    def _batch_shape_tensor(self):
        return jnp.zeros([], dtype=jnp.int32).shape
    
class ParamsTHMMEmissions(NamedTuple):
    """
    Per-state emission parameters.
 
    Shapes are (num_states, emission_dim) for locs and scales,
    and (num_states,) for df (one df per state, shared across dims).
    """
    locs:   Float[Array, "num_states emission_dim"] | ParameterProperties
    scales: Float[Array, "num_states emission_dim"] | ParameterProperties
    dfs:    Float[Array, " num_states"] | ParameterProperties


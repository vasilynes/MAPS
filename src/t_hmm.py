import jax.numpy as jnp
import jax.random as jr
from jax import jit, grad
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

from typing import NamedTuple
from jaxtyping import Array, Float

from dynamax.parameters import ParameterProperties
from dynamax.types import Scalar
from dynamax.hidden_markov_model.models.abstractions import HMMEmissions, HMM
from dynamax.hidden_markov_model.models.initial import (
    StandardHMMInitialState, ParamsStandardHMMInitialState
)
from dynamax.hidden_markov_model.models.transitions import (
    StandardHMMTransitions, ParamsStandardHMMTransitions
)

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
        return self._loc.shape
    
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
    betas: Float[Array, 'num_states input_dim emission_dim'] | ParameterProperties
    scales: Float[Array, 'num_states emission_dim'] | ParameterProperties
    dfs: Float[Array, 'num_states'] | ParameterProperties

class THMMEmissions(HMMEmissions):
    """
    Implements t-distributed emissions for HMM.

    Args:
        num_states: number of hidden states K.
        emission_dim: dimension of the observation vector D.
        df_init: initial df.
    """

    def __init__(
            self, 
            num_states: int, 
            emission_dim: int, 
            input_dim: int,
            df_init: float = 10.0
        ):
        super().__init__()
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.df_init = df_init
    
    @property
    def emission_shape(self) -> tuple[int]:
        return (self.emission_dim,)
    
    def distribution(
            self, 
            params: ParamsTHMMEmissions,
            state: int, 
            inputs=None
    ) -> tfd.Distribution:
        if inputs is None:
            raise ValueError('THMM requires input regressors at every time step!')
        loc = inputs @ params.betas[state]
        return StudentTDistribution(
            loc=loc,
            scale=params.scales[state],
            df=params.dfs[state]
        )
    
    def log_prior(self, *args, **kwargs) -> Scalar:
        return 0.0
    
    def initialize(
            self,
            key: Array = jr.PRNGKey(0),
            method='prior',
            emission_betas: Float[Array, 'num_states input_dim emission_dim'] | None = None,
            emission_scales: Float[Array, 'num_states emission_dims'] | None = None,
            emission_dfs: Float[Array, 'num_states'] | None = None,
            emissions: Float[Array, 'num_timesteps emission_dim'] | None = None
    ) -> tuple[ParamsTHMMEmissions, ParamsTHMMEmissions]:
        
        default = lambda x, x0: x if x is not None else x0

        key, k1 = jr.split(key, 2)
        _betas = jr.normal(k1, (self.num_states, self.input_dim, self.emission_dim))
        _scales = jnp.ones((self.num_states, self.emission_dim))
        _dfs = jnp.full((self.num_states,), self.df_init)

        params = ParamsTHMMEmissions(
            betas=default(emission_betas, _betas),
            scales=default(emission_scales, _scales),
            dfs=default(emission_dfs, _dfs)
        )
        props = ParamsTHMMEmissions(
            betas=ParameterProperties(),
            scales=ParameterProperties(constrainer=tfb.Softplus()),
            dfs=ParameterProperties(constrainer=tfb.Softplus())
        )
        return params, props

class ParamsTHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsTHMMEmissions
 
class THMM(HMM):
    """
    Hidden Markov Model with t-distributed emissions.
    """
    
    def __init__(
        self,
        num_states: int, 
        emission_dim: int,
        input_dim: int,
        df_init: float = 10.0,
        initial_probs_concentration: float | Float[Array, 'num_states'] = 1.1,
        transition_matrix_concentration: float | Float[Array, 'num_states'] = 1.1
    ):
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(
            num_states, initial_probs_concentration=initial_probs_concentration
        )
        transition_component = StandardHMMTransitions(
            num_states, concentration=transition_matrix_concentration
        )
        emission_component = THMMEmissions(
            num_states, emission_dim, input_dim, df_init
        )
        self.input_dim = input_dim

        super().__init__(
            num_states, 
            initial_component, 
            transition_component, 
            emission_component
        )

    def initialize(
        self,
        key: Array = jr.PRNGKey(0),
        method: str = 'prior',
        initial_probs: Float[Array, 'num_states'] | None  = None,
        transition_matrix: Float[Array, 'num_states num_states'] | None = None,
        emission_betas: Float[Array, 'num_states input_dim emission_dim'] | None = None,
        emission_scales: Float[Array, 'num_states emission_dim'] | None = None,
        emission_dfs: Float[Array, 'num_states'] | None = None,
        emissions: Float[Array, 'num_timesteps emission_dim'] | None = None,
    ) -> tuple[ParamsTHMM, ParamsTHMM]:
        k1, k2, k3 = jr.split(key, 3)
        params, props = {}, {}
        params['initial'], props['initial'] = self.initial_component.initialize(
            k1, method=method, initial_probs=initial_probs
        )
        params['transitions'], props['transitions'] = self.transition_component.initialize(
            k2, method=method, transition_matrix=transition_matrix
        )
        params['emissions'], props['emissions'] = self.emission_component.initialize(
            k3, 
            method=method,
            emission_betas=emission_betas, 
            emission_scales=emission_scales,
            emission_dfs=emission_dfs,
            emissions=emissions
        )
        return ParamsTHMM(**params), ParamsTHMM(**props)
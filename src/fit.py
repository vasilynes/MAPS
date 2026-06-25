import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from .t_hmm import THMM  

emissions = jnp.load('data/emissions.npy')
inputs    = jnp.load('data/inputs.npy')

thmm = THMM(num_states=2, emission_dim=1, input_dim=2, df_init=4.0)
key = jr.PRNGKey(0)
params, props = thmm.initialize(key=key, method='prior', emissions=emissions)

print('Initial betas:\n', params.emissions.betas)
print('Initial scales:\n', params.emissions.scales)
print('Initial dfs:', params.emissions.dfs)

params, losses = thmm.fit_sgd(
    params, props, emissions, inputs=inputs, num_epochs=2000
)

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss (negative expected log joint)')
plt.title('SGD convergence')
plt.tight_layout()
plt.savefig('fit_losses.png', dpi=120)
plt.show()

print('\nFitted betas (intercept, vix_lag1):\n', params.emissions.betas)
print('Fitted scales:\n', params.emissions.scales)
print('Fitted dfs:', params.emissions.dfs)
print('Fitted transition matrix:\n', params.transitions.transition_matrix)
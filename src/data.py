import yfinance as yf
import polars as pl
import matplotlib.pyplot as plt
import polars as pl
import jax.numpy as jnp

def download_analyse():
    spy = yf.download('SPY', period='5y', interval='1d')['Close']
    vix = yf.download('^VIX', period='5y', interval='1d')['Close']

    df = pl.DataFrame({
        'date': spy.index.to_numpy().astype('datetime64[us]'),
        'spy': spy.to_numpy().ravel(),
        'vix': vix.to_numpy().ravel(),
    }).with_columns(
        (pl.col('spy').log() - pl.col('spy').log().shift(1)).alias('spy_logret')
    ).drop_nulls() 

    df.write_parquet('data/spy_vix_data.parquet')
    print(df.head())

    print(df.shape)
    print(df.schema)
    print(df.head())
    print(df.tail())

    print(df['date'].min(), '->', df['date'].max())

    print(df.null_count())
    print(
        df.select(
            pl.col('spy_logret').is_infinite().sum().alias('inf_count'),
            pl.col('spy_logret').is_nan().sum().alias('nan_count'),
        )
    )

    print(df.select(['spy', 'vix', 'spy_logret']).describe())

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(df['date'], df['spy'])
    axes[0].set_ylabel('SPY price')

    axes[1].plot(df['date'], df['vix'], color='orange')
    axes[1].set_ylabel('VIX')
    axes[1].set_xlabel('date')

    plt.tight_layout()
    plt.savefig('spy_vix_overview.png', dpi=120)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df['vix'], df['spy_logret'], alpha=0.3, s=8)
    ax.set_xlabel('VIX')
    ax.set_ylabel('SPY log-return')
    ax.set_title('SPY log-return vs VIX level')
    plt.tight_layout()
    plt.savefig('logret_vs_vix.png', dpi=120)
    plt.show()

def build_emissions_inputs():
    df = pl.read_parquet('data/spy_vix_data.parquet')

    # Lag VIX by 1 day: x_{t-1} explains y_t
    df = df.with_columns(
        pl.col('vix').shift(1).alias('vix_lag1')
    ).drop_nulls()  # drops the first row, which has no lag

    # emissions: SPY log-return, shape (T, 1)
    emissions = jnp.array(df['spy_logret'].to_numpy()).reshape(-1, 1)

    # inputs: [intercept, vix_lag1], shape (T, 2)
    intercept = jnp.ones(df.height)
    vix_lag1  = jnp.array(df['vix_lag1'].to_numpy())
    inputs = jnp.stack([intercept, vix_lag1], axis=1)

    print('emissions shape:', emissions.shape)
    print('inputs shape:', inputs.shape)
    print('emissions[:5]:\n', emissions[:5])
    print('inputs[:5]:\n', inputs[:5])

    # sanity: no NaNs/infs after lagging
    assert jnp.all(jnp.isfinite(emissions))
    assert jnp.all(jnp.isfinite(inputs))

    jnp.save('data/emissions.npy', emissions)
    jnp.save('data/inputs.npy', inputs)

build_emissions_inputs()
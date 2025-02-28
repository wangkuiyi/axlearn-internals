import jax
import jax.numpy as jnp
import flax.linen
import flax.training.train_state
import optax
import math
from axlearn.common import config as ax_config


class LinearDataset:
    def __init__(self, true_a: float, true_b: float, rng_seed: int = 42, batch_size: int = 32):
        self._rng = jax.random.PRNGKey(rng_seed)
        self._batch_size = batch_size
        self.batches = self._synthesize(2000, true_a, true_b)

    def _synthesize(self, num_samples: int, a: float, b: float) -> tuple[jax.Array, jax.Array]:
        self._rng, x_key, noise_key = jax.random.split(self._rng, 3)
        x = jax.random.uniform(x_key, shape=(num_samples, 1), minval=-10.0, maxval=10.0)
        noise = jax.random.normal(noise_key, shape=(num_samples, 1)) * 0.1
        y = a * x + b + noise
        return self._batch(x, y)

    def _batch(self, x: jax.Array, y: jax.Array) -> list[tuple[jax.Array, jax.Array]]:
        return [
            (x[i : i + self._batch_size], y[i : i + self._batch_size])
            for i in range(0, len(x), self._batch_size)
        ]


def test_config_dataset():
    cfg = ax_config.config_for_class(LinearDataset)
    ds = cfg.set(true_a=2.5, true_b=1.0, rng_seed=42).instantiate()
    assert len(ds.batches) == math.ceil(2000 / cfg.batch_size)
    assert ds.batches[0][0].shape == (cfg.batch_size, 1)


class LinearRegressor(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        a = self.param("a", flax.linen.initializers.zeros, (1, 1))  # Slope
        b = self.param("b", flax.linen.initializers.zeros, (1,))  # Intercept
        y = x @ a + b
        return y

    def random_init(self, rng):
        rng, init_rng = jax.random.split(rng)
        dummy_input = jnp.ones((1, 1))
        return rng, self.init(init_rng, dummy_input)["params"]


# This is required for axlearn.common.config to handle the case where a Flax
# module's parent module is of type flax.linen.module._Sentinel.
ax_config.register_validator(
    match_fn=lambda v: isinstance(v, flax.linen.module._Sentinel),
    validate_fn=lambda _: None,
)


def test_config_model():
    cfg = ax_config.config_for_class(LinearRegressor)
    model = cfg.instantiate()
    assert isinstance(model, LinearRegressor)


def test_config_optax():
    cfg = ax_config.config_for_function(optax.adam)
    tx = cfg.set(learning_rate=0.01).instantiate()
    print(type(tx))


class Trainer:
    def __init__(self, model_cfg, dataset_cfg, tx_cfg, rng_seed: int = 42):
        self._rng = jax.random.PRNGKey(rng_seed)
        self.model = model_cfg.instantiate()
        self.dataset = dataset_cfg.instantiate()
        self.tx = tx_cfg.instantiate()

    def train_step(self, state, batch):
        def loss_fn(params):
            x, y = batch
            pred_y = self.model.apply({"params": params}, x)
            return jnp.mean((pred_y - y) ** 2)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Use jit after defining the methods. static_argnums=(0) don't trace self.
    train_step = jax.jit(train_step, static_argnums=(0,))

    def train(self):
        self._rng, params = self.model.random_init(self._rng)
        state = flax.training.train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=self.tx
        )
        for epoch in range(50):
            total_loss = 0.0
            num_batches = 0
            for batch in self.dataset.batches:
                state, loss = self.train_step(state, batch)
                total_loss += loss
                num_batches += 1
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}")
        return state

    def print(self, state):
        learned_a = state.params["a"].item()
        learned_b = state.params["b"].item()
        print(f"Learned: y = {learned_a:.4f}x + {learned_b:.4f}")
        print("True:    y = 2.5x - 1.0")


def test_config_trainer():
    cfg = ax_config.config_for_class(Trainer).set(
        dataset_cfg=ax_config.config_for_class(LinearDataset).set(true_a=2.5, true_b=1.0),
        model_cfg=ax_config.config_for_class(LinearRegressor),
        tx_cfg=ax_config.config_for_function(optax.adam).set(learning_rate=0.01),
    )
    tr = cfg.instantiate()
    tr.print(tr.train())


from axlearn.experiments.trainer_config_utils import TrainerConfigFn


def named_trainer_configs() -> dict[str, TrainerConfigFn]:
    return {
        "linear_regression": lambda: ax_config.config_for_class(Trainer).set(
            dataset_cfg=ax_config.config_for_class(LinearDataset).set(a=2.5, b=1.0, n_samples=2000),
            model_cfg=ax_config.config_for_class(LinearRegressor),
            tx_cfg=ax_config.config_for_function(optax.adam).set(learning_rate=0.01),
        )
    }

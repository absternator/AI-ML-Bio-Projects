from contextlib import nullcontext
from functools import partial
import time

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
except ImportError as exc:  # pragma: no cover - dependency error path
    raise ImportError(
        "SFNP requires the optional JAX stack (`jax`, `flax`, `optax`). Install it before importing this model."
    ) from exc


SFNP_MODEL_CONFIG_KEYS = (
    "hidden_size",
    "d_state",
    "rff_freqs",
    "num_layers",
    "num_scan_dirs",
    "chunk_size",
)

SFNP_RUN_CONFIG_KEYS = (
    "lr",
    "batch_size",
    "num_steps",
    "distill_steps",
    "n_couplings",
    "ode_steps",
    "coupling_ode_steps",
    "eval_ode_steps",
    "num_draws",
)


def calculate_dmin_analytical(max_points: int, coord_dim: int, domain_size: float | None = None) -> float:
    """Match the LASI analytical d_min heuristic used during training."""
    if domain_size is None:
        domain_size = 10.0 if coord_dim == 1 else 1.0

    if coord_dim == 1:
        return max(domain_size / (2.0 * max_points), 1e-4)
    return max(0.5 * domain_size / (max_points ** (1.0 / coord_dim)), 1e-4)


def split_sfnp_config(
    config: dict,
    *,
    coord_dim: int,
    max_points: int,
    domain_size: float | None = None,
):
    """Split a combined SFNP config into model kwargs and run kwargs.

    `coord_dim` and `d_min` stay task-derived, following the old LASI setup.
    """
    model_cfg = {key: config[key] for key in SFNP_MODEL_CONFIG_KEYS if key in config}
    model_cfg.setdefault("num_scan_dirs", 4)
    model_cfg.setdefault("chunk_size", 64)
    model_cfg["coord_dim"] = coord_dim
    model_cfg["d_min"] = calculate_dmin_analytical(
        max_points,
        coord_dim,
        domain_size=domain_size,
    )

    run_cfg = {key: config[key] for key in SFNP_RUN_CONFIG_KEYS if key in config}
    return model_cfg, run_cfg


def _pad_seq_dim(x: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    if pad_size == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, pad_size)
    return jnp.pad(x, pad_width, mode="constant", constant_values=0.0)


def segsum(x: jnp.ndarray) -> jnp.ndarray:
    """Stable segment sum. (..., T) -> (..., T, T)."""
    length = x.shape[-1]
    x_cumsum = jnp.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((length, length), dtype=bool), k=0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum


def ssd_forward_linear(
    x: jnp.ndarray,
    dt: jnp.ndarray,
    A: jnp.ndarray,
    B_mat: jnp.ndarray,
    C_mat: jnp.ndarray,
    chunk_size: int,
    D: jnp.ndarray,
    dt_bias: jnp.ndarray,
    dt_min: float,
    dt_max: float,
    initial_states: jnp.ndarray | None = None,
    return_final_states: bool = False,
    mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """SSD forward with O(K) inter-chunk recurrence via `jax.lax.scan`."""
    _batch_size, seq_len, num_heads, _head_dim = x.shape
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    dt = jax.nn.softplus(dt + dt_bias)
    dt = jnp.clip(dt, dt_min, dt_max)

    x_padded = _pad_seq_dim(x, pad_size)
    dt_padded = _pad_seq_dim(dt, pad_size)
    B_padded = _pad_seq_dim(B_mat, pad_size)
    C_padded = _pad_seq_dim(C_mat, pad_size)

    if mask is not None:
        mask_padded = _pad_seq_dim(mask[:, :, None].astype(x_padded.dtype), pad_size)
        x_padded = x_padded * mask_padded[:, :, None, :]
        dt_padded = dt_padded * mask_padded

    D_residual = D.reshape(1, 1, num_heads, 1) * x_padded

    x_disc = x_padded * dt_padded[..., None]
    A_disc = A.astype(x_disc.dtype) * dt_padded

    def chunk_tensor(tensor):
        batch_size, chunked_len, *rest = tensor.shape
        return tensor.reshape(batch_size, chunked_len // chunk_size, chunk_size, *rest)

    x_blk = chunk_tensor(x_disc)
    A_blk = chunk_tensor(A_disc)
    B_blk = chunk_tensor(B_padded)
    C_blk = chunk_tensor(C_padded)

    A_blk2 = jnp.transpose(A_blk, (0, 3, 1, 2))
    A_cumsum = jnp.cumsum(A_blk2, axis=-1)

    L_mat = jnp.exp(segsum(A_blk2))
    y_diag = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blk, B_blk, L_mat, x_blk)

    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
    states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", B_blk, decay_states, x_blk)

    states_t = jnp.transpose(states, (1, 0, 2, 3, 4))
    A_end_t = jnp.transpose(A_cumsum[..., -1], (2, 0, 1))

    if initial_states is not None:
        init_carry = initial_states[:, 0, ...]
    else:
        init_carry = jnp.zeros_like(states_t[0])

    def scan_fn(carry, chunk_inputs):
        chunk_state, a_end = chunk_inputs
        decay = jnp.exp(a_end)[..., None, None]
        new_carry = carry * decay + chunk_state
        return new_carry, new_carry

    final_carry, scan_outputs = jax.lax.scan(scan_fn, init_carry, (states_t, A_end_t))

    arriving_states = jnp.concatenate([init_carry[None], scan_outputs[:-1]], axis=0)
    new_states = jnp.transpose(arriving_states, (1, 0, 2, 3, 4))

    state_decay_out = jnp.exp(A_cumsum)
    y_off = jnp.einsum("bclhn,bchpn,bhcl->bclhp", C_blk, new_states, state_decay_out)

    y = y_diag + y_off
    batch_size, chunks, chunk_len, heads, head_dim = y.shape
    y = y.reshape(batch_size, chunks * chunk_len, heads, head_dim)
    y = y + D_residual

    if pad_size > 0:
        y = y[:, :seq_len, :, :]

    if return_final_states:
        return y, final_carry
    return y, None


_USE_SMOOTH_NOISE = True


def set_smooth_noise(enabled: bool) -> None:
    """Module-level gate for smooth_noise(). Call before any jit compilation."""
    global _USE_SMOOTH_NOISE
    _USE_SMOOTH_NOISE = bool(enabled)


def get_smooth_noise() -> bool:
    return _USE_SMOOTH_NOISE


_USE_DISTANCE_FREE = False


def set_distance_free(enabled: bool) -> None:
    """Ablation gate: when True, DistanceAwareMambaBlock zeros the distance
    feature fed into dt_proj. Call before any jit compilation."""
    global _USE_DISTANCE_FREE
    _USE_DISTANCE_FREE = bool(enabled)


def get_distance_free() -> bool:
    return _USE_DISTANCE_FREE


_USE_SINGLE_STATE = False


def set_single_state(enabled: bool) -> None:
    """Ablation gate: when True, SFNP overwrites the query portion of the
    unified hidden states with a broadcast of the last valid context position,
    forcing all context info through one fixed-dim vector."""
    global _USE_SINGLE_STATE
    _USE_SINGLE_STATE = bool(enabled)


def get_single_state() -> bool:
    return _USE_SINGLE_STATE


_USE_RANDOM_ORDER = False
_USE_HILBERT_ROT_JITTER = False


def set_random_order(enabled: bool) -> None:
    """Ablation gate: when True, replace Hilbert/coordinate sort with random
    permutations seeded per-example from a hash of coords. Removes all
    spatial locality from the scan order."""
    global _USE_RANDOM_ORDER
    _USE_RANDOM_ORDER = bool(enabled)


def get_random_order() -> bool:
    return _USE_RANDOM_ORDER


def set_hilbert_rot_jitter(enabled: bool) -> None:
    """Ablation gate: when True, apply a per-example random rotation around
    the unit-box centre to coords *before* computing Hilbert keys. Preserves
    spatial locality (rotation is an isometry) while randomising the specific
    1D ordering seen by the scan, so the model cannot memorise one canonical
    order. The RFF/value inputs continue to use the unrotated coords."""
    global _USE_HILBERT_ROT_JITTER
    _USE_HILBERT_ROT_JITTER = bool(enabled)


def get_hilbert_rot_jitter() -> bool:
    return _USE_HILBERT_ROT_JITTER


_USE_LEX_ORDER = False


def set_lex_order(enabled: bool) -> None:
    """Ablation gate: when True, replace Hilbert sort with lexicographic
    sort (argsort by x, then y). Preserves axis-aligned locality but loses
    the fractal locality of Hilbert curves."""
    global _USE_LEX_ORDER
    _USE_LEX_ORDER = bool(enabled)


def get_lex_order() -> bool:
    return _USE_LEX_ORDER


_USE_RAW_COORDS = False


def set_raw_coords(enabled: bool) -> None:
    """Ablation gate: when True, bypass Random Fourier Features and feed
    zero-padded raw coordinates into ctx_proj/qry_proj."""
    global _USE_RAW_COORDS
    _USE_RAW_COORDS = bool(enabled)


def get_raw_coords() -> bool:
    return _USE_RAW_COORDS


_USE_DISTANCE_CONSTANT = False


def set_distance_constant(enabled: bool) -> None:
    """Ablation gate: when True, replace the per-point distance feature with
    a single constant (per-batch mean), isolating whether the model uses
    point-specific distances or just a constant-magnitude signal."""
    global _USE_DISTANCE_CONSTANT
    _USE_DISTANCE_CONSTANT = bool(enabled)


def get_distance_constant() -> bool:
    return _USE_DISTANCE_CONSTANT


def smooth_noise(f0, coords, mask=None, bandwidth=None, max_points=2048):
    """Spatially smooth independent noise via Gaussian kernel.

    If the module-level flag is disabled (set_smooth_noise(False)), returns raw f0.
    """
    if not _USE_SMOOTH_NOISE:
        return f0

    num_points = coords.shape[1]

    if num_points > max_points:
        return f0

    if bandwidth is None:
        bandwidth = 1.0 / jnp.sqrt(jnp.maximum(num_points, 1.0).astype(jnp.float32))

    diff = coords[:, :, None, :] - coords[:, None, :, :]
    sq_dist = jnp.sum(diff**2, axis=-1)
    kernel = jnp.exp(-sq_dist / (2 * bandwidth**2 + 1e-8))

    if mask is not None:
        kernel = kernel * mask[:, None, :]
        kernel = kernel * mask[:, :, None]

    kernel = kernel / (jnp.sum(kernel, axis=-1, keepdims=True) + 1e-8)
    f0_smooth = jnp.matmul(kernel, f0)

    if mask is not None:
        valid_count = jnp.sum(mask, axis=1, keepdims=True)[:, :, None]
        mean = jnp.sum(f0_smooth * mask[:, :, None], axis=1, keepdims=True) / jnp.maximum(
            valid_count, 1.0
        )
        var = jnp.sum((f0_smooth - mean) ** 2 * mask[:, :, None], axis=1, keepdims=True) / jnp.maximum(
            valid_count, 1.0
        )
        f0_smooth = (f0_smooth - mean) / (jnp.sqrt(var) + 1e-6)
        f0_smooth = f0_smooth * mask[:, :, None]
    else:
        std = jnp.std(f0_smooth, axis=1, keepdims=True)
        f0_smooth = f0_smooth / (std + 1e-6)

    return f0_smooth


def _xy_to_hilbert(x, y, order=16):
    """Convert [0, 1] coordinates to a Hilbert curve index."""
    n = 2**order
    ix = jnp.clip((x * n).astype(jnp.int32), 0, n - 1)
    iy = jnp.clip((y * n).astype(jnp.int32), 0, n - 1)

    def _xy2d(xv, yv):
        dist = jnp.int32(0)

        def body_fn(carry, _):
            d_val, x_c, y_c, scale = carry
            rx = jnp.where((x_c & scale) > 0, jnp.int32(1), jnp.int32(0))
            ry = jnp.where((y_c & scale) > 0, jnp.int32(1), jnp.int32(0))
            d_val = d_val + scale * scale * ((3 * rx) ^ ry)
            swap = ry == 0
            flip = swap & (rx == 1)
            x_c = jnp.where(flip, scale - 1 - x_c, x_c)
            y_c = jnp.where(flip, scale - 1 - y_c, y_c)
            x_c, y_c = jnp.where(swap, y_c, x_c), jnp.where(swap, x_c, y_c)
            return (d_val, x_c, y_c, scale // 2), None

        scale_values = n // 2
        (dist, _, _, _), _ = jax.lax.scan(
            body_fn,
            (dist, xv, yv, jnp.int32(scale_values)),
            None,
            length=order,
        )
        return dist

    return jax.vmap(_xy2d)(ix, iy)


class FrozenRFF(nnx.Module):
    """Random Fourier features with a frozen frequency matrix."""

    def __init__(
        self,
        in_features: int,
        num_freqs: int,
        d_min: float,
        c_scale: float = 1.0,
        seed: int = 42,
    ):
        key = jax.random.PRNGKey(seed)
        sigma = c_scale / (d_min + 1e-6)
        sigma = float(jnp.clip(sigma, 0.0, 200.0))
        self.B = jax.random.normal(key, (num_freqs, in_features)) * sigma
        self._d_min_train = d_min

    def rescale_for_density(self, d_min_eval: float):
        ratio = self._d_min_train / (d_min_eval + 1e-6)
        ratio = float(jnp.clip(ratio, 0.01, 100.0))
        self.B = self.B * ratio
        self._d_min_train = d_min_eval

    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        if _USE_RAW_COORDS:
            out_dim = self.B.shape[0] * 2
            pad = out_dim - coords.shape[-1]
            return jnp.pad(coords, [(0, 0)] * (coords.ndim - 1) + [(0, pad)], mode="constant")
        proj = 2 * jnp.pi * jnp.dot(coords, self.B.T)
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


def get_timestep_embedding(timesteps: jnp.ndarray, embedding_dim: int = 256) -> jnp.ndarray:
    """Sinusoidal timestep embedding for continuous t in [0, 1]."""
    t_scaled = timesteps * 1000.0
    half_dim = embedding_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = t_scaled[:, None] * emb[None, :]
    return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)


class ZeroLinear(nnx.Module):
    """Linear layer initialized to zero weights and biases."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.kernel = nnx.Param(jnp.zeros((in_features, out_features)))
        self.bias = nnx.Param(jnp.zeros((out_features,)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.kernel + self.bias


class DistanceAwareMambaBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        num_heads: int = 4,
        chunk_size: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size

        self.in_proj = nnx.Linear(d_model, d_model * 2, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dt_proj = nnx.Linear(d_model + 1, num_heads, rngs=rngs)
        self.bc_proj = nnx.Linear(d_model, d_state * 2, rngs=rngs)

        key = rngs.params()
        self.A_log = nnx.Param(
            jnp.log(jax.random.uniform(key, (num_heads,), minval=1.0, maxval=16.0))
        )
        self.D = nnx.Param(jnp.ones(num_heads))
        self.dt_bias = nnx.Param(jnp.zeros(num_heads))

    def __call__(self, x: jnp.ndarray, dists: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_in, z = jnp.split(xz, 2, axis=-1)
        x_in = jax.nn.silu(x_in)

        if _USE_DISTANCE_FREE:
            dt = self.dt_proj(jnp.concatenate([x_in, jnp.zeros_like(dists)], axis=-1))
        elif _USE_DISTANCE_CONSTANT:
            dist_mean = jnp.mean(dists)
            dt = self.dt_proj(jnp.concatenate([x_in, jnp.full_like(dists, dist_mean)], axis=-1))
        else:
            dt = self.dt_proj(jnp.concatenate([x_in, dists], axis=-1))

        bc = self.bc_proj(x_in)
        B_mat, C_mat = jnp.split(bc, 2, axis=-1)
        B_mat = jnp.broadcast_to(
            B_mat[:, :, None, :],
            (batch_size, seq_len, self.num_heads, self.d_state),
        )
        C_mat = jnp.broadcast_to(
            C_mat[:, :, None, :],
            (batch_size, seq_len, self.num_heads, self.d_state),
        )

        x_heads = x_in.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        A = -jnp.exp(self.A_log[:].astype(jnp.float32))
        y, _ = ssd_forward_linear(
            x=x_heads,
            dt=dt,
            A=A,
            B_mat=B_mat,
            C_mat=C_mat,
            chunk_size=self.chunk_size,
            D=self.D[:],
            dt_bias=self.dt_bias[:],
            dt_min=1e-3,
            dt_max=10.0,
            mask=mask,
        )

        y = y.reshape(batch_size, seq_len, self.d_model)
        y = y * jax.nn.silu(z)
        return self.out_proj(y)


class SFNP(nnx.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        d_state: int = 64,
        rff_freqs: int = 64,
        d_min: float = 0.01,
        coord_dim: int = 1,
        num_scan_dirs: int = 4,
        chunk_size: int = 64,
        num_layers: int = 4,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.coord_dim = coord_dim
        self.num_scan_dirs = num_scan_dirs
        self.num_layers = num_layers
        self.mesh = mesh

        self.rff = FrozenRFF(in_features=coord_dim, num_freqs=rff_freqs, d_min=d_min)
        rff_dim = rff_freqs * 2

        self.ctx_proj = nnx.Linear(rff_dim + 1, hidden_size, rngs=rngs)
        self.qry_proj = nnx.Linear(rff_dim + 1, hidden_size, rngs=rngs)

        self.time_mlp = nnx.Sequential(
            nnx.Linear(256, hidden_size, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
        )

        mamba_layers = []
        layer_norms = []
        adaln_projs = []
        for _ in range(num_layers):
            mamba_layers.append(
                DistanceAwareMambaBlock(
                    d_model=hidden_size,
                    d_state=d_state,
                    num_heads=4,
                    chunk_size=chunk_size,
                    rngs=rngs,
                )
            )
            layer_norms.append(
                nnx.LayerNorm(hidden_size, use_bias=False, use_scale=False, rngs=rngs)
            )
            adaln_projs.append(ZeroLinear(hidden_size, 3 * hidden_size, rngs=rngs))
        self.mamba_layers = nnx.List(mamba_layers)
        self.layer_norms = nnx.List(layer_norms)
        self.adaln_projs = nnx.List(adaln_projs)

        self.dir_merge = nnx.Linear(num_scan_dirs * hidden_size, hidden_size, rngs=rngs)
        self.final_norm = nnx.LayerNorm(hidden_size, use_bias=False, use_scale=False, rngs=rngs)
        self.final_adaln = ZeroLinear(hidden_size, 2 * hidden_size, rngs=rngs)
        self.flow_out = ZeroLinear(hidden_size, 1, rngs=rngs)

    def _get_sort_indices(self, coords):
        n = coords.shape[0]

        if _USE_RANDOM_ORDER:
            seed = jnp.abs(
                jnp.sum((coords * 1e5).astype(jnp.int32))
            ).astype(jnp.uint32)
            key = jax.random.key(seed)
            keys = jax.random.split(key, max(self.num_scan_dirs, 1))
            dirs = [
                jax.random.permutation(keys[i], n).astype(jnp.int32)
                for i in range(self.num_scan_dirs)
            ]
            return tuple(dirs)

        if _USE_LEX_ORDER:
            if self.coord_dim == 1:
                forward = jnp.argsort(coords[:, 0]).astype(jnp.int32)
                reverse = forward[::-1]
                if self.num_scan_dirs <= 2:
                    return (forward, reverse)[: self.num_scan_dirs]
                dirs = [forward, reverse]
                while len(dirs) < self.num_scan_dirs:
                    dirs.extend([forward, reverse])
                return tuple(dirs[: self.num_scan_dirs])
            forward = jnp.lexsort(
                jnp.stack([coords[:, 1], coords[:, 0]], axis=0)
            ).astype(jnp.int32)
            reverse = forward[::-1]
            if self.num_scan_dirs == 2:
                return (forward, reverse)
            transpose_forward = jnp.lexsort(
                jnp.stack([coords[:, 0], coords[:, 1]], axis=0)
            ).astype(jnp.int32)
            transpose_reverse = transpose_forward[::-1]
            dirs = [forward, reverse, transpose_forward, transpose_reverse]
            if self.num_scan_dirs > 4:
                while len(dirs) < self.num_scan_dirs:
                    dirs.extend(dirs[:4])
            return tuple(dirs[: self.num_scan_dirs])

        if self.coord_dim == 1:
            forward = jnp.argsort(coords[:, 0]).astype(jnp.int32)
            reverse = forward[::-1]
            if self.num_scan_dirs <= 2:
                return (forward, reverse)[: self.num_scan_dirs]
            dirs = [forward, reverse]
            while len(dirs) < self.num_scan_dirs:
                dirs.extend([forward, reverse])
            return tuple(dirs[: self.num_scan_dirs])

        if _USE_HILBERT_ROT_JITTER:
            seed = jnp.abs(jnp.sum((coords * 1e5).astype(jnp.int32))).astype(jnp.uint32)
            theta = jax.random.uniform(jax.random.key(seed), shape=(), minval=0.0, maxval=2.0 * jnp.pi)
            cos_t = jnp.cos(theta)
            sin_t = jnp.sin(theta)
            cx = jnp.mean(coords[:, 0])
            cy = jnp.mean(coords[:, 1])
            dx = coords[:, 0] - cx
            dy = coords[:, 1] - cy
            rot_x = cx + cos_t * dx - sin_t * dy
            rot_y = cy + sin_t * dx + cos_t * dy
            # _xy_to_hilbert expects [0, 1] coords; rescale to rotated bbox so we
            # don't lose the ordering when a rotation sends corners outside.
            xmin = jnp.min(rot_x); xmax = jnp.max(rot_x)
            ymin = jnp.min(rot_y); ymax = jnp.max(rot_y)
            sort_x = (rot_x - xmin) / (xmax - xmin + 1e-8)
            sort_y = (rot_y - ymin) / (ymax - ymin + 1e-8)
        else:
            sort_x = coords[:, 0]
            sort_y = coords[:, 1]

        hilbert_keys = _xy_to_hilbert(sort_x, sort_y)
        forward = jnp.argsort(hilbert_keys).astype(jnp.int32)
        reverse = forward[::-1]
        if self.num_scan_dirs == 2:
            return (forward, reverse)

        hilbert_keys_t = _xy_to_hilbert(sort_y, sort_x)
        transpose_forward = jnp.argsort(hilbert_keys_t).astype(jnp.int32)
        transpose_reverse = transpose_forward[::-1]
        dirs = [forward, reverse, transpose_forward, transpose_reverse]
        if self.num_scan_dirs > 4:
            while len(dirs) < self.num_scan_dirs:
                dirs.extend(dirs[:4])
        return tuple(dirs[: self.num_scan_dirs])

    def precompute_routing(self, x_ctx, x_qry, mask=None):
        coords = jnp.concatenate([x_ctx, x_qry], axis=1)
        num_dirs = self.num_scan_dirs

        if mask is not None:
            num_valid = jnp.sum(mask, axis=1, keepdims=True).astype(jnp.float32)
            coords_for_sort = jnp.where(mask[:, :, None], coords, jnp.finfo(jnp.float32).max)
        else:
            num_valid = jnp.full((coords.shape[0], 1), coords.shape[1], dtype=jnp.float32)
            coords_for_sort = coords

        all_idx = jax.vmap(self._get_sort_indices)(coords_for_sort)
        all_inv_idx = tuple(
            jax.vmap(lambda idx: jnp.argsort(idx).astype(jnp.int32))(all_idx[k])
            for k in range(num_dirs)
        )

        def compute_dists(crd, idx, nv):
            sorted_coords = crd[idx]
            diffs = sorted_coords[1:] - sorted_coords[:-1]
            dist = jnp.sqrt(jnp.sum(diffs**2, axis=-1, keepdims=True) + 1e-8)
            dist = jnp.pad(dist, ((1, 0), (0, 0)))
            dist = jnp.minimum(dist, 10.0)
            dist = dist * jnp.sqrt(jnp.maximum(nv, 1.0))
            return dist

        sorted_dists = []
        for k in range(num_dirs):
            sorted_dists.append(jax.vmap(compute_dists)(coords_for_sort, all_idx[k], num_valid))

        stacked_dists = jnp.concatenate(sorted_dists, axis=0)
        return all_idx, all_inv_idx, stacked_dists

    def __call__(self, x_ctx, y_ctx, x_qry, f_qry_t, t, routing_cache=None, mask=None):
        num_dirs = self.num_scan_dirs

        rff_ctx = self.rff(x_ctx)
        rff_qry = self.rff(x_qry)

        tok_ctx = self.ctx_proj(jnp.concatenate([rff_ctx, y_ctx], axis=-1))
        tok_qry = self.qry_proj(jnp.concatenate([rff_qry, f_qry_t], axis=-1))

        cond = self.time_mlp(get_timestep_embedding(t))
        tokens = jnp.concatenate([tok_ctx, tok_qry], axis=1)
        num_context = x_ctx.shape[1]

        if routing_cache is not None:
            all_idx, all_inv_idx, stacked_dists = routing_cache
        else:
            all_idx, all_inv_idx, stacked_dists = self.precompute_routing(x_ctx, x_qry, mask=mask)

        stacked_idx = jnp.concatenate([all_idx[k] for k in range(num_dirs)], axis=0)
        stacked_tokens_src = jnp.tile(tokens, (num_dirs, 1, 1))
        stacked_tokens = jax.vmap(lambda tok, idx: tok[idx])(stacked_tokens_src, stacked_idx)

        stacked_mask = None
        if mask is not None:
            stacked_mask_src = jnp.tile(mask, (num_dirs, 1))
            stacked_mask = jax.vmap(lambda m, idx: m[idx])(stacked_mask_src, stacked_idx)

        stacked_cond = jnp.tile(cond, (num_dirs, 1))

        if self.mesh is not None:
            sh3 = NamedSharding(self.mesh, P("dev", None, None))
            sh2 = NamedSharding(self.mesh, P("dev", None))
            stacked_tokens = jax.lax.with_sharding_constraint(stacked_tokens, sh3)
            stacked_dists = jax.lax.with_sharding_constraint(stacked_dists, sh3)
            stacked_cond = jax.lax.with_sharding_constraint(stacked_cond, sh2)

        hidden = stacked_tokens
        for norm, layer, adaln in zip(self.layer_norms, self.mamba_layers, self.adaln_projs):
            mod = jax.nn.silu(stacked_cond)
            shift, scale, gate = jnp.split(adaln(mod), 3, axis=-1)
            hidden_norm = norm(hidden) * (1 + scale[:, None, :]) + shift[:, None, :]
            hidden = hidden + gate[:, None, :] * layer(hidden_norm, stacked_dists, mask=stacked_mask)

        chunks = jnp.split(hidden, num_dirs, axis=0)
        h_dirs = [jax.vmap(lambda h, idx: h[idx])(chunks[k], all_inv_idx[k]) for k in range(num_dirs)]
        h_unified = self.dir_merge(jnp.concatenate(h_dirs, axis=-1))

        if mask is not None:
            h_unified = h_unified * mask[:, :, None]

        if _USE_SINGLE_STATE:
            h_ctx = h_unified[:, :num_context, :]
            if mask is not None:
                ctx_mask = mask[:, :num_context]
                ctx_counts = jnp.sum(ctx_mask.astype(jnp.int32), axis=1)
                last_idx = jnp.maximum(ctx_counts - 1, 0)
            else:
                last_idx = jnp.full((h_ctx.shape[0],), num_context - 1, dtype=jnp.int32)
            last_ctx_state = jax.vmap(lambda h, i: h[i])(h_ctx, last_idx)
            num_query = h_unified.shape[1] - num_context
            broadcast_state = jnp.broadcast_to(
                last_ctx_state[:, None, :],
                (h_unified.shape[0], num_query, h_unified.shape[2]),
            )
            h_unified = jnp.concatenate([h_ctx, broadcast_state], axis=1)
            if mask is not None:
                h_unified = h_unified * mask[:, :, None]

        h_query = h_unified[:, num_context:, :]
        final_mod = jax.nn.silu(cond)
        final_shift, final_scale = jnp.split(self.final_adaln(final_mod), 2, axis=-1)
        h_query = self.final_norm(h_query) * (1 + final_scale[:, None, :]) + final_shift[:, None, :]

        return self.flow_out(h_query)


def _get_jit_fns(model):
    if not hasattr(model, "_jit_cache"):
        graphdef, _ = nnx.split(model)

        @partial(jax.jit, static_argnums=(7,))
        def _euler(state, x_ctx, y_ctx, x_qry, eval_keys, ctx_mask, qry_mask, ode_steps):
            num_draws = eval_keys.shape[0]
            merged = nnx.merge(graphdef, state)
            x_ctx_b = jnp.tile(x_ctx, (num_draws, 1, 1))
            y_ctx_b = jnp.tile(y_ctx, (num_draws, 1, 1))
            x_qry_b = jnp.tile(x_qry, (num_draws, 1, 1))
            ctx_mask_b = jnp.tile(ctx_mask, (num_draws, 1))
            qry_mask_b = jnp.tile(qry_mask, (num_draws, 1))
            full_mask = jnp.concatenate([ctx_mask_b, qry_mask_b], axis=1)
            routing_cache = merged.precompute_routing(x_ctx_b, x_qry_b, mask=full_mask)
            f_0 = jax.vmap(
                lambda key: jax.random.normal(key, shape=(x_qry.shape[1], 1))
            )(eval_keys)
            f_0 = smooth_noise(f_0, x_qry_b, mask=qry_mask_b)
            dt = 1.0 / ode_steps
            steps = jnp.arange(ode_steps, dtype=jnp.float32)

            def step_fn(f_t, step_idx):
                t_val = step_idx * dt
                t_batch = jnp.full((num_draws,), jnp.maximum(t_val, 1e-4))
                velocity = merged(
                    x_ctx_b,
                    y_ctx_b,
                    x_qry_b,
                    f_t,
                    t_batch,
                    routing_cache=routing_cache,
                    mask=full_mask,
                )
                return f_t + dt * velocity, None

            f_1, _ = jax.lax.scan(step_fn, f_0, steps)
            return f_1

        @partial(jax.jit, static_argnums=(7,))
        def _heun(state, x_ctx, y_ctx, x_qry, eval_keys, ctx_mask, qry_mask, ode_steps):
            num_draws = eval_keys.shape[0]
            merged = nnx.merge(graphdef, state)
            x_ctx_b = jnp.tile(x_ctx, (num_draws, 1, 1))
            y_ctx_b = jnp.tile(y_ctx, (num_draws, 1, 1))
            x_qry_b = jnp.tile(x_qry, (num_draws, 1, 1))
            ctx_mask_b = jnp.tile(ctx_mask, (num_draws, 1))
            qry_mask_b = jnp.tile(qry_mask, (num_draws, 1))
            full_mask = jnp.concatenate([ctx_mask_b, qry_mask_b], axis=1)
            routing_cache = merged.precompute_routing(x_ctx_b, x_qry_b, mask=full_mask)
            f_0 = jax.vmap(
                lambda key: jax.random.normal(key, shape=(x_qry.shape[1], 1))
            )(eval_keys)
            f_0 = smooth_noise(f_0, x_qry_b, mask=qry_mask_b)
            dt = 1.0 / ode_steps
            steps = jnp.arange(ode_steps, dtype=jnp.float32)

            def step_fn(f, step_idx):
                t1 = step_idx * dt
                t2 = (step_idx + 1.0) * dt
                t1_b = jnp.full((num_draws,), jnp.maximum(t1, 1e-4))
                t2_b = jnp.full((num_draws,), jnp.minimum(t2, 1.0 - 1e-4))
                v1 = merged(
                    x_ctx_b,
                    y_ctx_b,
                    x_qry_b,
                    f,
                    t1_b,
                    routing_cache=routing_cache,
                    mask=full_mask,
                )
                f_tmp = jnp.clip(f + dt * v1, -6.0, 6.0)
                v2 = merged(
                    x_ctx_b,
                    y_ctx_b,
                    x_qry_b,
                    f_tmp,
                    t2_b,
                    routing_cache=routing_cache,
                    mask=full_mask,
                )
                return jnp.clip(f + (dt / 2.0) * (v1 + v2), -6.0, 6.0), None

            f_1, _ = jax.lax.scan(step_fn, f_0, steps)
            return f_1

        @partial(jax.jit, static_argnums=(7,))
        def _single_heun(state, x_ctx, y_ctx, x_qry, key, ctx_mask, qry_mask, ode_steps):
            merged = nnx.merge(graphdef, state)
            ctx_mask_b = ctx_mask[None] if ctx_mask.ndim == 1 else ctx_mask
            qry_mask_b = qry_mask[None] if qry_mask.ndim == 1 else qry_mask
            full_mask = jnp.concatenate([ctx_mask_b, qry_mask_b], axis=1)
            routing_cache = merged.precompute_routing(x_ctx, x_qry, mask=full_mask)
            f_t = jax.random.normal(key, shape=(x_qry.shape[1], 1))[None]
            f_t = smooth_noise(f_t, x_qry, mask=qry_mask_b)
            dt = 1.0 / ode_steps
            steps = jnp.arange(ode_steps, dtype=jnp.float32)

            def step_fn(f, step_idx):
                t1 = step_idx * dt
                t2 = (step_idx + 1.0) * dt
                t1_b = jnp.full((1,), jnp.maximum(t1, 1e-4))
                t2_b = jnp.full((1,), jnp.minimum(t2, 1.0 - 1e-4))
                v1 = merged(x_ctx, y_ctx, x_qry, f, t1_b, routing_cache=routing_cache, mask=full_mask)
                f_tmp = jnp.clip(f + dt * v1, -6.0, 6.0)
                v2 = merged(x_ctx, y_ctx, x_qry, f_tmp, t2_b, routing_cache=routing_cache, mask=full_mask)
                return jnp.clip(f + (dt / 2.0) * (v1 + v2), -6.0, 6.0), None

            f_1, _ = jax.lax.scan(step_fn, f_t, steps)
            return f_1

        @partial(jax.jit, static_argnums=(7,))
        def _single_euler(state, x_ctx, y_ctx, x_qry, key, ctx_mask, qry_mask, ode_steps):
            merged = nnx.merge(graphdef, state)
            ctx_mask_b = ctx_mask[None] if ctx_mask.ndim == 1 else ctx_mask
            qry_mask_b = qry_mask[None] if qry_mask.ndim == 1 else qry_mask
            full_mask = jnp.concatenate([ctx_mask_b, qry_mask_b], axis=1)
            routing_cache = merged.precompute_routing(x_ctx, x_qry, mask=full_mask)
            f_t = jax.random.normal(key, shape=(x_qry.shape[1], 1))[None]
            f_t = smooth_noise(f_t, x_qry, mask=qry_mask_b)
            dt = 1.0 / ode_steps
            steps = jnp.arange(ode_steps, dtype=jnp.float32)

            def step_fn(f, step_idx):
                t_val = step_idx * dt
                t_batch = jnp.full((1,), jnp.maximum(t_val, 1e-4))
                velocity = merged(x_ctx, y_ctx, x_qry, f, t_batch, routing_cache=routing_cache, mask=full_mask)
                return f + dt * velocity, None

            f_1, _ = jax.lax.scan(step_fn, f_t, steps)
            return f_1

        @partial(jax.jit, static_argnums=(7,))
        def _teacher_traj(state, x_ctx, y_ctx, x_qry, keys, ctx_mask, qry_mask, ode_steps):
            batch_size, num_query, _ = x_qry.shape
            merged = nnx.merge(graphdef, state)
            full_mask = jnp.concatenate([ctx_mask, qry_mask], axis=1)
            routing_cache = merged.precompute_routing(x_ctx, x_qry, mask=full_mask)
            f_0 = jax.vmap(lambda key: jax.random.normal(key, (num_query, 1)))(keys)
            f_0 = smooth_noise(f_0, x_qry, mask=qry_mask)
            dt = 1.0 / ode_steps
            steps = jnp.arange(ode_steps, dtype=jnp.float32)

            def step_fn(f, step_idx):
                t1 = step_idx * dt
                t2 = (step_idx + 1.0) * dt
                t1_b = jnp.full((batch_size,), jnp.maximum(t1, 1e-4))
                t2_b = jnp.full((batch_size,), jnp.minimum(t2, 1.0 - 1e-4))
                v1 = merged(x_ctx, y_ctx, x_qry, f, t1_b, routing_cache=routing_cache, mask=full_mask)
                f_tmp = f + dt * v1
                v2 = merged(x_ctx, y_ctx, x_qry, f_tmp, t2_b, routing_cache=routing_cache, mask=full_mask)
                return f + (dt / 2.0) * (v1 + v2), None

            f_1, _ = jax.lax.scan(step_fn, f_0, steps)
            return f_0, f_1

        model._jit_cache = (_euler, _heun, _single_heun, _single_euler, _teacher_traj)

    _, state = nnx.split(model)
    return model._jit_cache, state


def sample_euler(model, x_ctx, y_ctx, x_qry, eval_keys, ode_steps=50, ctx_mask=None, qry_mask=None):
    if ctx_mask is None:
        ctx_mask = jnp.ones(x_ctx.shape[:2], dtype=bool)
    if qry_mask is None:
        qry_mask = jnp.ones(x_qry.shape[:2], dtype=bool)
    (euler_fn, _, _, _, _), state = _get_jit_fns(model)
    return euler_fn(state, x_ctx, y_ctx, x_qry, eval_keys, ctx_mask, qry_mask, ode_steps)


def sample_heun(model, x_ctx, y_ctx, x_qry, eval_keys, ode_steps=15, ctx_mask=None, qry_mask=None):
    if ctx_mask is None:
        ctx_mask = jnp.ones(x_ctx.shape[:2], dtype=bool)
    if qry_mask is None:
        qry_mask = jnp.ones(x_qry.shape[:2], dtype=bool)
    (_, heun_fn, _, _, _), state = _get_jit_fns(model)
    return heun_fn(state, x_ctx, y_ctx, x_qry, eval_keys, ctx_mask, qry_mask, ode_steps)


def sample_single_heun(model, x_ctx, y_ctx, x_qry, key, ode_steps=15, ctx_mask=None, qry_mask=None):
    if ctx_mask is None:
        ctx_mask = jnp.ones(x_ctx.shape[:2], dtype=bool)
    if qry_mask is None:
        qry_mask = jnp.ones(x_qry.shape[:2], dtype=bool)
    (_, _, single_heun_fn, _, _), state = _get_jit_fns(model)
    return single_heun_fn(state, x_ctx, y_ctx, x_qry, key, ctx_mask, qry_mask, ode_steps)


def sample_single_euler(model, x_ctx, y_ctx, x_qry, key, ode_steps=1, ctx_mask=None, qry_mask=None):
    if ctx_mask is None:
        ctx_mask = jnp.ones(x_ctx.shape[:2], dtype=bool)
    if qry_mask is None:
        qry_mask = jnp.ones(x_qry.shape[:2], dtype=bool)
    (_, _, _, single_euler_fn, _), state = _get_jit_fns(model)
    return single_euler_fn(state, x_ctx, y_ctx, x_qry, key, ctx_mask, qry_mask, ode_steps)


def generate_teacher_trajectories(
    model,
    x_ctx,
    y_ctx,
    x_qry,
    keys,
    ode_steps=15,
    ctx_mask=None,
    qry_mask=None,
):
    if ctx_mask is None:
        ctx_mask = jnp.ones(x_ctx.shape[:2], dtype=bool)
    if qry_mask is None:
        qry_mask = jnp.ones(x_qry.shape[:2], dtype=bool)
    (_, _, _, _, teacher_fn), state = _get_jit_fns(model)
    return teacher_fn(state, x_ctx, y_ctx, x_qry, keys, ctx_mask, qry_mask, ode_steps)


def compute_nll_sfnp(model, x_ctx, y_ctx, x_qry, f_qry, key, ode_steps=50, ctx_mask=None, qry_mask=None):
    """Compute per-point NLL via reverse ODE + Hutchinson trace estimator."""
    if ctx_mask is None:
        ctx_mask = jnp.ones(x_ctx.shape[:2], dtype=bool)
    if qry_mask is None:
        qry_mask = jnp.ones(x_qry.shape[:2], dtype=bool)

    if not hasattr(model, "_nll_jit_cache"):
        graphdef, _ = nnx.split(model)

        @partial(jax.jit, static_argnums=(8,))
        def _nll(state, x_ctx, y_ctx, x_qry, f_1, key, ctx_mask, qry_mask, ode_steps):
            merged = nnx.merge(graphdef, state)
            batch_size = f_1.shape[0]

            full_mask = jnp.concatenate([ctx_mask, qry_mask], axis=1)
            routing_cache = merged.precompute_routing(x_ctx, x_qry, mask=full_mask)

            epsilon = 2.0 * jax.random.bernoulli(key, 0.5, shape=f_1.shape).astype(jnp.float32) - 1.0
            dt = jnp.float32(1.0 / ode_steps)
            steps = jnp.arange(ode_steps, dtype=jnp.float32)

            def step_fn(carry, step_idx):
                f_t, log_det = carry
                t_val = jnp.float32(1.0) - step_idx * dt
                t_batch = jnp.full((batch_size,), jnp.clip(t_val, 1e-4, 1.0 - 1e-4))

                def v_fn(f):
                    return merged(
                        x_ctx,
                        y_ctx,
                        x_qry,
                        f,
                        t_batch,
                        routing_cache=routing_cache,
                        mask=full_mask,
                    )

                velocity, dv = jax.jvp(v_fn, (f_t,), (epsilon,))
                velocity = velocity.astype(jnp.float32)
                dv = dv.astype(jnp.float32)
                per_point_div = epsilon * dv

                f_t_new = (f_t - dt * velocity).astype(jnp.float32)
                log_det_new = (log_det - dt * per_point_div[:, :, 0]).astype(jnp.float32)
                return (f_t_new, log_det_new), None

            init_log_det = jnp.zeros((batch_size, x_qry.shape[1]), dtype=jnp.float32)
            (f_0, log_det), _ = jax.lax.scan(step_fn, (f_1, init_log_det), steps)

            log_p_base = -0.5 * (f_0[:, :, 0] ** 2 + jnp.log(2 * jnp.pi))
            log_p = log_p_base + log_det
            valid = qry_mask.astype(jnp.float32)
            nll = -jnp.sum(log_p * valid) / jnp.maximum(jnp.sum(valid), 1.0)
            return nll

        model._nll_jit_cache = _nll

    _, state = nnx.split(model)
    return float(model._nll_jit_cache(state, x_ctx, y_ctx, x_qry, f_qry, key, ctx_mask, qry_mask, ode_steps))


def _make_sharding_helpers(mesh):
    shard3d = lambda x: jax.device_put(x, NamedSharding(mesh, P("batch", None, None)))
    shard2d = lambda x: jax.device_put(x, NamedSharding(mesh, P("batch", None)))
    shard1d = lambda x: jax.device_put(x, NamedSharding(mesh, P("batch")))
    replicate = NamedSharding(mesh, P())
    return shard3d, shard2d, shard1d, replicate


def _replicate_module(module, replicated_sharding):
    state = nnx.state(module)
    state = jax.device_put(state, replicated_sharding)
    nnx.update(module, state)


def _clamp_warmup_steps(num_steps: int, warmup: int) -> int:
    if num_steps <= 1:
        return 0
    return max(0, min(warmup, num_steps - 1))


def train_cfm(
    data_fn,
    model_cfg: dict,
    num_steps: int = 1000,
    warmup: int = 500,
    batch_size: int = 128,
    lr: float = 5e-4,
    ema_decay: float = 0.9995,
    seed: int = 42,
    log_every: int = 500,
    log_fn=None,
    mesh=None,
):
    import optax

    print(f"Initializing SFNP training (d_min={model_cfg['d_min']:.4f})...")
    rngs = nnx.Rngs(0)
    model = SFNP(**model_cfg, rngs=rngs)

    warmup = _clamp_warmup_steps(num_steps, warmup)
    schedule = optax.warmup_cosine_decay_schedule(1e-5, lr, warmup, num_steps, 1e-5)
    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule)),
        wrt=nnx.Param,
    )

    ema_params = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
    ema_step_size = 1.0 - ema_decay

    shard3d = None
    if mesh is not None:
        num_devices = len(mesh.devices.flat)
        assert batch_size % num_devices == 0, (
            f"batch_size ({batch_size}) must be divisible by device count ({num_devices})"
        )
        shard3d, shard2d, _shard1d, replicated = _make_sharding_helpers(mesh)
        _replicate_module(model, replicated)
        _replicate_module(optimizer, replicated)
        ema_params = jax.device_put(ema_params, replicated)
        print(f"  Data-parallel: {num_devices} devices, {batch_size // num_devices} per device")

    @nnx.jit
    def train_step(model, optimizer, x_ctx, y_ctx, x_qry, f_qry, ctx_mask, qry_mask, prng_key, ema):
        key_t, key_n = jax.random.split(prng_key)
        batch_size_inner = x_ctx.shape[0]
        t = jax.random.uniform(key_t, (batch_size_inner,))
        f_0 = jax.random.normal(key_n, shape=f_qry.shape)
        f_0 = f_0 * qry_mask[:, :, None]
        f_0 = smooth_noise(f_0, x_qry, mask=qry_mask)
        t_exp = t.reshape(-1, 1, 1)
        f_t = (1.0 - t_exp) * f_0 + t_exp * f_qry
        target_v = f_qry - f_0

        full_mask = jnp.concatenate([ctx_mask, qry_mask], axis=1)

        def loss_fn(model_instance):
            pred_v = model_instance(x_ctx, y_ctx, x_qry, f_t, t, routing_cache=None, mask=full_mask)
            sq_err = (pred_v - target_v) ** 2
            masked_err = sq_err * qry_mask[:, :, None]
            return jnp.sum(masked_err) / jnp.maximum(jnp.sum(qry_mask), 1.0)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)

        current_params = nnx.state(model, nnx.Param)
        ema = optax.incremental_update(current_params, ema, ema_step_size)
        return loss, ema

    mesh_ctx = mesh if mesh is not None else nullcontext()
    data_fn_jit = jax.jit(data_fn, static_argnums=(1,))
    key = jax.random.PRNGKey(seed)

    hist = []
    t0 = time.time()
    data_time = 0.0
    train_time = 0.0
    with mesh_ctx:
        for step in range(num_steps):
            key, train_key, data_key = jax.random.split(key, 3)
            td0 = time.time()
            result = data_fn_jit(data_key, batch_size)
            if len(result) == 6:
                x_c, y_c, x_q, f_q, c_mask, q_mask = result
            else:
                x_c, y_c, x_q, f_q = result
                c_mask = jnp.ones(x_c.shape[:2], dtype=bool)
                q_mask = jnp.ones(x_q.shape[:2], dtype=bool)
            data_time += time.time() - td0
            if shard3d is not None:
                x_c, y_c, x_q, f_q = map(shard3d, (x_c, y_c, x_q, f_q))
                c_mask = shard2d(c_mask.astype(jnp.float32)).astype(bool)
                q_mask = shard2d(q_mask.astype(jnp.float32)).astype(bool)
            tt0 = time.time()
            loss, ema_params = train_step(
                model, optimizer, x_c, y_c, x_q, f_q, c_mask, q_mask, train_key, ema_params
            )
            loss_val = float(loss)
            train_time += time.time() - tt0
            hist.append(loss_val)
            if step % log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  Step {step:05d} | CFM Loss: {loss_val:.4f} | {elapsed:.1f}s "
                    f"(data: {data_time:.1f}s, train: {train_time:.1f}s)"
                )
                data_time = 0.0
                train_time = 0.0
            if log_fn is not None:
                log_fn(step, loss_val)

    elapsed = time.time() - t0
    print(f"  Final  | CFM Loss: {loss_val:.4f} | {elapsed:.1f}s")
    print("  Loading EMA weights...")
    nnx.update(model, ema_params)
    return model, hist


def distill_reflow(
    teacher_model,
    data_fn,
    model_cfg: dict,
    ode_steps: int = 15,
    num_steps: int = 500,
    batch_size: int = 128,
    lr: float = 1e-4,
    ema_decay: float = 0.9995,
    seed: int = 42,
    log_every: int = 100,
    log_fn=None,
    mesh=None,
):
    import optax

    print("\nInitializing SFNP reflow distillation...")
    rngs = nnx.Rngs(seed)
    student = SFNP(**model_cfg, rngs=rngs)

    teacher_state = nnx.state(teacher_model, nnx.Param)
    nnx.update(student, teacher_state)

    warmup = _clamp_warmup_steps(num_steps, 100)
    schedule = optax.warmup_cosine_decay_schedule(1e-5, lr, warmup, num_steps, 1e-5)
    optimizer = nnx.Optimizer(
        student,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule)),
        wrt=nnx.Param,
    )

    ema_params = jax.tree.map(lambda x: x.copy(), nnx.state(student, nnx.Param))
    ema_step_size = 1.0 - ema_decay

    shard3d = None
    if mesh is not None:
        num_devices = len(mesh.devices.flat)
        assert batch_size % num_devices == 0, (
            f"batch_size ({batch_size}) must be divisible by device count ({num_devices})"
        )
        shard3d, shard2d, _shard1d, replicated = _make_sharding_helpers(mesh)
        _replicate_module(student, replicated)
        _replicate_module(optimizer, replicated)
        ema_params = jax.device_put(ema_params, replicated)
        print(f"  Distill data-parallel: {num_devices} devices, {batch_size // num_devices} per device")

    @nnx.jit
    def reflow_step(student, optimizer, x_ctx, y_ctx, x_qry, f_0, f_1, ctx_mask, qry_mask, prng_key, ema):
        key_t, _ = jax.random.split(prng_key)
        batch_size_inner = x_ctx.shape[0]
        t = jax.random.uniform(key_t, (batch_size_inner,))
        t_exp = t.reshape(-1, 1, 1)
        f_t = (1.0 - t_exp) * f_0 + t_exp * f_1
        target_v = f_1 - f_0

        full_mask = jnp.concatenate([ctx_mask, qry_mask], axis=1)

        def loss_fn(model_instance):
            pred_v = model_instance(x_ctx, y_ctx, x_qry, f_t, t, routing_cache=None, mask=full_mask)
            sq_err = (pred_v - target_v) ** 2
            masked_err = sq_err * qry_mask[:, :, None]
            return jnp.sum(masked_err) / jnp.maximum(jnp.sum(qry_mask), 1.0)

        loss, grads = nnx.value_and_grad(loss_fn)(student)
        optimizer.update(student, grads)

        current_params = nnx.state(student, nnx.Param)
        ema = optax.incremental_update(current_params, ema, ema_step_size)
        return loss, ema

    mesh_ctx = mesh if mesh is not None else nullcontext()
    data_fn_jit = jax.jit(data_fn, static_argnums=(1,))
    key = jax.random.PRNGKey(seed)

    hist = []
    t0 = time.time()
    with mesh_ctx:
        for step in range(num_steps):
            key, teacher_key, train_key, data_key = jax.random.split(key, 4)
            result = data_fn_jit(data_key, batch_size)
            if len(result) == 6:
                x_c, y_c, x_q, _f_q, c_mask, q_mask = result
            else:
                x_c, y_c, x_q, _f_q = result
                c_mask = jnp.ones(x_c.shape[:2], dtype=bool)
                q_mask = jnp.ones(x_q.shape[:2], dtype=bool)

            keys = jax.random.split(teacher_key, batch_size)
            f_0, f_1 = generate_teacher_trajectories(
                teacher_model,
                x_c,
                y_c,
                x_q,
                keys,
                ode_steps=ode_steps,
                ctx_mask=c_mask,
                qry_mask=q_mask,
            )

            if shard3d is not None:
                x_c, y_c, x_q, f_0, f_1 = map(shard3d, (x_c, y_c, x_q, f_0, f_1))
                c_mask = shard2d(c_mask.astype(jnp.float32)).astype(bool)
                q_mask = shard2d(q_mask.astype(jnp.float32)).astype(bool)

            loss, ema_params = reflow_step(
                student,
                optimizer,
                x_c,
                y_c,
                x_q,
                f_0,
                f_1,
                c_mask,
                q_mask,
                train_key,
                ema_params,
            )
            loss_val = float(loss)
            hist.append(loss_val)
            if step % log_every == 0:
                elapsed = time.time() - t0
                print(f"  Step {step:05d} | Distill Loss: {loss_val:.4f} | {elapsed:.1f}s")
            if log_fn is not None:
                log_fn(step, loss_val)

    print("  Loading EMA weights...")
    nnx.update(student, ema_params)
    return student, hist

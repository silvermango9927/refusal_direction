import torch
import einops
from torch import Tensor
from jaxtyping import Int, Float


def get_device() -> str:
    """Return the best available torch device: CUDA, then Apple Silicon (MPS), then CPU.

    This lets the pipeline run on machines without a CUDA GPU (e.g. Apple Silicon
    Macs), which the original code assumed via hard-coded ``device_map="cuda"``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def high_precision_dtype(device) -> torch.dtype:
    """Dtype for high-precision accumulators (mean activations, CE loss, KL div).

    The original code uses ``torch.float64`` to avoid numerical drift. Apple's MPS
    backend does not support float64 at all, so we fall back to ``float32`` there.
    On CUDA/CPU we keep the original float64 behaviour.
    """
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    return torch.float32 if device_type == "mps" else torch.float64


def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, 'd_model']) -> Float[Tensor, '... d_model']:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)

    proj = einops.einsum(matrix, vec.unsqueeze(-1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj
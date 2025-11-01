"""
TT-NN TRAINING DEMO: Host-side grads + TT-NN forward (Softmax Regression on MNIST)
----------------------------------------------------------------------------------

Overview
========
This script trains a single-layer softmax regression model on MNIST where:
  • FORWARD PASS runs on a Tenstorrent device via TT-NN (ttnn.linear).
  • LOSS + GRADIENTS + WEIGHT UPDATES happen on the host (PyTorch/Torch ops).

Why this design?
----------------
The public TT-NN API is inference-first and does not expose autograd. To "train",
we offload the matrix-multiply heavy forward pass to the device, then bring logits
back to the host, compute the cross-entropy loss + exact gradients, update, and
re-upload new weights for the next step's forward.

Data/Shape Conventions
----------------------
MNIST images: (B, 1, 28, 28)  → flattened to X: (B, 784)
Weights W: (10, 784) and bias b: (10,)
Logits = X @ W^T + b → (B, 10)

Device Layouts
--------------
We upload tensors to TT with a chosen layout (default TILE). For weights, we first
upload in ROW_MAJOR and then transpose to (784, 10) and convert to TILE for the
ttnn.linear call. This pattern mirrors common TT-NN examples.

Numerics
--------
We compute forward in bf16 on device for performance, then convert logits back
to float32 on host for stable softmax/CE. You can change dtype/layout in Config.

Extending to MLP
----------------
You can add a hidden layer by:
  1) Doing X→W1,b1→ReLU→W2,b2 forward on the device (keep intermediates),
  2) Pulling intermediates/logits to host,
  3) Computing grads dW2,db2 and dW1,db1 using saved activations on host,
  4) Updating host weights and re-uploading each step.

Troubleshooting
---------------
• First-run compile: the very first call to an op on TT compiles kernels. Expect
  a slower first batch and faster subsequent iterations (program cache).
• Memory: If you hit L1/DRAM issues, reduce batch_size or switch memory_config.
• API drift: Ensure your `ttnn` version matches your hardware SDK.

"""

from loguru import logger
import os, time, math
from dataclasses import dataclass
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import ttnn

logger.remove()
logger.add(lambda m: print(m, end=""))

# ----------------------------------------------------------------------------- #
#                             Device context manager                            #
# ----------------------------------------------------------------------------- #
class TTDeviceContext:
    """
    RAII-style context that opens a TT device on enter and guarantees closure
    on exit, even if exceptions occur. This keeps each run self-contained and
    avoids "device already open" errors between separate sections/notebooks.

    Usage:
        with TTDeviceContext(device_id=0) as device:
            # ... run TT-NN ops with `device` ...
    """
    def __init__(self, device_id: int = 0, **open_kwargs):
        self.device_id = device_id
        self.open_kwargs = open_kwargs
        self.device = None

    def __enter__(self):
        self.device = ttnn.open_device(device_id=self.device_id, **self.open_kwargs)
        return self.device

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.device is not None:
                ttnn.close_device(self.device)
        finally:
            self.device = None
        # propagate exceptions
        return False


# ----------------------------------------------------------------------------- #
#                                    Config                                     #
# ----------------------------------------------------------------------------- #
@dataclass
class Config:
    data_root: str = "./data"
    batch_size: int = 256              # ↑ reduces host<->device overhead; lower if OOM
    epochs: int = 2
    lr: float = 0.1                    # simple SGD
    device_id: int = 0
    dtype_tt = ttnn.bfloat16           # device compute dtype
    layout = ttnn.TILE_LAYOUT          # device layout for main ops
    seed: int = 0                      # host RNG seed
    log_interval: int = 50             # log every N train batches
    eval_batches: int = 40             # ~eval_batches * batch_size samples in quick eval
    # Optional knobs you can toggle later:
    weight_memcfg = None               # e.g., ttnn.L1_MEMORY_CONFIG to stash bias/weights
    warmup_forward: bool = True        # one "dry-run" to compile kernels before timing

CFG = Config()


# ----------------------------------------------------------------------------- #
#                               Data: MNIST loaders                             #
# ----------------------------------------------------------------------------- #
def get_loaders() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Returns train/test DataLoaders with standard MNIST normalization.
    Train loader shuffles and drops last incomplete batch for consistent shapes.
    """
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(CFG.data_root, train=True,  download=True, transform=tfm)
    test  = torchvision.datasets.MNIST(CFG.data_root, train=False, download=True, transform=tfn_to01())
    train_loader = torch.utils.data.DataLoader(train, batch_size=CFG.batch_size, shuffle=True,  drop_last=True)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=CFG.batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader

def tfn_to01():
    """
    Same normalization as training. Kept separate to make it obvious where to
    tweak test-time preprocessing if you later want different stats.
    """
    return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])


# ----------------------------------------------------------------------------- #
#                              Model (host weights)                             #
# ----------------------------------------------------------------------------- #
class SoftmaxReg(torch.nn.Module):
    """
    Softmax Regression:
        logits = X @ W^T + b
    Shapes:
        X: (B, 784), W: (10, 784), b: (10,), logits: (B, 10)

    We store weights in PyTorch tensors but *do not* use autograd here.
    Gradients are computed manually with explicit formulas.
    """
    def __init__(self, in_dim: int = 28*28, num_classes: int = 10):
        super().__init__()
        torch.manual_seed(CFG.seed)
        # Small random init for W; zero init for b is fine in softmax regression.
        self.W = torch.randn(num_classes, in_dim) * 0.01   # [10, 784]
        self.b = torch.zeros(num_classes)                  # [10]

    @torch.no_grad()
    def sgd_step(self, dW: torch.Tensor, db: torch.Tensor, lr: float) -> None:
        """
        In-place SGD update. Keep tensors contiguous to avoid unexpected copies
        before the next host→device transfer.
        """
        self.W -= lr * dW
        self.b -= lr * db
        self.W = self.W.contiguous()
        self.b = self.b.contiguous()


# ----------------------------------------------------------------------------- #
#                         Host <-> TT-NN conversion helpers                      #
# ----------------------------------------------------------------------------- #
def host_to_tt(t: torch.Tensor, device, dtype=CFG.dtype_tt, layout=CFG.layout, memcfg=None):
    """
    Upload a contiguous host tensor to TT device with chosen dtype/layout.
    """
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=memcfg)

def to_row_major_on_device(t: torch.Tensor, device, dtype=CFG.dtype_tt):
    """
    Upload to ROW_MAJOR first, then transpose/convert to TILE on device.
    This often mirrors TT-NN examples and can be convenient for weight handling.
    """
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

def tt_to_host(tt):
    """
    Download TT tensor to a PyTorch tensor on host. Result dtype/layout become torch defaults.
    """
    return ttnn.to_torch(tt)


# ----------------------------------------------------------------------------- #
#                           Device-side forward (logits)                         #
# ----------------------------------------------------------------------------- #
def logits_forward_tt(device, X_host: torch.Tensor, W_host: torch.Tensor, b_host: torch.Tensor) -> torch.Tensor:
    """
    Compute logits on TT device:

        logits = X @ W^T + b

    Steps:
      1) Upload X (B, 784) → device (TILE by default).
      2) Upload W (10, 784) → device ROW_MAJOR → transpose to (784, 10) → convert to TILE.
      3) Upload b (10,) → reshape to (1, 10) for broadcast add.
      4) Run ttnn.linear and bring logits back to host as float32.

    Returns:
      logits_host: torch.Tensor (B, 10) in float32 for stable loss/grads.
    """
    # 1) Inputs
    X_tt  = host_to_tt(X_host, device, dtype=CFG.dtype_tt, layout=CFG.layout)

    # 2) Weights: keep a ROW_MAJOR copy on device, then transpose+layout-convert.
    #    If you want to pin in L1 between steps, pass memory_config=CFG.weight_memcfg.
    W_rm  = to_row_major_on_device(W_host, device)
    W_ttT = ttnn.to_layout(ttnn.transpose(W_rm, 0, 1), CFG.layout)  # (784, 10) TILE

    # 3) Bias
    b_tt  = host_to_tt(b_host.view(1, -1), device, dtype=CFG.dtype_tt, layout=CFG.layout, memcfg=CFG.weight_memcfg)

    # 4) Linear on device
    logits_tt = ttnn.linear(X_tt, W_ttT, bias=b_tt)

    # Back to host for numerically-stable softmax/CE
    logits = tt_to_host(logits_tt).to(torch.float32)
    return logits


# ----------------------------------------------------------------------------- #
#                       Loss, gradients, and metrics (host)                      #
# ----------------------------------------------------------------------------- #
def softmax_cross_entropy_with_logits(logits: torch.Tensor, targets: torch.Tensor):
    """
    Numerically stable softmax + cross-entropy.
    Args:
        logits: (B, C)
        targets: (B,) class indices
    Returns:
        loss (scalar tensor), probs (B, C)
    """
    z = logits - logits.max(dim=1, keepdim=True).values
    exp_z = torch.exp(z)
    probs = exp_z / exp_z.sum(dim=1, keepdim=True)
    N = logits.size(0)
    loss = -torch.log(probs[torch.arange(N), targets] + 1e-9).mean()
    return loss, probs

def compute_grads_logreg(X: torch.Tensor, probs: torch.Tensor, targets: torch.Tensor, num_classes: int = 10):
    """
    Derivatives for softmax regression:
      dW = ((probs - onehot).T @ X) / N
      db = mean(probs - onehot, dim=0)
    Shapes:
      X:     (B, 784)
      probs: (B, 10)
      dW:    (10, 784)
      db:    (10,)
    """
    N = X.size(0)
    onehot = torch.zeros_like(probs)
    onehot[torch.arange(N), targets] = 1.0
    diff = (probs - onehot)        # (B, 10)
    dW = diff.t() @ X              # (10, 784)
    dW /= N
    db = diff.mean(dim=0)          # (10,)
    return dW, db

@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Top-1 accuracy from raw logits.
    """
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


# ----------------------------------------------------------------------------- #
#                                    Training                                   #
# ----------------------------------------------------------------------------- #
def train() -> None:
    """
    Full train loop:
      • Open TT device
      • Iterate epochs/batches
      • Forward on TT, loss+grads on host, host-side SGD update
      • Periodic logging + quick eval
      • Save final host weights to disk
    """
    # Repro
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)

    train_loader, test_loader = get_loaders()
    model = SoftmaxReg()

    with TTDeviceContext(device_id=CFG.device_id) as device:
        logger.info(f"\n[TT] Opened device {CFG.device_id}\n")

        # Optional warmup to compile kernels & prime program cache.
        if CFG.warmup_forward:
            images, labels = next(iter(train_loader))
            X_warm = images.view(images.size(0), -1).to(torch.float32)
            _ = logits_forward_tt(device, X_warm, model.W, model.b)
            logger.info("[warmup] Completed first forward to prime program cache.\n")

        global_step = 0
        for ep in range(1, CFG.epochs + 1):
            t0 = time.time()
            running_loss, running_acc = 0.0, 0.0

            for bi, (images, labels) in enumerate(train_loader, start=1):
                # Flatten to (B, 784); keep on host (CPU) for grads.
                X = images.view(images.size(0), -1).to(torch.float32)
                y = labels

                # --- forward on TT device ---
                logits = logits_forward_tt(device, X, model.W, model.b)

                # --- loss + grads on host ---
                loss, probs = softmax_cross_entropy_with_logits(logits, y)
                dW, db = compute_grads_logreg(X, probs, y, num_classes=10)

                # --- host-side SGD update ---
                model.sgd_step(dW, db, lr=CFG.lr)

                # Metrics
                acc = accuracy_from_logits(logits, y)
                running_loss += float(loss.item())
                running_acc  += acc
                global_step  += 1

                if bi % CFG.log_interval == 0:
                    avg_loss = running_loss / CFG.log_interval
                    avg_acc  = running_acc  / CFG.log_interval
                    logger.info(f"[ep {ep:02d}] step {bi:04d}  loss={avg_loss:.4f}  acc={avg_acc*100:.2f}%\n")
                    running_loss = 0.0
                    running_acc  = 0.0

            dt = time.time() - t0
            logger.info(f"[ep {ep:02d}] epoch time: {dt:.1f}s\n")

            # Quick evaluation on a subset (speeds up iteration during dev)
            eval_acc = evaluate(device, model, test_loader, max_batches=CFG.eval_batches)
            logger.info(f"[ep {ep:02d}] eval acc (~{CFG.eval_batches*CFG.batch_size} samples): {eval_acc*100:.2f}%\n")

        logger.info("[TT] Training complete; closing device.\n")

    # Persist host-side weights for later inference/export.
    os.makedirs("ttnn_ckpts", exist_ok=True)
    torch.save({"W": model.W, "b": model.b}, "ttnn_ckpts/logreg_mnist_host.pt")
    logger.info("Saved host weights to ttnn_ckpts/logreg_mnist_host.pt\n")


# ----------------------------------------------------------------------------- #
#                                   Evaluation                                  #
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(device, model: SoftmaxReg, test_loader, max_batches: int = 40) -> float:
    """
    Light-weight evaluation that still runs forward on TT device (to measure real
    device behavior). Limits to `max_batches` for speed during development.
    """
    seen, correct = 0, 0
    for bi, (images, labels) in enumerate(test_loader, start=1):
        if bi > max_batches:
            break
        X = images.view(images.size(0), -1).to(torch.float32)
        logits = logits_forward_tt(device, X, model.W, model.b)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        seen += labels.numel()
    return correct / max(1, seen)


# ----------------------------------------------------------------------------- #
#                                      Main                                     #
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    train()

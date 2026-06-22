import torch
import torch.nn.functional as F
from torchmetrics import Metric
class CreteRoffetBlur(Metric):
    """
    Args:
        spatial_dims (int): 2 for (H, W), 3 for (H, W, D)
        return_map (bool): If True, return full blur map. If False, return scalar blur score.
        kernel_size (int): Size of blur kernel (default: 5)
        eps (float): Small value to avoid division by zero (default: 1e-8)
        normalize (bool): Whether to normalize input tensors to [0, 1] (default: True)
        channel_average (bool): Whether to average across channels (default: True)
    """
    def __init__(
        self,  spatial_dims: int = 3,
        return_map: bool = False,   kernel_size: int = 5, eps: float = 1e-8,
        normalize: bool = True, channel_average: bool = True,  **kwargs):
        super().__init__(**kwargs)
        assert spatial_dims in (2, 3), "Only 2D or 3D inputs supported"
        assert kernel_size % 2 == 1, "Kernel size should be odd"
        self.spatial_dims = spatial_dims
        self.return_map = return_map
        self.kernel_size = kernel_size
        self.eps = eps
        self.normalize = normalize
        self.channel_average = channel_average
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def _neighbor_diff_max(self, x):
        """Compute max of finite differences along spatial dimensions."""
        # diff of each ro/col & past one
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        dx = F.pad(dx, (0, 0, 0, 1)) # pad zero ro at last ro to get the ip dim
        dy = F.pad(dy, (0, 1, 0, 0))
        return torch.maximum(dx.abs(), dy.abs())

    def _make_kernel(self, device, dtype):
        """Create isotropic blur kernel."""
        k = self.kernel_size
        kernel = torch.ones((1, 1, k, k), device=device, dtype=dtype) / (k * k)
        return kernel
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute Crété-Roffet blur score or blur map.
        Args:
            images: Tensor of shape (B, C, H, W) or (B, C, H, W, D)
        Returns:
            Tensor: (B,) if return_map=False and channel_average=True
                    (B, C) if return_map=False and channel_average=False
                    else: full blur maps
        """
        assert images.ndim in (4, 5), "Expected shape (B, C, H, W) or (B, C, H, W, D)"
        B, C = images.shape[:2]
        
        # norm image 0->1
        if self.normalize:
            images = (images - images.amin(dim=tuple(range(2, images.ndim)), keepdim=True)) / \
                    (images.amax(dim=tuple(range(2, images.ndim)), keepdim=True) + self.eps)
        
        # diff of ip pixels 
        df = self._neighbor_diff_max(images)
        
        # blur ip
        kernel = self._make_kernel(images.device, images.dtype)
        blurred = F.conv2d(images.view(-1, 1, *images.shape[2:]), kernel, padding=self.kernel_size // 2)
        blurred = blurred.view(B, C, *images.shape[2:])
        
        # diff of blurr pixels 
        db = self._neighbor_diff_max(blurred)

        # make zero the best 
        v = torch.clamp(df - db, min=0)
        blur_map = 1.0 - v / (df + self.eps)
        blur_map = blur_map.clamp(0.0, 1.0)
       
        # mean map
        if self.return_map:
            return blur_map
        else:
            score = blur_map.mean(dim=tuple(range(2, images.ndim)))  # shape (B, C)
            return score.mean(dim=1) if self.channel_average else score  # (B,)
        


    def update(self, images: torch.Tensor):
        score = self.forward(images)  # shape (B,)
        self.total += score.sum()
        self.count += score.numel()
    def compute(self):
        return self.total / self.count

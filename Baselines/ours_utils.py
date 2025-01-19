import torch
from typing import Optional, Union, Callable
from abc import ABCMeta, abstractmethod
from torch.nn import functional as F
from typing import Optional, Union
from typing import Any, Dict, Optional, Sequence





activation_t = Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]

class Buffer(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class RandomBuffer(torch.nn.Linear, Buffer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=torch.float,
        activation: Optional[activation_t] = torch.relu_,
    ) -> None:
        super(torch.nn.Linear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.activation: activation_t = (
            torch.nn.Identity() if activation is None else activation
        )
        W = torch.empty((out_features, in_features), **factory_kwargs)
        b = torch.empty(out_features, **factory_kwargs) if bias else None

        # Using buffer instead of parameter
        self.register_buffer("weight", W)
        self.register_buffer("bias", b)

        # Random Initialization
        self.reset_parameters()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        return self.activation(super().forward(X))


class AnalyticLinear(torch.nn.Linear, metaclass=ABCMeta):
    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super(torch.nn.Linear, self).__init__()  # Skip the Linear class
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gamma: float = gamma
        self.bias: bool = bias
        self.dtype = dtype

        # Linear Layer
        if bias:
            in_features += 1
        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

    # @torch.inference_mode()
    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)
        return X @ self.weight

    @property
    def in_features(self) -> int:
        if self.bias:
            return self.weight.shape[0] - 1
        return self.weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def reset_parameters(self) -> None:
        # Following the equation (4) of ACIL, self.weight is set to \hat{W}_{FCN}^{-1}
        self.weight = torch.zeros((self.weight.shape[0], 0)).to(self.weight)

    @abstractmethod
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        raise NotImplementedError()

    def update(self) -> None:
        assert torch.isfinite(self.weight).all(), (
            "Pay attention to the numerical stability! "
            "A possible solution is to increase the value of gamma. "
            "Setting self.dtype=torch.double also helps."
        )


class GeneralizedARM(AnalyticLinear):
    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        A = torch.zeros((0, in_features, in_features), **factory_kwargs)
        self.register_buffer("A", A)

        C = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("C", C)

        self.cnt = torch.zeros(0, dtype=torch.int, device=device)

    @property
    def out_features(self) -> int:
        return self.C.shape[1]

    # @torch.inference_mode()
    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.weight)
        ### 
        y = torch.argmax(y, dim=1)
        # Bias
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1)), dim=-1)

        # GCIL
        num_targets = int(y.max()) + 1
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            torch.cuda.empty_cache()
            # Increment C
            tail = torch.zeros((self.C.shape[0], increment_size)).to(self.weight)
            self.C = torch.cat((self.C, tail), dim=1)
            # Increment cnt
            tail = torch.zeros((increment_size,)).to(self.cnt)
            self.cnt = torch.cat((self.cnt, tail))
            # Increment A
            tail = torch.zeros((increment_size, self.in_features, self.in_features))
            self.A = torch.cat((self.A, tail.to(self.A)))
            torch.cuda.empty_cache()
        else:
            num_targets = self.out_features

        # ACIL
        Y = F.one_hot(y, max(num_targets, num_targets)).to(self.C)
        self.C += X.T @ Y

        # Label Balancing
        y_labels, label_cnt = torch.unique(y, sorted=True, return_counts=True)
        y_labels, label_cnt = y_labels.to(self.cnt.device), label_cnt.to(self.cnt.device)
        self.cnt[y_labels] += label_cnt

        # Accumulate
        for i in range(num_targets):
            X_i = X[y == i]
            self.A[i] += X_i.T @ X_i

    # @torch.inference_mode()
    @torch.no_grad()
    def update(self):
        cnt_inv = 1 / self.cnt.to(self.dtype)
        cnt_inv[torch.isinf(cnt_inv)] = 0  # replace inf with 0
        cnt_inv *= len(self.cnt) / cnt_inv.sum()

        weighted_A = torch.sum(cnt_inv[:, None, None].mul(self.A), dim=0)
        A = weighted_A + self.gamma * torch.eye(self.in_features).to(self.A)
        C = self.C.mul(cnt_inv[None, :])

        self.weight = torch.inverse(A) @ C



class Regression(torch.nn.Module):
    def __init__(
        self,
        backbone_output: int,
        buffer_size: int = 2048,
        gamma: float = 0.1,
        device=None,
        dtype=torch.double,
        linear = GeneralizedARM,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone_output = backbone_output
        self.buffer_size = buffer_size
        self.buffer = RandomBuffer(backbone_output, buffer_size, **factory_kwargs)
        self.analytic_linear = linear(buffer_size, gamma, **factory_kwargs)
        self.eval()


    @torch.no_grad()
    def inference(self, X):
        return self.analytic_linear(self.feature_expansion(X))

    @torch.no_grad()
    def feature_expansion(self, X: torch.Tensor) -> torch.Tensor:
        return self.buffer(X)
    
    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.analytic_linear(self.feature_expansion(X))

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> None:
        Y = torch.nn.functional.one_hot(y)
        X = self.feature_expansion(X)
        self.analytic_linear.fit(X, Y)

    @torch.no_grad()
    def update(self) -> None:
        self.analytic_linear.update()

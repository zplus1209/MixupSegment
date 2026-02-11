import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import minimize, Bounds, minimize_scalar


def _check_param_device(param, old_param_device) -> int:
    """This helper function is to check if the parameters are located
        in the same device. Currently, the conversion between model parameters
        and single vector form is not supported for multiple allocations,
        e.g. parameters in different GPUs, or mixture of CPU/GPU.

        The implementation is taken from: https://github.com/pytorch/pytorch/blob/22a34bcf4e5eaa348f0117c414c3dd760ec64b13/torch/nn/utils/convert_parameters.py#L57

    Args:
        param ([TensorType]): a Tensor of a parameter of a model.
        old_param_device ([int]): the device where the first parameter
            of a model is allocated.

    Returns:
        old_param_device (int): report device for the first time

    """
    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError(
                "Found two parameters on different devices, "
                "this is currently not supported."
            )
    return old_param_device


def apply_vector_grad_to_parameters(vec, parameters, accumulate: bool = False):
    """Apply vector gradients to the parameters

    Args:
        vec (TensorType): a single vector represents the gradients of a model.
        parameters (Iterable[TensorType]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old grad of the parameter
        if accumulate:
            param.grad = (
                param.grad + vec[pointer : pointer + num_param].view_as(param).data
            )
        else:
            param.grad = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param



class CAGClassifierV0(nn.Module):
    def __init__(self, args) -> None:
        super(CAGClassifierV0, self).__init__()
    
        self.args = args


class CAGSegmenterV0(nn.Module):
    def __init__(self, args) -> None:
        super(CAGSegmenterV0, self).__init__()

        self.args = args
        self.cagrad_c = args.cagrad_c
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, pred, target) -> torch.Tensor:
        B, C, _, _ = tuple(pred.size())

        self.losses = torch.zeros(C).to(pred.device)

        _preds = pred.permute(0, 2, 3, 1).flatten(0, -2)
        _target = target.permute(0, 2, 3, 1).flatten(0, -2)

        for cidx in range(C):
            target_bool = _target[:, cidx] == 1
            if target_bool.sum() != 0:
                c_pred = _preds[target_bool]
                c_target = _target[target_bool]

                c_loss = self.loss_fn(c_pred, c_target)
            else:
                c_loss = F.binary_cross_entropy_with_logits(input = pred[:, cidx, :, :], target = target[:, cidx, :, :])

            self.losses[cidx] = c_loss

        return torch.sum(self.losses)

    def backward(self, parameters):
        grad = []
        
        for loss in self.losses:
            grad.append(
                tuple(
                    _grad.contiguous()
                    for _grad in torch.autograd.grad(
                        loss,
                        parameters,
                        # retain_graph=(False or idx != self.args.seg_n_classes - 1),
                        retain_graph=True,
                        allow_unused=False,
                    )
                )
            )
        
        grad_vec = torch.cat(
            list(
                map(lambda x: torch.nn.utils.parameters_to_vector(x).unsqueeze(0), grad)
            ),
            dim=0,
        )

        regularized_grad = self.cagrad_exact(grad_vec, self.args.seg_n_classes)
        apply_vector_grad_to_parameters(regularized_grad, parameters)

    
    def cagrad_exact(self, grad_vec, num_tasks):
        grads = grad_vec / 100.
        g0 = grads.mean(0)
        GG = grads.mm(grads.t())
        x_start = np.ones(num_tasks)/num_tasks
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.cpu().numpy()
        b = x_start.copy()
        c = (self.cagrad_c*g0.norm()).cpu().item()
        def objfn(x):
            return (x.reshape(1,num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + \
                    c * np.sqrt(x.reshape(1,num_tasks).dot(A).dot(x.reshape(num_tasks,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww= torch.Tensor(w_cpu).to(grad_vec.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-4)
        g = (g0 + lmbda * gw) / (1 + lmbda)
        return g * 100
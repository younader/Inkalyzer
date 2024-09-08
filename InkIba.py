from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn.functional as F
from distributions import *
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

def animate_history(history):
    fig, ax = plt.subplots()
    artists = []
    for i in range(history.shape[0]):
        container = ax.plot(np.arange(history.shape[1]),history[i],color='b')
        artists.append(container)
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=80)
    return HTML(ani.to_html5_video())

def get_noise(data,noise_type):
    """
    data: np.ndarray, shape (1, 1, D, size, size)
        The image to be explained.
    noise_type: str
        The type of noise to be used. Can be 'top', 'bottom', 'median', 'mean', 'max' or 'top85'.
    """
    if noise_type=='top':
        return data[:,:,0]
    elif noise_type=='bottom':
        return data[:,:,-1]
    elif noise_type=='median':
        return np.median(data,axis=2)
    elif noise_type=='mean':
        return np.mean(data,axis=2)
    elif noise_type=='top85':
        return np.percentile(data,85,axis=2)
    elif noise_type=='max':
        return np.max(data,axis=2)
    else:
        raise ValueError('noise_type should be one of top, bottom, median, mean or top75')
def attribute_z_diffmask(
    data,
    noise,
    model,
    loss_func,
    n_steps=10,
    lr=2e-1,
    beta=0.05,
    size=512,
    pool=False,
    batch_size=1,
):
    """
    data: torch.Tensor, shape (1, 1, D, size, size)
        The image to be explained.
    noise: torch.Tensor, shape (1, 1, D, size, size)
        The noise image to be used to bottleneck the information.
    model: torch.nn.Module
        The model to be explained.
    loss_func: Callable
        The loss function to be used. It should take two arguments: y_preds and y.
        This loss optimizes for perserving the mutual information between output y and intermediate bottleneck Z.
    n_steps: int
        The number of per-sample IBA optimization steps.
    lr: float
        The learning rate for the gates.
    beta: float
        The weight of the loss term restricting the information flow from X to Z.
    size: int
        The size of the input image.
    pool: bool
        Whether to use a pooling layer to smooth the gates due to the stochasticity.
    batch_size: int
        The batch size for the optimization. Since diffmask is sampling based, we can run multiple samples in parallel.
    """
    alpha = torch.full(
        (batch_size, 1, 30, 1, 1),
        5.0,
        requires_grad=True,
        device=next(model.parameters()).device,
    )
    data = data.repeat(batch_size, 1, 1, 1, 1)
    noise = noise.repeat(batch_size, 1, 1, 1, 1)
    alpha_history = []
    alpha_history.append(alpha.clone().detach())
    optimizer = torch.optim.RMSprop([alpha], lr=lr, centered=True)
    b, c, d, h, w = alpha.size()
    with torch.autocast(device_type="cuda"):
        y = model(data).detach()
        for _ in tqdm(range(n_steps)):
            optimizer.zero_grad()
            dist = RectifiedStreched(
                BinaryConcrete(torch.full_like(alpha, 0.2), alpha),
                l=-0.2,
                r=1.0,
            )
            gates = dist.rsample()
            gates = gates.repeat(1, 1, 1, size, size)
            if pool:
                gates = (
                    F.avg_pool1d(f.view(b, c, d), kernel_size=3, stride=1, padding=1)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            model_input = gates * data + (1 - gates) * noise
            y_preds = model(model_input)
            loss1 = loss_func(y_preds, y)
            loss2 = dist.log_expected_L0().mean()
            loss = loss1 + beta * loss2
            loss.backward()
            optimizer.step()
            alpha_history.append(alpha.clone().detach())
        gates = alpha.sigmoid().repeat(1, 1, 1, size, size).mean(dim=0).unsqueeze(0)
        final_pred = model(
            gates * data[0].unsqueeze(0) + (1 - gates) * noise[0].unsqueeze(0)
        )
    return (
        alpha.sigmoid().detach().cpu().numpy(),
        alpha_history,
        final_pred.detach().cpu(),
    )


def attribute_z_with_diffmask_pooling(
    data, noise, model, loss_func, n_steps=10, lr=2e-1, beta=0.05, size=512
):
    """
    data: torch.Tensor, shape (1, 1, D, size, size)
        The image to be explained.
    noise: torch.Tensor, shape (1, 1, D, size, size)
        The noise image to be used to bottleneck the information.
    model: torch.nn.Module
        The model to be explained.
    loss_func: Callable
        The loss function to be used. It should take two arguments: y_preds and y.
        This loss optimizes for perserving the mutual information between output y and intermediate bottleneck Z.
    n_steps: int
        The number of per-sample IBA optimization steps.
    lr: float
        The learning rate for the gates.
    beta: float
        The weight of the loss term restricting the information flow from X to Z.
    size: int
        The size of the input image.
    """
    alpha = torch.full(
        (1, 1, 30, 1, 1),
        5.0,
        requires_grad=True,
        device=next(model.parameters()).device,
    )
    alpha_history = []
    alpha_history.append(alpha.clone().detach())
    optimizer = torch.optim.RMSprop([alpha], lr=lr, centered=True)
    b, c, d, h, w = alpha.size()
    with torch.autocast(device_type="cuda"):
        y = model(data).detach()
        for _ in tqdm(range(n_steps)):
            optimizer.zero_grad()
            dist = RectifiedStreched(
                BinaryConcrete(torch.full_like(alpha, 0.2), alpha),
                l=-0.2,
                r=1.0,
            )
            gates = dist.rsample()
            gates = gates.repeat(1, 1, 1, size, size)
            gates = (
                F.avg_pool1d(f.view(b, c, d), kernel_size=3, stride=1, padding=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            model_input = gates * data + (1 - gates) * noise
            y_preds = model(model_input)
            loss1 = loss_func(y_preds, y)
            loss2 = dist.log_expected_L0().mean()
            loss = loss1 + beta * loss2
            loss.backward()
            optimizer.step()
            alpha_history.append(alpha.clone().detach())
        gates = alpha.sigmoid().repeat(1, 1, 1, size, size)
        final_pred = model(gates * data + (1 - gates) * noise)
    return (
        alpha.sigmoid().detach().cpu().numpy(),
        alpha_history,
        final_pred.detach().cpu(),
    )


def attribute__random_iters(
    data, noise, model, loss_func, n_steps=10, lr=2e-1, beta=0.05, size=512
):
    """
    data: torch.Tensor, shape (1, 1, D, size, size)
        The image to be explained.
    noise: torch.Tensor, shape (1, 1, D, size, size)
        The noise image to be used to bottleneck the information.
    model: torch.nn.Module
        The model to be explained.
    loss_func: Callable
        The loss function to be used. It should take two arguments: y_preds and y.
        This loss optimizes for perserving the mutual information between output y and intermediate bottleneck Z.
    n_steps: int
        The number of per-sample IBA optimization steps.
    lr: float
        The learning rate for the gates.
    beta: float
        The weight of the loss term restricting the information flow from X to Z.
    size: int
        The size of the input image.
    """
    alpha = torch.FloatTensor(1, 1, 30, 1, 1, device="cpu").uniform_(-5, 5).cuda()
    alpha.requires_grad = True
    alpha_history = []
    alpha_history.append(alpha.clone().detach())
    optimizer = torch.optim.RMSprop([alpha], lr=lr, centered=True)
    with torch.autocast(device_type="cuda"):
        y = model(data).detach()
        for _ in tqdm(range(n_steps)):
            optimizer.zero_grad()
            gates = alpha.sigmoid().repeat(1, 1, 1, size, size)
            model_input = gates * data + (1 - gates) * noise
            y_preds = model(model_input)
            loss1 = loss_func(y_preds, y)
            loss2 = gates.mean()
            loss = loss1 + beta * loss2
            loss.backward()
            optimizer.step()
            alpha_history.append(alpha.clone().detach())
        gates = alpha.sigmoid().repeat(1, 1, 1, size, size)
        final_pred = model(gates * data + (1 - gates) * noise)
    return (
        alpha.sigmoid().detach().cpu().numpy(),
        alpha_history,
        final_pred.detach().cpu(),
    )

def attribute_z_lowres(
    data, noise, model, loss_func, n_steps=10, lr=2e-1, beta=0.05, size=512
):
    """
    data: torch.Tensor, shape (1, 1, D, size, size)
        The image to be explained.
    noise: torch.Tensor, shape (1, 1, D, size, size)
        The noise image to be used to bottleneck the information.
    model: torch.nn.Module
        The model to be explained.
    loss_func: Callable
        The loss function to be used. It should take two arguments: y_preds and y.
        This loss optimizes for perserving the mutual information between output y and intermediate bottleneck Z.
    n_steps: int
        The number of per-sample IBA optimization steps.
    lr: float
        The learning rate for the gates.
    beta: float
        The weight of the loss term restricting the information flow from X to Z.
    size: int
        The size of the input image.
    """
    alpha = torch.full(
        (1, 1, 15, 1, 1),
        5.0,
        requires_grad=True,
        device=next(model.parameters()).device,
    )
    alpha_history = []
    alpha_history.append(alpha.clone().detach())
    optimizer = torch.optim.Adam([alpha], lr=lr)
    # optimizer = torch.optim.RMSprop([alpha], lr=lr, centered=True)
    with torch.autocast(device_type="cuda"):
        y = model(data).detach()
        for _ in tqdm(range(n_steps)):
            optimizer.zero_grad()
            gates = F.interpolate(alpha.sigmoid().squeeze(-1).squeeze(-1),scale_factor=2,mode='linear').unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, size, size)
            model_input = gates * data + (1 - gates) * noise
            y_preds = model(model_input)
            loss1 = loss_func(y_preds, y)
            loss2 = gates.mean()
            loss = loss1 + beta * loss2
            loss.backward()
            optimizer.step()
            alpha_history.append(alpha.clone().detach())
        gates = F.interpolate(alpha.sigmoid().squeeze(-1).squeeze(-1),scale_factor=2,mode='linear').unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, size, size)
        final_pred = model(gates * data + (1 - gates) * noise)
    return (
        alpha.sigmoid().detach().cpu().numpy(),
        alpha_history,
        final_pred.detach().cpu(),
    )

def attribute_z(
    data, noise, model, loss_func, n_steps=10, lr=2e-1, beta=0.05, size=512
):
    """
    data: torch.Tensor, shape (1, 1, D, size, size)
        The image to be explained.
    noise: torch.Tensor, shape (1, 1, D, size, size)
        The noise image to be used to bottleneck the information.
    model: torch.nn.Module
        The model to be explained.
    loss_func: Callable
        The loss function to be used. It should take two arguments: y_preds and y.
        This loss optimizes for perserving the mutual information between output y and intermediate bottleneck Z.
    n_steps: int
        The number of per-sample IBA optimization steps.
    lr: float
        The learning rate for the gates.
    beta: float
        The weight of the loss term restricting the information flow from X to Z.
    size: int
        The size of the input image.
    """
    alpha = torch.full(
        (1, 1, 30, 1, 1),
        5.0,
        requires_grad=True,
        device=next(model.parameters()).device,
    )
    alpha_history = []
    alpha_history.append(alpha.clone().detach())
    optimizer = torch.optim.Adam([alpha], lr=lr)
    # optimizer = torch.optim.RMSprop([alpha], lr=lr, centered=True)
    with torch.autocast(device_type="cuda"):
        y = model(data).detach()
        for _ in tqdm(range(n_steps)):
            optimizer.zero_grad()
            gates = alpha.sigmoid().repeat(1, 1, 1, size, size)
            model_input = gates * data + (1 - gates) * noise
            y_preds = model(model_input)
            loss1 = loss_func(y_preds, y)
            loss2 = gates.mean()
            loss = loss1 + beta * loss2
            loss.backward()
            optimizer.step()
            alpha_history.append(alpha.clone().detach())
        gates = alpha.sigmoid().repeat(1, 1, 1, size, size)
        final_pred = model(gates * data + (1 - gates) * noise)
    return (
        alpha.sigmoid().detach().cpu().numpy(),
        alpha_history,
        final_pred.detach().cpu(),
    )


def attribute_3d(
    data,
    noise,
    model,
    loss_func,
    pred_label=None,
    n_steps=20,
    lr=2e-1,
    beta=0.05,
    size=512,
    window_size=16,
):
    """
    runs attribution on 3d data, similar to attribute_z but defines gates as patches in x y domain. For better convergence,
    data: torch.Tensor, shape (1, 1, D, size, size)
        The image to be explained.
    noise: torch.Tensor, shape (1, 1, D, size, size)
        The noise image to be used to bottleneck the information.
    model: torch.nn.Module
        The model to be explained.
    loss_func: Callable
        The loss function to be used. It should take two arguments: y_preds and y.
        This loss optimizes for perserving the mutual information between output y and intermediate bottleneck Z.
    pred_label: torch.Tensor, shape (1, 1, D, size, size)
        The predicted label of the image to be used as a mask for the gates in x y
    n_steps: int
        The number of per-sample IBA optimization steps.
    lr: float
        The learning rate for the gates.
    beta: float
        The weight of the loss term restricting the information flow from X to Z.
    size: int
        The size of the input image.
    window_size: int
        The size of the window to be used for the patches.
    """
    alpha_prelim, _, _ = attribute_z(
        data, noise, model, loss_func, n_steps=n_steps, lr=lr, beta=beta, size=size
    )

    alpha = (
        torch.Tensor(alpha_prelim)
        .repeat(1, 1, 1, size // window_size, size // window_size)
        .clone()
    ).to(next(model.parameters()).device)
    if pred_label is not None:
        alpha = alpha * pred_label.to(next(model.parameters()).device)
    alpha.requires_grad = True
    optimizer = torch.optim.RMSprop([alpha], lr=2e-2, centered=True)
    with torch.autocast(device_type="cuda"):
        y = model(data).detach()
        for _ in tqdm(range(n_steps)):
            optimizer.zero_grad()
            gates = F.interpolate(
                alpha.sigmoid().squeeze(0), scale_factor=window_size, mode="bilinear"
            ).unsqueeze(0)
            model_input = gates * data + (1 - gates) * noise
            y_preds = model(model_input)
            loss1 = loss_func(y_preds, y)
            loss2 = gates.mean()
            loss = loss1 + beta * loss2
            loss.backward()
            optimizer.step()
        gates = F.interpolate(
            alpha.sigmoid().squeeze(0), scale_factor=window_size, mode="bilinear"
        ).unsqueeze(0)
        final_pred = model(gates * data + (1 - gates) * noise)
    return (
        F.interpolate(
            alpha.sigmoid().squeeze(0), scale_factor=window_size, mode="bilinear"
        )
        .unsqueeze(0)
        .detach()
        .cpu()
        .numpy(),
        final_pred.detach().cpu().numpy(),
    )



def otsu_intraclass_variance(array, threshold):
    """
    Otsu's intra-class variance for a one-dimensional array.
    If all elements are above or below the threshold, this will throw a warning that can safely be ignored.
    """
    return np.nansum(
        [
            np.mean(cls) * np.var(array, where=cls)
            #   weight   ·  intra-class variance
            for cls in [array >= threshold, array < threshold]
        ]
    )


def get_otsu_threshold(array):
    """
    Get the Otsu threshold for a one-dimensional array.
    """

    return min(
        range(int(np.min(array) + 1), int(np.max(array))),
        key=lambda th: otsu_intraclass_variance(array, th),
    )

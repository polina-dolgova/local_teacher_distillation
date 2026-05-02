from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


def wrapper_method(func):
    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get("_attacks", {}).values():
            eval("atk." + func.__name__ + "(*args, **kwargs)")
        return result

    return wrapper_func


class Attack(object):
    def __init__(self, name, model):
        self.attack = name
        self._attacks = OrderedDict()
        self.set_model(model)

        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = None
            print("Failed to set device automatically, please try set_device() manual.")

        self.attack_mode = "default"
        self.supported_mode = ["default"]
        self.targeted = False
        self._target_map_function = None

        self.normalization_used = None
        self._normalization_applied = None

        if self.model.__class__.__name__ == "RobModel":
            self._set_rmodel_normalization_used(model)

        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, inputs, labels=None, *args, **kwargs):
        raise NotImplementedError

    @wrapper_method
    def set_model(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits = self.model(inputs)
        return logits

    @wrapper_method
    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag

    @wrapper_method
    def set_device(self, device):
        self.device = device

    @wrapper_method
    def _set_rmodel_normalization_used(self, model):
        mean = getattr(model, "mean", None)
        std = getattr(model, "std", None)
        if mean is not None and std is not None:
            self.normalization_used = {}
            n_channels = len(mean)
            mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
            std = torch.tensor(std).reshape(1, n_channels, 1, 1)
            self.normalization_used["mean"] = mean
            self.normalization_used["std"] = std
            self._set_normalization_applied(True)

    def normalize(self, inputs):
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return (inputs - mean) / std


class PGDL2(Attack):
    def __init__(
        self,
        model,
        eps=0.04,
        steps=50,
        random_start=False,
        eps_for_division=1e-10,
    ):
        super().__init__("PGDL2", model)
        self.eps = eps
        self.alpha = self.eps / 10.0
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.supported_mode = ["default", "targeted"]

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        device = images.device
        batch_size_all = images.size(0)
        batch_size = images.size(0)

        eps_all = torch.full((batch_size_all,), self.eps, device=device)
        alpha_all = eps_all / 10.0
        adv_images_all = images.clone().detach()
        adv_preds_all = torch.zeros_like(labels) - 1
        indices_all = torch.arange(batch_size_all, device=device)
        indices = indices_all.clone()
        delta_all = torch.zeros_like(eps_all)

        eps = torch.full((batch_size,), self.eps, device=device)
        alpha = eps / 10.0
        images = images.clone().detach()
        adv_images = images.clone().detach()
        labels = labels.clone().detach()
        orig_pred = None
        loss_fn = nn.CrossEntropyLoss()
        cont_flag = True

        while indices.numel() > 0:
            for _ in range(self.steps):
                adv_images.requires_grad_()
                logits = self.get_logits(adv_images)
                preds = logits.argmax(1)
                still_correct = preds.eq(labels)

                if still_correct.sum() == 0:
                    delta = adv_images - images[indices]
                    d_norm = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
                    adv_images_all[indices] = adv_images
                    eps_all[indices] = eps
                    alpha_all[indices] = alpha
                    adv_preds_all[indices] = preds
                    cont_flag = False
                    break
                else:
                    if ~still_correct.sum() != 0:
                        adv_preds_all[indices[~still_correct]] = preds[~still_correct]

                if orig_pred is None:
                    orig_pred = preds.detach()

                loss = loss_fn(logits, labels)
                grad = torch.autograd.grad(loss, adv_images)[0]
                grad_norm = (
                    grad.view(batch_size, -1).norm(p=2, dim=1) + self.eps_for_division
                )
                grad = grad / grad_norm.view(-1, 1, 1, 1)
                grad[~still_correct] = 0.0

                adv_images = adv_images.detach() + (alpha.view(-1, 1, 1, 1) * grad)
                delta = adv_images - images[indices]
                d_norm = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
                factor = torch.minimum(eps / d_norm, torch.ones_like(d_norm))
                adv_images = torch.clamp(
                    images[indices] + delta * factor.view(-1, 1, 1, 1),
                    min=0.0,
                    max=1.0,
                ).detach()

                adv_images = adv_images[still_correct]
                batch_size = still_correct.sum().item()
                eps = eps[still_correct]
                alpha = alpha[still_correct]
                preds = preds[still_correct]
                labels = labels[still_correct]
                indices = indices[still_correct]
                delta_all[indices] = d_norm[still_correct]
                adv_images_all[indices] = adv_images
                eps_all[indices] = eps
                alpha_all[indices] = alpha

            if not cont_flag:
                break

            torch.cuda.empty_cache()
            with torch.no_grad():
                preds = self.get_logits(adv_images).argmax(1)
                still_correct = preds.eq(labels)
                if still_correct.sum() != 0:
                    eps[still_correct] *= 2
                    alpha = eps / 10.0

        delta_final = (adv_images_all - images).view(batch_size_all, -1)
        delta_norm = delta_final.norm(p=2, dim=1)
        return adv_images_all, orig_pred, adv_preds_all, eps_all, delta_norm

from __future__ import annotations

import os
import time

import pandas as pd
import torch
from torch.utils.data import Dataset

from cifar.methods.amun.attack import PGDL2


class simpleDataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.data = self.data.detach().cpu().numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = image.transpose(1, 2, 0)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def compute_advset(
    args,
    forgetset,
    net,
    initial_eps=0.01,
    outdir=None,
    transform=None,
    indices=None,
):
    adv_img_list = None
    s_eps_list = []
    idx_list = []
    adv_pred_list = []
    orig_pred_list = []
    label_list = []
    delta_norm_list = []

    forgetset_subset = (
        torch.utils.data.Subset(forgetset, indices)
        if indices is not None
        else forgetset
    )
    dataloader = torch.utils.data.DataLoader(
        forgetset_subset,
        shuffle=False,
        batch_size=args.unlearning_batch_size,
        num_workers=4,
        pin_memory=True,
    )

    start_time = time.time()
    global_idx = 0
    for idx, (inputs, targets) in enumerate(dataloader):
        if idx % 50 == 0:
            print(idx)

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        if args.attack == "pgdl2":
            pgd_finder = PGDL2(net, eps=initial_eps)
            adv_img, orig_pred, adv_pred, eps_vec, dnorm_vec = pgd_finder.forward(
                inputs,
                targets,
            )
        else:
            raise ValueError(f"Unsupported attack: {args.attack}")

        adv_img_list = (
            adv_img
            if adv_img_list is None
            else torch.cat((adv_img_list, adv_img), dim=0)
        )

        batch_size = inputs.size(0)
        idx_list.extend(range(global_idx, global_idx + batch_size))
        label_list.extend(targets.cpu().tolist())
        orig_pred_list.extend(orig_pred.cpu().tolist())
        adv_pred_list.extend(adv_pred.cpu().tolist())
        s_eps_list.extend(eps_vec.cpu().tolist())
        delta_norm_list.extend(dnorm_vec.cpu().tolist())
        global_idx += batch_size

    print("Total time to compute adv set: ", time.time() - start_time)

    smallest_eps = os.path.join(outdir, f"smallest_eps_{args.attack}.csv")
    df = pd.DataFrame(
        {
            "idx": idx_list,
            "label": label_list,
            "orig_pred": orig_pred_list,
            "adv_pred": adv_pred_list,
            "smallest_eps": s_eps_list,
            "delta_norm": delta_norm_list,
        }
    )
    df.to_csv(smallest_eps, index=False)

    advset = simpleDataset(adv_img_list, adv_pred_list, transform=transform)
    print("length of advset: ", len(advset))
    return advset, df

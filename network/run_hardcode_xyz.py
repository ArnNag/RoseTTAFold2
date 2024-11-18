from icecream import ic
import torch
import os

import util
from predict import Predictor, pae_unbin
from parsers import parse_a3m, parse_pdb_w_seq, read_template_pdb
import numpy as np
from torch import nn
from chemical import INIT_CRDS
from kinematics import xyz_to_t2d

use_template = True
pdb_name = "globin"
a3m_name = "globin" # only used to get sequence

model = os.path.dirname(__file__) + "/weights/RF2_jan24.pt"
pred = Predictor(model, torch.device("cuda:0"))

pred.model.eval()

pred.xyz_converter = pred.xyz_converter.to(pred.device)
pred.lddt_bins = pred.lddt_bins.to(pred.device)

a3m = f"a3m/{a3m_name}.a3m"
msa, ins, Ls = parse_a3m(a3m)
msa = torch.tensor(msa).long()
ins = torch.tensor(ins).long()
nseqs = 256
actual_seq = msa[0].unsqueeze(0)

pdb_path = f"pdb/{pdb_name}.pdb"
L = Ls[0]

splits_with_ends = [0, 100, L]
new_mask_by_split = torch.array([False, True, False])

xyz_t, t1d, _ = read_template_pdb(L, pdb_path, align_conf=1.0)
new_mask = torch.full((xyz_t.shape[1], 27, 3), True)
for split_idx in range(len(splits_with_ends) - 1):
    start_idx = splits_with_ends[split_idx]
    end_idx = splits_with_ends[split_idx + 1]
    new_mask[start_idx:end_idx, :] = new_mask_by_split[split_idx]

xyz_t = util.realign_missing(xyz_t[0, :, :, :], new_mask, sigma=1e-2)[None, None, :, :, :].to(pred.device)
mask_t = new_mask[None, None, :, :].to(pred.device) # (B, T, L, A)
t1d = t1d.unsqueeze(0).to(pred.device)
mask_t_2d = mask_t[:, :, :, :3].all(dim=-1)  # (B, T, L)
mask_t_2d = mask_t_2d[:, :, None] * mask_t_2d[:, :, :, None]  # (B, T, L, L)
t2d = xyz_to_t2d(xyz_t, mask_t_2d).half()
seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
alpha, _, alpha_mask, _ = pred.xyz_converter.get_torsions(
    xyz_t.reshape(-1, L, 27, 3), seq_tmp, mask_in=mask_t.reshape(-1, L, 27)
)
xyz_t = xyz_t[:, :, :, 1]
alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[..., 0]))

alpha[torch.isnan(alpha)] = 0.0
alpha = alpha.reshape(1, -1, L, 10, 2)
alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 3 * 10)

msa_seed = torch.zeros(1, nseqs, L, 48).to(pred.device)
msa_full = torch.zeros(1, nseqs, L, 48).to(pred.device)
msa_extra = torch.zeros(1, nseqs, L, 25).to(pred.device)
seq_input = torch.zeros_like(actual_seq).to(pred.device)
idx = torch.arange(L)[None, :].to(pred.device)

with torch.cuda.amp.autocast(True):
    (
        logit_s,
        _,
        _,
        logits_pae,
        p_bind,
        xyz_prev,
        alpha,
        symmsub,
        pred_lddt,
        msa_prev,
        pair_prev,
        state_prev,
    ) = pred.model(
        msa_latent=msa_seed,
        msa_full=msa_extra,
        seq=seq_input,
        xyz=new_xyz,
        idx=idx,
        t1d=t1d,
        t2d=t2d,
        xyz_t=xyz_t,
        alpha_t=alpha_t,
        mask_t=mask_t_2d,
        same_chain=None,
        msa_prev=None,
        pair_prev=None,
        state_prev=None,
        mask_recycle=None,
        topk_crop=1538,
        symmids=None,
        symmsub=None,
        symmRs=None,
        symmmeta=None,
        striping=None,
    )
    xyz_prev = xyz_prev[-1]
    alpha = alpha[-1]
    _, xyz_prev = pred.xyz_converter.compute_all_atom(actual_seq, xyz_prev, alpha)

pred_lddt = nn.Softmax(dim=1)(pred_lddt.half()) * pred.lddt_bins[None, :, None]
pred_lddt = pred_lddt.sum(dim=1)
logits_pae = pae_unbin(logits_pae.half())

print(
    f"plddt {pred_lddt.mean():.3f} pae {logits_pae.mean():.3f}"
)
util.writepdb(f"hardcode_{pdb_name}.pdb", xyz_prev[0], actual_seq[0], Ls, bfacts=100 * pred_lddt[0])

torch.cuda.empty_cache()

np.savez_compressed(
    f"{pdb_name}_hardcoded",
    lddt=pred_lddt[0].detach().cpu().numpy().astype(np.float16),
    pae=logits_pae[0].detach().cpu().numpy().astype(np.float16),
)

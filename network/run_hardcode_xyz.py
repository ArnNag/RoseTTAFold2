import torch
import os
from predict import Predictor, pae_unbin
from parsers import parse_a3m, parse_pdb_w_seq, read_template_pdb
import numpy as np
from torch import nn
from chemical import INIT_CRDS
from kinematics import xyz_to_t2d

use_template = True
pdb_name = "atpbind"
a3m_name = "atpbind_atom" # only used to get sequence
model = os.path.dirname(__file__) + "/weights/RF2_jan24.pt"
pred = Predictor(model, torch.device("cuda:0"))

pred.model.eval()

pred.xyz_converter = pred.xyz_converter.to(pred.device)

a3m = f"a3m/{a3m_name}.a3m"
msa, ins, Ls = parse_a3m(a3m)
msa = torch.tensor(msa).long()
ins = torch.tensor(ins).long()
nseqs = 256
actual_seq = msa[0].unsqueeze(0)

pdb_path = f"pdb/{pdb_name}.pdb"
new_xyz = torch.from_numpy(
    parse_pdb_w_seq(pdb_path)[
        0
    ]
).to(pred.device).unsqueeze(0)
L = Ls[0]

if use_template:
    xyz_t, t1d, mask_t = read_template_pdb(L, pdb_path, align_conf=1.0)
    mask_t = mask_t.unsqueeze(0)
    mask_t_2d = mask_t[:, :, :, :3].all(dim=-1)  # (B, T, L)
    mask_t_2d = mask_t_2d[:, :, None] * mask_t_2d[:, :, :, None]  # (B, T, L, L)
    t2d = xyz_to_t2d(xyz_t, mask_t_2d).half()
    xyz_t = xyz_t.to(pred.device)
    t1d = t1d.to(pred.device)
else:
    # dummy template
    n_templ = 1
    SYMM_OFFSET_SCALE = 1.0
    xyz_t = (
            INIT_CRDS.reshape(1, 1, 27, 3).repeat(n_templ, L, 1, 1)
            + torch.rand(n_templ, L, 1, 3) * 5.0
            - 2.5
            + L ** (1 / 2)  # note: offset based on symmgroup
    )
    xyz_t = xyz_t.float().unsqueeze(0)
    xyz_t = xyz_t[:, :, :, 1].to(pred.device)
    t1d = torch.zeros(1, 1, L, 22).to(pred.device)
    t2d = torch.zeros(1, 1, L, L, 44).to(pred.device)
    mask_t = torch.zeros(1, 1, L, L).to(pred.device)

L = Ls[0]
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
        alpha_t=None,
        mask_t=mask_t,
        same_chain=None,
        msa_prev=None,
        pair_prev=None,
        state_prev=None,
        mask_recycle=None,
        symmids=None,
        symmsub=None,
        symmRs=None,
        symmmeta=None,
        striping=None,
    )
    xyz_prev = xyz_prev[-1].to(actual_seq.device)
    _, xyz_prev = pred.xyz_converter.compute_all_atom(actual_seq, xyz_prev, alpha)

pred_lddt = nn.Softmax(dim=1)(pred_lddt.half()) * pred.lddt_bins[None, :, None]
pred_lddt = pred_lddt.sum(dim=1)
logits_pae = pae_unbin(logits_pae.half())

print(
    f"plddt {pred_lddt.mean():.3f} pae {logits_pae.mean():.3f}"
)

torch.cuda.empty_cache()

np.savez_compressed(
    f"{pdb_name}_hardcoded",
    lddt=pred_lddt[0].detach().cpu().numpy().astype(np.float16),
    pae=logits_pae[0].detach().cpu().numpy().astype(np.float16),
)

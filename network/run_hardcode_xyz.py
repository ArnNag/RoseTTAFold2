import torch
import os
from predict import Predictor, pae_unbin
from parsers import parse_a3m, parse_pdb_w_seq
from featurizing import MSAFeaturize
import numpy as np
from torch import nn

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
seq, _, _, _, _ = MSAFeaturize(
    msa,
    ins,
    p_mask=0.0,
    params={"MAXLAT": nseqs, "MAXSEQ": 2048, "MAXCYCLE": 1},
)
seq = seq.unsqueeze(0)

new_xyz = torch.from_numpy(
    parse_pdb_w_seq(f"pdb/{pdb_name}.pdb")[
        0
    ]
).to(pred.device).unsqueeze(0)

L = Ls[0]
msa_seed = torch.zeros(1, nseqs, L, 48).to(pred.device)
msa_full = torch.zeros(1, nseqs, L, 48).to(pred.device)
msa_extra = torch.zeros(1, nseqs, L, 25).to(pred.device)
seq_input = torch.zeros_like(seq).to(pred.device)
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
        t1d=None,
        t2d=None,
        xyz_t=None,
        alpha_t=None,
        mask_t=None,
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
    xyz_prev = xyz_prev[-1].to(seq.device)
    _, xyz_prev = pred.xyz_converter.compute_all_atom(seq, xyz_prev, alpha)

mask_recycle = None
pair_prev = pair_prev.cpu()
msa_prev = msa_prev.cpu()

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

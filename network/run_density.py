import torch
from predict import Predictor, merge_a3m_homo, get_striping_parameters, pae_unbin
from symmetry import symm_subunit_matrix, find_symm_subs
from chemical import INIT_CRDS
from parsers import parse_a3m, parse_pdb_w_seq
from data_loader import merge_a3m_hetero
from kinematics import xyz_to_t2d
import util
import numpy as np
from torch import nn
import os
from icecream import ic

torch.backends.cuda.preferred_linalg_library(
    backend="magma"
)  # avoid issue with cuSOLVER when computing SVD
(use_template, use_xyz_prev, use_state_prev, use_pair_prev, a3m_name, map_name) = (
    False,
    True,
    False,
    False,
    "atpbind",
    "emd_14914",
)

def nan_check_hook(module, inputs):
    def check_tensor(tensor, name):
        if isinstance(tensor, torch.Tensor):
            if tensor.shape[-2:] == torch.Size([27, 3]):
                # positions tensor contains NaNs for undefined atoms. don't want to error on these.
                if torch.isnan(tensor[...,1,:]).any():  # first atom
                    raise RuntimeError(f"NaN detected in first atom of {name} to {type(module).__name__}")
            elif torch.isnan(tensor).any():
                raise RuntimeError(f"NaN detected in {name} to {type(module).__name__}")

    if isinstance(inputs, tuple):
        for idx, input_tensor in enumerate(inputs):
            check_tensor(input_tensor, f"input[{idx}]")
    else:
        check_tensor(inputs, "input")

model = os.path.dirname(__file__) + "/weights/RF2_jan24.pt"
pred = Predictor(model, torch.device("cuda:0"))

for name, module in pred.model.named_modules():
    if not isinstance(module, torch.jit.ScriptModule):
        module.register_forward_pre_hook(nan_check_hook)

symm = "C1"
nseqs_full = 2048
n_templ = 1
n_recycles = 3
nseqs = 256
subcrop = -1
topk = -1
low_vram = False
B = 1
msa_concat_mode = "diag"
pred.xyz_converter = pred.xyz_converter.cpu()
out_prefix = f"test_predict_{a3m_name}_{map_name}"

###
# pass 1, combined MSA
a3m = f"a3m/{a3m_name}.a3m"
msa, ins, Ls = parse_a3m(a3m)
msa_orig = torch.tensor(msa).long()
ins_orig = torch.tensor(ins).long()

symmids, symmRs, symmmeta, symmoffset = symm_subunit_matrix(symm)

###
# pass 2, templates
L = sum(Ls)

# dummy template
SYMM_OFFSET_SCALE = 1.0
xyz_t = (
    INIT_CRDS.reshape(1, 1, 27, 3).repeat(n_templ, L, 1, 1)
    + torch.rand(n_templ, L, 1, 3) * 5.0
    - 2.5
    + SYMM_OFFSET_SCALE * symmoffset * L ** (1 / 2)  # note: offset based on symmgroup
)


mask_t = torch.full((n_templ, L, 27), False)
t1d = torch.nn.functional.one_hot(
    torch.full((n_templ, L), 20).long(), num_classes=21
).float()  # all gaps
t1d = torch.cat((t1d, torch.zeros((n_templ, L, 1)).float()), -1)

maxtmpl = 1

same_chain = torch.full((1, L, L), True, dtype=torch.bool, device=xyz_t.device)

# template features
xyz_t = xyz_t[:maxtmpl].float().unsqueeze(0)
mask_t = mask_t[:maxtmpl].unsqueeze(0)
t1d = t1d[:maxtmpl].float().unsqueeze(0)

seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
alpha, _, alpha_mask, _ = pred.xyz_converter.get_torsions(
    xyz_t.reshape(-1, L, 27, 3), seq_tmp, mask_in=mask_t.reshape(-1, L, 27)
)
alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[..., 0]))

alpha[torch.isnan(alpha)] = 0.0
alpha = alpha.reshape(1, -1, L, 10, 2)
alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 3 * 10)

###
# pass 3, symmetry
xyz_prev = xyz_t[:, 0]
xyz_prev, symmsub = find_symm_subs(xyz_prev[:, :L], symmRs, symmmeta)

Osub = symmsub.shape[0]
mask_t = mask_t.repeat(1, 1, Osub, 1)
alpha_t = alpha_t.repeat(1, 1, Osub, 1)
mask_prev = mask_t[:, 0]
xyz_t = xyz_t.repeat(1, 1, Osub, 1, 1)
t1d = t1d.repeat(1, 1, Osub, 1)

# symmetrize msa
if Osub > 1:
    msa_orig, ins_orig = merge_a3m_homo(msa_orig, ins_orig, Osub, mode=msa_concat_mode)

# index
idx_pdb = torch.arange(Osub * L)[None, :]

same_chain = torch.zeros((1, Osub * L, Osub * L)).long()
i_start = 0
for o_i in range(Osub):
    i_stop = i_start + L
    idx_pdb[:, i_stop:] += 100
    same_chain[:, i_start:i_stop, i_start:i_stop] = 1
    i_start = i_stop

mask_t_2d = mask_t[:, :, :, :3].all(dim=-1)  # (B, T, L)
mask_t_2d = mask_t_2d[:, :, None] * mask_t_2d[:, :, :, None]  # (B, T, L, L)
mask_t_2d = (
    mask_t_2d.float() * same_chain.float()[:, None]
)  # (ignore inter-chain region)

pred.model.eval()

pred.xyz_converter = pred.xyz_converter.to(pred.device)
pred.lddt_bins = pred.lddt_bins.to(pred.device)

STRIPE = get_striping_parameters(low_vram)

with torch.no_grad():
    msa = msa_orig.long().to(pred.device)  # (N, L)
    ins = ins_orig.long().to(pred.device)

    print(f"N={msa.shape[0]} L={msa.shape[1]}")

    #
    t1d = t1d.to(pred.device).half()
    t2d = xyz_to_t2d(xyz_t, mask_t_2d).half()
    if not low_vram:
        t2d = t2d.to(pred.device)  # .half()
    idx_pdb = idx_pdb.to(pred.device)
    xyz_t = xyz_t[:, :, :, 1].to(pred.device)
    mask_t_2d = mask_t_2d.to(pred.device)
    alpha_t = alpha_t.to(pred.device)
    xyz_prev = xyz_prev.to(pred.device)
    mask_prev = mask_prev.to(pred.device)
    same_chain = same_chain.to(pred.device)
    symmids = symmids.to(pred.device)
    symmsub = symmsub.to(pred.device)
    symmRs = symmRs.to(pred.device)

    subsymms, _ = symmmeta
    for i in range(len(subsymms)):
        subsymms[i] = subsymms[i].to(pred.device)

    msa_prev = None
    pair_prev = None
    state_prev = None
    mask_recycle = mask_prev[:, :, :3].bool().all(dim=-1)
    mask_recycle = mask_recycle[:, :, None] * mask_recycle[:, None, :]  # (B, L, L)
    mask_recycle = same_chain.float() * mask_recycle.float()

    best_lddt = torch.tensor([-1.0], device=pred.device)
    best_xyz = None
    best_logit = None
    best_pae = None

    if map_name is not None:
        from pyrosetta import init, rosetta
        from density import setup_docking_mover

        init(
            "-beta -crystal_refine -mute core -unmute core.scoring.electron_density -multithreading:total_threads 4"
        )
        dock_into_dens: (
            rosetta.protocols.electron_density.DockFragmentsIntoDensityMover
        ) = setup_docking_mover(counts=1)

    for i_cycle in range(n_recycles + 1):
        from featurizing import MSAFeaturize

        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
            msa,
            ins,
            p_mask=0.0,
            params={"MAXLAT": nseqs, "MAXSEQ": nseqs_full, "MAXCYCLE": 1},
        )

        seq = seq.unsqueeze(0)
        msa_seed = msa_seed.unsqueeze(0)
        msa_extra = msa_extra.unsqueeze(0)

        # fd memory savings
        msa_seed = msa_seed.half()  # GPU ONLY
        msa_extra = msa_extra.half()  # GPU ONLY

        xyz_prev_prev = xyz_prev.clone()

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
                msa_seed,
                msa_extra,
                seq,
                xyz_prev,
                idx_pdb,
                t1d=t1d,
                t2d=t2d,
                xyz_t=xyz_t,
                alpha_t=alpha_t,
                mask_t=mask_t_2d,
                same_chain=same_chain,
                msa_prev=msa_prev,
                pair_prev=pair_prev,
                state_prev=state_prev,
                p2p_crop=subcrop,
                topk_crop=topk,
                mask_recycle=mask_recycle,
                symmids=symmids,
                symmsub=symmsub,
                symmRs=symmRs,
                symmmeta=symmmeta,
                striping=STRIPE,
            )
            alpha = alpha[-1].to(seq.device)
            xyz_prev = xyz_prev[-1].to(seq.device)
            _, xyz_prev = pred.xyz_converter.compute_all_atom(seq, xyz_prev, alpha)

        mask_recycle = None
        pair_prev = pair_prev.cpu()
        msa_prev = msa_prev.cpu()

        pred_lddt = nn.Softmax(dim=1)(pred_lddt.half()) * pred.lddt_bins[None, :, None]
        pred_lddt = pred_lddt.sum(dim=1)
        logits_pae = pae_unbin(logits_pae.half())

        # TODO: what is the point of the new singleton dimension (N) in xyz_prev_prev[None]?

        print(
            f"recycle {i_cycle} plddt {pred_lddt.mean():.3f} pae {logits_pae.mean():.3f} rmsd: TODO"
        )

        torch.cuda.empty_cache()
        if pred_lddt.mean() < best_lddt.mean():
            # TODO: are B-factors modified during the docking process? should we use these instead of pLDDT?
            pred_lddt, logits_pae, logit_s = None, None, None
            continue

        best_xyz = xyz_prev
        best_logit = logit_s
        best_lddt = pred_lddt.half().cpu()
        best_pae = logits_pae.half().cpu()
        best_logit = [l.half().cpu() for l in logit_s]
        logits_pae, logit_s = None, None

        if map_name is not None:
            from pyrosetta import rosetta, Pose, pose_from_pdb
            from density import split_by_pae
            import shutil

            mapfile = f"map/{map_name}.map"
            rosetta.core.scoring.electron_density.getDensityMap(mapfile)
            new_xyz = torch.zeros_like(xyz_prev)
            splits: list[int] = split_by_pae(best_pae[0], min_split_length=100)
            ic(splits)
            splits.insert(0, 0)
            for split in range(len(splits) - 1):
                start_idx = splits[split]
                end_idx = splits[split + 1]
                before_dock_file = f"test_{a3m_name}_{map_name}_before_dock_cycle_{i_cycle}_split_{split}.pdb"
                util.writepdb(
                    before_dock_file,
                    xyz_prev[0][start_idx:end_idx],
                    seq[0][start_idx:end_idx],
                    [end_idx - start_idx],
                    bfacts=100 * pred_lddt[0],
                )
                pose_before_fit: Pose = pose_from_pdb(before_dock_file)
                dock_into_dens.apply(pose_before_fit)
                after_dock_file = f"test_{a3m_name}_{map_name}_after_dock_cycle_{i_cycle}_split_{split}.pdb"
                shutil.copyfile("EMPTY_JOB_use_jd2_000001.pdb", after_dock_file)
                new_xyz[0][start_idx:end_idx] = torch.from_numpy(
                    parse_pdb_w_seq(after_dock_file)[0]
                )

        else:
            # hard-code the new_xyz based on a provided PDB file instead of doing density fitting
            # TODO: allow a structure other than myoglobin
            new_xyz = torch.from_numpy(
                parse_pdb_w_seq("pdb/rotated_structures/rotated_alpha000_beta000.pdb")[
                    0
                ]
            ).unsqueeze(0)

        pred_lddt = None

        # xyz_globin_masked_centered_realigned = util.realign_missing(new_xyz[0, :, :, :], new_mask_t[0, 0, :, :], sigma=1e-1).unsqueeze(0)
        if use_template:
            xyz_t = new_xyz[None, :, 1, :].to(xyz_t)
        if use_xyz_prev:
            xyz_prev = new_xyz.to(xyz_prev)
        if not use_pair_prev:
            pair_prev = torch.zeros_like(pair_prev)
        if not use_state_prev:
            state_prev = torch.zeros_like(state_prev)

    # free more memory
    pair_prev, msa_prev, t2d = None, None, None

    prob_s = list()
    for logit in best_logit:
        prob = pred.active_fn(logit.to(pred.device).float())  # distogram
        prob_s.append(prob.half().cpu())

best_xyz = best_xyz.float().cpu()
outdata = {}

# RMS
outdata["mean_plddt"] = best_lddt.mean().item()
Lstarti = 0
for i, li in enumerate(Ls):
    Lstartj = 0
    for j, lj in enumerate(Ls):
        if j > i:
            outdata["pae_chain_" + str(i) + "_" + str(j)] = (
                0.5
                * (
                    best_pae[
                        :, Lstarti : (Lstarti + li), Lstartj : (Lstartj + lj)
                    ].mean()
                    + best_pae[
                        :, Lstartj : (Lstartj + lj), Lstarti : (Lstarti + li)
                    ].mean()
                ).item()
            )
        Lstartj += lj
    Lstarti += li

outfile = f"{out_prefix}_{use_template=}_{use_xyz_prev=}_{use_state_prev=}_{use_pair_prev=}_pred.pdb"
util.writepdb(outfile, best_xyz[0], seq[0], Ls, bfacts=100 * best_lddt[0])

prob_s = [
    prob.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float16)
    for prob in prob_s
]
np.savez_compressed(
    f"{out_prefix}_{use_template=}_{use_xyz_prev=}_{use_state_prev=}_{use_pair_prev=}",
    dist=prob_s[0].astype(np.float16),
    lddt=best_lddt[0].detach().cpu().numpy().astype(np.float16),
    pae=best_pae[0].detach().cpu().numpy().astype(np.float16),
)

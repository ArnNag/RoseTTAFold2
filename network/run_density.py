import glob

import torch
from predict import Predictor, merge_a3m_homo, get_striping_parameters, pae_unbin
from chemical import INIT_CRDS
from parsers import parse_a3m, parse_pdb_w_seq, read_template_pdb
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
(use_template, use_xyz_prev, use_state_prev, use_pair_prev, use_msa, a3m_name, map_name, pdb_name) = (
    True,
    True,
    True,
    False,
    True,
    "atpbind_atom",
    "emd_14914",
    None
)

assert (map_name is None) + (pdb_name is None) == 1

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

# for name, module in pred.model.named_modules():
#     if not isinstance(module, torch.jit.ScriptModule):
#         module.register_forward_pre_hook(nan_check_hook)

symm = "C1"
nseqs_full = 2048
n_templ = 1
n_recycles = 3
nseqs = 256
subcrop = -1
topk = 1536
B = 1
pred.xyz_converter = pred.xyz_converter.cpu()
out_prefix = f"test_predict_{a3m_name}_{f'map_{map_name}' if map_name is not None else f'pdb_{pdb_name}'}_pdb_{pdb_name}_{use_template=}_{use_xyz_prev=}_{use_state_prev=}_{use_pair_prev=}_{use_msa=}"

###
# pass 1, combined MSA
a3m = f"a3m/{a3m_name}.a3m"
msa, ins, Ls = parse_a3m(a3m)
msa_orig = torch.tensor(msa).long()
ins_orig = torch.tensor(ins).long()

###
# pass 2, templates
L = sum(Ls)

# dummy template
SYMM_OFFSET_SCALE = 1.0
xyz_t = (
    INIT_CRDS.reshape(1, 1, 27, 3).repeat(n_templ, L, 1, 1)
    + torch.rand(n_templ, L, 1, 3) * 5.0
    - 2.5
    + L ** (1 / 2)  # note: offset based on symmgroup
)


mask_t = torch.full((n_templ, L, 27), False)
t1d = torch.nn.functional.one_hot(
    torch.full((n_templ, L), 20).long(), num_classes=21
).float()  # all gaps
t1d = torch.cat((t1d, torch.zeros((n_templ, L, 1)).float()), -1)

maxtmpl = 1

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
xyz_prev = xyz_t[:, 0].to(pred.device)

mask_prev_orig = mask_t[:, 0].to(pred.device)

# index
idx_pdb = torch.arange(L)[None, :]

mask_t_2d = mask_t[:, :, :, :3].all(dim=-1)  # (B, T, L)
mask_t_2d = mask_t_2d[:, :, None] * mask_t_2d[:, :, :, None]  # (B, T, L, L)

pred.model.eval()

pred.xyz_converter = pred.xyz_converter.to(pred.device)
pred.lddt_bins = pred.lddt_bins.to(pred.device)

with torch.no_grad():
    msa = msa_orig.long().to(pred.device)  # (N, L)
    ins = ins_orig.long().to(pred.device)

    print(f"N={msa.shape[0]} L={msa.shape[1]}")

    #
    t1d = t1d.to(pred.device).half()
    t2d = xyz_to_t2d(xyz_t, mask_t_2d).half()
    t2d = t2d.to(pred.device)  # .half()
    idx_pdb = idx_pdb.to(pred.device)
    xyz_t = xyz_t[:, :, :, 1].to(pred.device)
    mask_t_2d = mask_t_2d.to(pred.device)
    alpha_t = alpha_t.to(pred.device)
    mask_prev = mask_prev_orig.clone()

    msa_prev = None
    pair_prev = None
    state_prev = None

    best_lddt = torch.tensor([-1.0], device=pred.device)
    best_xyz = None
    best_logit = None
    best_pae = None

    if map_name is not None:
        from pyrosetta import init, rosetta
        from density import setup_docking_mover

        init(
            "-beta -crystal_refine -mute core -multithreading:total_threads 4"
        )
        dock_into_dens: (
            rosetta.protocols.electron_density.DockFragmentsIntoDensityMover
        ) = setup_docking_mover(counts=1)

    for i_cycle in range(n_recycles + 1):

        mask_recycle = mask_prev[:, :, :3].bool().all(dim=-1)
        mask_recycle = mask_recycle[:, :, None] * mask_recycle[:, None, :]  # (B, L, L)
        mask_recycle = mask_recycle.float()

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
                _,
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
                same_chain=None,
                msa_prev=msa_prev,
                pair_prev=pair_prev,
                state_prev=state_prev,
                p2p_crop=subcrop,
                topk_crop=topk,
                mask_recycle=mask_recycle,
                symmids=None,
                symmsub=None,
                symmRs=None,
                symmmeta=None,
                striping=None,
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
            f"recycle {i_cycle} plddt {pred_lddt.mean():.3f} pae {logits_pae.mean():.3f}"
        )

        torch.cuda.empty_cache()

        best_xyz = xyz_prev
        best_logit = logit_s
        best_lddt = pred_lddt.half().cpu()
        best_pae = logits_pae.half().cpu()
        best_logit = [l.half().cpu() for l in logit_s]
        logits_pae, logit_s = None, None
        metrics_file = f"test_{a3m_name}_{map_name}_cycle_{i_cycle}"
        np.savez_compressed(
            metrics_file,
            lddt=best_lddt[0].detach().cpu().numpy().astype(np.float16),
            pae=best_pae[0].detach().cpu().numpy().astype(np.float16),
        )

        if map_name is not None and i_cycle == 0:
            from pyrosetta import rosetta, Pose, pose_from_pdb
            from density import split_by_pae
            import shutil

            mapfile = f"map/{map_name}.map"
            rosetta.core.scoring.electron_density.getDensityMap(mapfile)
            new_xyz = torch.zeros_like(xyz_prev)
            splits: list[int] = split_by_pae(best_pae[0].to(torch.float32), min_split_length=100)
            print(f"{splits=}")
            splits_with_ends = [0]
            splits_with_ends.extend(splits)
            splits_with_ends.append(len(best_pae[0]))
            for split_idx in range(len(splits_with_ends) - 1):
                start_idx = splits_with_ends[split_idx]
                end_idx = splits_with_ends[split_idx + 1]
                print(f"{split_idx=}")
                print(f"{start_idx=}")
                print(f"{end_idx=}")
                before_dock_file = f"test_{a3m_name}_{map_name}_before_dock_cycle_{i_cycle}_split_{split_idx}.pdb"
                util.writepdb(
                    before_dock_file,
                    xyz_prev[0, start_idx:end_idx],
                    seq[0, start_idx:end_idx],
                    [end_idx - start_idx],
                    bfacts=100 * pred_lddt[0, start_idx:end_idx],
                )
                pose_before_fit: Pose = pose_from_pdb(before_dock_file)
                dock_into_dens.apply(pose_before_fit)
                after_dock_file = f"test_{a3m_name}_{map_name}_after_dock_cycle_{i_cycle}_split_{split_idx}.pdb"
                shutil.copyfile("EMPTY_JOB_use_jd2_000001.pdb", after_dock_file)

                # grab top 'count' poses
                allfiles: list[str] = glob.glob('EMPTY_JOB_use_jd2_*.pdb')
                allfiles.sort()
                allfiles.pop(0)
                for j, file in enumerate(allfiles):
                    hit = j + 1
                    next_best_hit_file = f"test_{a3m_name}_{map_name}_after_dock_cycle_{i_cycle}_split_{split_idx}_hit_{hit}.pdb"
                    shutil.copyfile(file, next_best_hit_file)
                new_xyz[0, start_idx:end_idx] = torch.from_numpy(
                    parse_pdb_w_seq(after_dock_file)[0]
                )

            long_jump_threshold = 10.
            is_long_jump = torch.full((len(splits) + 2,), True, dtype=torch.bool)
            # default to True for either end since we want to mask fragments on the ends if the only jump they touch
            # is longer than long_jump_threshold
            for split_idx, split_pt in enumerate(splits, start=1):
                assert split_pt >= 1
                jump_dist = torch.norm(new_xyz[0, split_pt, 0] - new_xyz[0, split_pt - 1, 0])
                print(f"{jump_dist=}")
                is_long_jump[split_idx] = jump_dist > long_jump_threshold

            new_mask = torch.full_like(mask_prev_orig, True, dtype=torch.bool)
            for split_idx in range(len(splits_with_ends) - 1):
                start_idx = splits_with_ends[split_idx]
                end_idx = splits_with_ends[split_idx + 1]
                if is_long_jump[split_idx] and is_long_jump[split_idx + 1]:
                    new_mask[0, start_idx:end_idx] = False

            new_xyz = util.realign_missing(new_xyz[0, :, :, :], new_mask[0, :, :], sigma=1e-1).unsqueeze(0)

            new_pdb_path = f"new_xyz_cycle_{i_cycle}.pdb"
            util.writepdb(
                new_pdb_path,
                new_xyz[0],
                seq[0],
                Ls,
                bfacts=100 * pred_lddt[0],
            )

        if pdb_name is not None:
            # hard-code the new_xyz based on a provided PDB file instead of doing density fitting
            new_pdb_path = f"pdb/{pdb_name}.pdb"
            new_xyz = torch.from_numpy(
                parse_pdb_w_seq(new_pdb_path)[
                    0
                ]
            ).to(xyz_prev).unsqueeze(0)

        pred_lddt = None

        if i_cycle == 0:
            if use_template:
                xyz_t, t1d, mask_t = read_template_pdb(L, new_pdb_path, align_conf=1.0)
                xyz_t = xyz_t.unsqueeze(0).to(pred.device)
                mask_t = mask_t.unsqueeze(0).to(pred.device)
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
            if use_xyz_prev:
                xyz_prev = new_xyz
            if not use_pair_prev:
                pair_prev = torch.zeros_like(pair_prev)
            if not use_state_prev:
                state_prev = torch.zeros_like(state_prev)
            if not use_msa:
                msa_seed = torch.zeros_like(msa_seed)
                msa_extra = torch.zeros_like(msa_extra)
                msa_prev = torch.zeros_like(msa_prev)
                seq = torch.zeros_like(seq)

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

util.writepdb(f"{out_prefix}.pdb", best_xyz[0], seq[0], Ls, bfacts=100 * best_lddt[0])

prob_s = [
    prob.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float16)
    for prob in prob_s
]
np.savez_compressed(
    out_prefix,
    dist=prob_s[0].astype(np.float16),
    lddt=best_lddt[0].detach().cpu().numpy().astype(np.float16),
    pae=best_pae[0].detach().cpu().numpy().astype(np.float16),
)

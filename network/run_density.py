import glob

import torch
from predict import Predictor, pae_unbin
from chemical import INIT_CRDS
from parsers import parse_a3m, parse_pdb_w_seq, parse_pdb_w_b_factor
from kinematics import xyz_to_t2d
import util
import numpy as np
from torch import nn
import os
from icecream import ic
from datetime import datetime
from pathlib import Path

a3m_name = "atpbind"
map_name = None
pdb_name = "atpbind"
replace_template = True
replace_xyz_prev = True
use_state_prev = True
use_pair_prev = False
use_msa = True
use_msa_prev = True
prediction_pdb = None
freeze_masked_msa2pair = True

n_recycles = 4
dock_cycle = 3
plddt_cutoff = 0.4
fit_score_threshold = 1.0
long_jump_threshold = 45.
clash_threshold = 0.5
temp_conf = 0.8
min_split_length = 80

assert (map_name is None) + (pdb_name is None) == 1, "Either a map or a pdb file must be specified."

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
out_dir = f"{dt_string}"
Path(out_dir).mkdir()
hyperparams = f"{replace_template=}\n{replace_xyz_prev=}\n{use_state_prev=}\n{use_pair_prev=}\n{use_msa=}\n{use_msa_prev=}\n{a3m_name=}\n{map_name=}\n{pdb_name=}\n{prediction_pdb=}\n{n_recycles=}\n{dock_cycle=}\n{plddt_cutoff=}\n{fit_score_threshold=}\n{long_jump_threshold=}\n{clash_threshold=}\n{temp_conf=}\n{min_split_length=}\n{freeze_masked_msa2pair=}"
with open(f"{out_dir}/hyperparams.txt", "w") as f:
    f.write(hyperparams)

model = os.path.dirname(__file__) + "/weights/RF2_jan24.pt"
pred = Predictor(model, torch.device("cuda:0"))

nseqs_full = 2048
nseqs = 256
topk = 1536
pred.xyz_converter = pred.xyz_converter.cpu()

###
# pass 1, combined MSA
a3m = f"a3m/{a3m_name}.a3m"
msa, ins, Ls = parse_a3m(a3m)
msa_orig = torch.tensor(msa).long()
ins_orig = torch.tensor(ins).long()

###
# pass 2, templates
n_templ = 1
L = sum(Ls)

# dummy template
xyz_t = (
        INIT_CRDS.reshape(1, 1, 27, 3).repeat(n_templ, L, 1, 1)
        + torch.rand(n_templ, L, 1, 3) * 5.0
        - 2.5 + L ** (1 / 2)
).float()

mask_t = torch.full((n_templ, L, 27), False)
mask_t_2d = mask_t[:, :, :3].all(dim=-1)  # (T, L)
mask_t_2d = mask_t_2d[:, None] * mask_t_2d[:, :, None]  # (T, L, L)

t1d = torch.nn.functional.one_hot(
    torch.full((n_templ, L), 20).long(), num_classes=21
).float()  # all gaps
t1d = torch.cat((t1d, torch.zeros((n_templ, L, 1)).float()), -1)

# template features
t1d = t1d.float()

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
xyz_prev = xyz_t[0, :, :, :].to(pred.device)  # select the 0th template

# index
idx_pdb = torch.arange(L)[None, :]

pred.model.eval()
pred.xyz_converter = pred.xyz_converter.to(pred.device)
pred.lddt_bins = pred.lddt_bins.to(pred.device)

with torch.no_grad():
    msa = msa_orig.long().to(pred.device)  # (N, L)
    ins = ins_orig.long().to(pred.device)

    print(f"N={msa.shape[0]} L={msa.shape[1]}")

    t1d = t1d.to(pred.device).half()
    t2d = xyz_to_t2d(xyz_t.unsqueeze(0), mask_t_2d.unsqueeze(0)).half()
    t2d = t2d.to(pred.device)  # .half()
    idx_pdb = idx_pdb.to(pred.device)
    xyz_t = xyz_t[:, :, 1, :].to(pred.device)  # select alpha carbon
    mask_t_2d = mask_t_2d.to(pred.device)
    msa2pair_freeze_mask = torch.ones(L, L)
    alpha_t = alpha_t.to(pred.device)

    msa_prev = None
    pair_prev = None
    state_prev = None
    mask_recycle = mask_t_2d[0, :, :][None, :, :]  # replace template axis with batch axis

    from featurizing import MSAFeaturize

    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
        msa,
        ins,
        p_mask=0.0,
        params={"MAXLAT": nseqs, "MAXSEQ": nseqs_full, "MAXCYCLE": 1},
    )

    msa_seed = msa_seed.unsqueeze(0)
    msa_extra = msa_extra.unsqueeze(0)

    # fd memory savings
    msa_seed = msa_seed.half()  # GPU ONLY
    msa_extra = msa_extra.half()  # GPU ONLY

    if map_name is not None:
        from pyrosetta import init, rosetta
        from density import setup_docking_mover

        init(
            "-beta -crystal_refine -mute core -multithreading:total_threads 4"
        )
        dock_into_dens: (
            rosetta.protocols.electron_density.DockFragmentsIntoDensityMover
        ) = setup_docking_mover(counts=1)
        mapfile = f"map/{map_name}.map"
        rosetta.core.scoring.electron_density.getDensityMap(mapfile)

    for i_cycle in range(n_recycles + 1):

        if prediction_pdb is None or i_cycle > dock_cycle:
            with torch.cuda.amp.autocast(True):
                (
                    _,
                    _,
                    _,
                    logits_pae,
                    _,
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
                    seq.unsqueeze(0),
                    xyz_prev.unsqueeze(0),
                    idx_pdb,
                    t1d=t1d.unsqueeze(0),
                    t2d=t2d,
                    xyz_t=xyz_t.unsqueeze(0),
                    alpha_t=alpha_t,
                    mask_t=mask_t_2d.unsqueeze(0),
                    same_chain=None,
                    msa_prev=msa_prev,
                    pair_prev=pair_prev,
                    state_prev=state_prev,
                    p2p_crop=-1,
                    topk_crop=topk,
                    mask_recycle=mask_recycle,
                    symmids=None,
                    symmsub=None,
                    symmRs=None,
                    symmmeta=None,
                    striping=None,
                )
                alpha = alpha[-1, 0, :, :, :].to(seq.device)
                xyz_prev = xyz_prev[-1, 0, :, :, :].to(seq.device)
                _, xyz_prev = pred.xyz_converter.compute_all_atom(seq[None, :], xyz_prev[None, :, :, :],
                                                                  alpha[None, :, :, :])
                xyz_prev = xyz_prev[0, :, :, :]

            pair_prev = pair_prev.cpu()
            msa_prev = msa_prev.cpu()

            pred_lddt = nn.Softmax(dim=1)(pred_lddt.half()) * pred.lddt_bins[None, :, None]
            pred_lddt = pred_lddt.sum(dim=1)
            logits_pae = pae_unbin(logits_pae.half())

            print(
                f"recycle {i_cycle} plddt {pred_lddt.mean():.3f} pae {logits_pae.mean():.3f}"
            )

            torch.cuda.empty_cache()

            metrics_file = f"{out_dir}/cycle_{i_cycle}"

            if i_cycle == dock_cycle:
                np.savez_compressed(
                    metrics_file,
                    lddt=pred_lddt[0].detach().cpu().numpy().astype(np.float16),
                    pae=logits_pae[0].detach().cpu().numpy().astype(np.float16),
                    msa_seed=msa_seed.detach().cpu().numpy(),
                    msa_extra=msa_extra.detach().cpu().numpy(),
                    msa_prev=msa_prev.detach().cpu().numpy(),
                    pair_prev=pair_prev.detach().cpu().numpy(),
                    state_prev=state_prev.detach().cpu().numpy(),
                )
            else:
                np.savez_compressed(
                    metrics_file,
                    lddt=pred_lddt[0].detach().cpu().numpy().astype(np.float16),
                    pae=logits_pae[0].detach().cpu().numpy().astype(np.float16),
                )
            util.writepdb(f"{out_dir}/full_prediction_{i_cycle}.pdb", xyz_prev, seq, Ls, bfacts=100 * pred_lddt[0])

        else:
            xyz_prev_path = f"pdb/{prediction_pdb}.pdb"
            xyz_prev = torch.from_numpy(
                parse_pdb_w_seq(xyz_prev_path)[
                    0
                ]
            ).to(xyz_prev)

            metrics_file = f"pdb/{prediction_pdb}.npz"
            metrics = np.load(metrics_file)
            pred_lddt = metrics["lddt"].unsqueeze(0)
            logits_pae = metrics["pae"].unsqueeze(0)
            if use_msa:
                msa_seed = metrics["msa_seed"]
                msa_extra = metrics["msa_extra"]
                msa_prev = metrics["msa_prev"]
            if use_pair_prev:
                pair_prev = metrics["pair_prev"]
            if use_state_prev:
                state_prev = metrics["state_prev"]

        if i_cycle == dock_cycle:

            if map_name is not None:
                from pyrosetta import rosetta, Pose, pose_from_pdb
                from density import split_by_pae, check_clash
                import shutil

                new_xyz = torch.full_like(xyz_prev, torch.nan)
                splits: list[int] = split_by_pae(logits_pae[0].to(torch.float32), min_split_length=min_split_length)
                print(f"{splits=}")
                splits_with_ends = [0]
                splits_with_ends.extend(splits)
                splits_with_ends.append(len(logits_pae[0]))
                fit_score_by_residue = torch.full((len(xyz_prev),), torch.nan, device=new_xyz.device)
                mean_fit_score_by_split = torch.full((len(splits_with_ends) - 1,), torch.nan)
                remaining_idxs = torch.nonzero(pred_lddt[0, :] > plddt_cutoff).flatten()
                frag_remaining_start = torch.full((len(splits_with_ends) - 1,), -1, dtype=torch.int16)
                frag_remaining_end = torch.full((len(splits_with_ends) - 1,), -1, dtype=torch.int16)
                print(f"{remaining_idxs=}")
                for split_idx in range(len(splits_with_ends) - 1):
                    start_idx = splits_with_ends[split_idx]
                    end_idx = splits_with_ends[split_idx + 1]
                    print(f"{split_idx=}")
                    print(f"{start_idx=}")
                    print(f"{end_idx=}")

                    idx_into_remaining_idxs_start = torch.searchsorted(remaining_idxs, start_idx)
                    idx_into_remaining_idxs_end = torch.searchsorted(remaining_idxs, end_idx)
                    remaining_idxs_in_frag = remaining_idxs[idx_into_remaining_idxs_start:idx_into_remaining_idxs_end]
                    print(f"{remaining_idxs_in_frag=}")
                    frag_remaining_start[split_idx] = remaining_idxs_in_frag[0]
                    frag_remaining_end[split_idx] = remaining_idxs_in_frag[-1] + 1  # end index is exclusive

                    min_residues_per_dock = 20
                    if len(remaining_idxs_in_frag) < min_residues_per_dock:
                        continue

                    before_trim_file = f"{out_dir}/before_trim_cycle_{i_cycle}_split_{split_idx}.pdb"
                    util.writepdb(
                        before_trim_file,
                        xyz_prev[start_idx:end_idx, :, :],
                        seq[start_idx:end_idx],
                        [end_idx - start_idx],
                        bfacts=100 * pred_lddt[0, start_idx:end_idx],
                    )

                    before_dock_file = f"{out_dir}/before_dock_cycle_{i_cycle}_split_{split_idx}.pdb"
                    util.writepdb(
                        before_dock_file,
                        xyz_prev[remaining_idxs_in_frag],
                        seq[remaining_idxs_in_frag],
                        [len(remaining_idxs_in_frag)],
                        bfacts=100 * pred_lddt[0][remaining_idxs_in_frag],
                    )

                    pose_before_fit: Pose = pose_from_pdb(before_dock_file)
                    dock_into_dens.apply(pose_before_fit)
                    after_dock_file = f"{out_dir}/after_dock_cycle_{i_cycle}_split_{split_idx}_best.pdb"
                    shutil.copyfile("EMPTY_JOB_use_jd2_000001.pdb", after_dock_file)

                    # grab top 'count' poses
                    allfiles: list[str] = glob.glob('EMPTY_JOB_use_jd2_*.pdb')
                    allfiles.sort()
                    allfiles.pop(0)
                    for j, file in enumerate(allfiles):
                        hit = j + 1
                        next_best_hit_file = f"{out_dir}/after_dock_cycle_{i_cycle}_split_{split_idx}_hit_{hit}.pdb"
                        shutil.copyfile(file, next_best_hit_file)
                    loaded_xyz, _, _, loaded_fit_score = parse_pdb_w_b_factor(after_dock_file)
                    new_xyz[remaining_idxs_in_frag] = torch.from_numpy(loaded_xyz).to(new_xyz)
                    fit_score_by_residue[remaining_idxs_in_frag] = torch.from_numpy(loaded_fit_score).to(
                        fit_score_by_residue)
                    mean_fit_score_by_split[split_idx] = loaded_fit_score.mean()

                print(f"{frag_remaining_start=}")
                print(f"{frag_remaining_end=}")
                new_mask = torch.logical_and(~torch.isnan(new_xyz).all(dim=-1),
                                             (fit_score_by_residue > fit_score_threshold)[:, None])
                new_xyz = util.realign_missing(new_xyz, new_mask, sigma=0.)

                new_mask_by_split = torch.full((len(splits_with_ends) - 1,), True)

                is_long_jump = torch.full((len(splits) + 2,), True, dtype=torch.bool)
                # default to True for either end since we want to mask fragments on the ends if the only jump they touch
                # is longer than long_jump_threshold
                for split_idx, split_pt in enumerate(splits):
                    print(f"{split_idx=}: {split_pt=}")
                    before_split_pt = frag_remaining_end[split_idx]
                    after_split_pt = frag_remaining_start[split_idx + 1]
                    jump_dist = torch.norm(new_xyz[after_split_pt, 1, :] - new_xyz[before_split_pt, 1, :])
                    print(f"{jump_dist=}")
                    is_long_jump[split_idx + 1] = jump_dist > long_jump_threshold

                print(f"{is_long_jump=}")

                for split_idx in range(len(splits_with_ends) - 1):
                    start_idx = splits_with_ends[split_idx]
                    end_idx = splits_with_ends[split_idx + 1]
                    if is_long_jump[split_idx] and is_long_jump[split_idx + 1]:
                        new_mask_by_split[split_idx] = False

                clash_mask_by_split = check_clash(new_xyz, splits_with_ends, mean_fit_score_by_split,
                                                  clash_threshold=clash_threshold)

                new_mask_by_split = torch.logical_and(new_mask_by_split, clash_mask_by_split)

                for split_idx in range(len(splits_with_ends) - 1):
                    start_idx = splits_with_ends[split_idx]
                    end_idx = splits_with_ends[split_idx + 1]
                    new_mask[start_idx:end_idx, :] = torch.logical_and(new_mask[start_idx:end_idx, :],
                                                                       new_mask_by_split[split_idx])

                new_pdb_path_before_realign = f"{out_dir}/new_xyz_before_realign_cycle_{i_cycle}.pdb"
                util.writepdb(
                    new_pdb_path_before_realign,
                    new_xyz,
                    seq,
                    Ls,
                    bfacts=100 * pred_lddt[0],
                )
                new_xyz = util.realign_missing(new_xyz, new_mask, sigma=0.5)

                new_pdb_path = f"{out_dir}/new_xyz_after_realign_cycle_{i_cycle}.pdb"
                util.writepdb(
                    new_pdb_path,
                    new_xyz,
                    seq,
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
                ).to(xyz_prev)

                if pdb_name == "globin":
                    splits_with_ends = [0, 100, 130, L]
                    new_mask_by_split = torch.tensor([True, False, True])
                elif pdb_name == "atpbind":
                    splits_with_ends = [0, 198, 306, 439, 545, 671, L]
                    new_mask_by_split = torch.tensor([True, False, True, True, True, True])
                elif pdb_name == "atpbind_modelangelo":
                    splits_with_ends = [0, L]
                    new_mask_by_split = torch.tensor([True])
                else:
                    raise ValueError(f"Unknown pdb name provided: {pdb_name}")

                new_mask = torch.full((len(new_xyz), 27), True, device=xyz_prev.device)
                for split_idx in range(len(splits_with_ends) - 1):
                    start_idx = splits_with_ends[split_idx]
                    end_idx = splits_with_ends[split_idx + 1]
                    new_mask[start_idx:end_idx, :] = torch.logical_and(new_mask[start_idx:end_idx, :],
                                                                       new_mask_by_split[split_idx])

                new_xyz = util.realign_missing(new_xyz, new_mask, sigma=0.5)

                new_pdb_path = f"{out_dir}/new_xyz_after_realign_cycle_{i_cycle}.pdb"
                util.writepdb(
                    new_pdb_path,
                    new_xyz,
                    seq,
                    Ls,
                    bfacts=100 * pred_lddt[0],
                )

            if replace_template:
                conf = torch.where(new_mask.all(dim=-1), temp_conf, 0.0).to(seq.device)
                seq_w_gaps = torch.where(new_mask.all(dim=-1), seq, 20)
                seq_onehot = torch.nn.functional.one_hot(seq, num_classes=21).float()
                t1d = torch.cat((seq_onehot, conf[:, None]), -1).unsqueeze(0)
                xyz_t = new_xyz[None, :, :, :]  # (T, L, A, X)
                mask_t = new_mask[None, :, :]  # (T, L, A)
                mask_t_2d = mask_t[:, :, :3].all(dim=-1)  # (T, L)
                mask_t_2d = mask_t_2d[:, None, :] * mask_t_2d[:, :, None]  # (T, L, L)
                t2d = xyz_to_t2d(xyz_t[None, :, :, :, :], mask_t_2d[None, :, :, :]).half()
                seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
                alpha, _, alpha_mask, _ = pred.xyz_converter.get_torsions(
                    xyz_t.reshape(-1, L, 27, 3).float(), seq_tmp, mask_in=mask_t.reshape(-1, L, 27)
                )
                xyz_t = xyz_t[:, :, 1, :]
                alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[..., 0]))

                alpha[torch.isnan(alpha)] = 0.0
                alpha = alpha.reshape(1, -1, L, 10, 2)
                alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
                alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 3 * 10)
            if replace_xyz_prev:
                xyz_prev = new_xyz
                mask_recycle = new_mask[None, :, :3].bool().all(dim=-1)
                mask_recycle = mask_recycle[:, :, None] * mask_recycle[:, None, :]  # (B, L, L)
            else:
                mask_recycle = None
            if not use_pair_prev:
                pair_prev = torch.zeros_like(pair_prev)
            if not use_state_prev:
                state_prev = torch.zeros_like(state_prev)
            if not use_msa:
                msa_seed = torch.zeros_like(msa_seed)
                msa_extra = torch.zeros_like(msa_extra)
                seq = torch.zeros_like(seq)
            if not use_msa_prev:
                msa_prev = torch.zeros_like(msa_prev)
            if freeze_masked_msa2pair:
                msa2pair_freeze_mask = torch.einsum("i,j->ij", new_mask, new_mask)


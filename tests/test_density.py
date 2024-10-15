import argparse
import os

import pytest
import torch
from icecream import ic

L = 5
MAX_NUM_ATOMS_PER_RESIDUE = 27
NUM_EUCLIDEAN_DIMS = 3
MAX_AMINO_ACID_IDX = 10  # there are more amino acids than this but it works for the test


def get_parser() -> argparse.ArgumentParser:
    """
    Same as network.predict.get_args, but does not call argparse.ArgumentParser.parse_args at the end.
    """

    default_model = os.path.dirname(__file__) + "/weights/RF2_jan24.pt"
    parser = argparse.ArgumentParser(description="RoseTTAFold2NA")
    parser.add_argument("-inputs", help="R|Input data in format A:B:C, with\n"
                                        "   A = multiple sequence alignment file\n"
                                        "   B = hhpred hhr file\n"
                                        "   C = hhpred atab file\n"
                                        "Spaces seperate multiple inputs.  The last two arguments may be omitted\n",
                        required=True, nargs='+')
    parser.add_argument("-mapfile", default=None, type=str, help="Electron density map.")
    parser.add_argument("-db", help="HHpred database location", default=None)
    parser.add_argument("-prefix", default="S", type=str, help="Output file prefix [S]")
    parser.add_argument("-symm", default="C1",
                        help="Symmetry group (Cn,Dn,T,O, or I).  If provided, 'input' should cover the asymmetric unit. [C1]")
    parser.add_argument("-model", default=default_model, help="Model weights. [weights/RF2_jan24.pt]")
    parser.add_argument("-n_recycles", default=3, type=int, help="Number of recycles to use [3].")
    parser.add_argument("-n_models", default=1, type=int, help="Number of models to predict [1].")
    parser.add_argument("-subcrop", default=-1, type=int,
                        help="Subcrop pair-to-pair updates. A value of -1 means no subcropping. [-1]")
    parser.add_argument("-topk", default=1536, type=int,
                        help="Limit number of residue-pair neighbors in structure updates. A value of -1 means no subcropping. [2048]")
    parser.add_argument("-low_vram", default=False,
                        help="Offload some computations to CPU to allow larger systems in low VRAM. [False]",
                        action='store_true')
    parser.add_argument("-nseqs", default=256, type=int,
                        help="The number of MSA sequences to sample in the main 1D track [256].")
    parser.add_argument("-nseqs_full", default=2048, type=int,
                        help="The number of MSA sequences to sample in the wide-MSA 1D track [2048].")
    return parser


def test_density():
    from network.density import rosetta_density_dock, params
    plddt_example_min = params["PLDDT_CUT"]  # most of the residues in our example will remain, but not all

    L_s = [L]  # lengths of chains TODO: what happens to pae and xyz dims if there are multiple chains
    pae = torch.rand((L, L))
    plddt = plddt_example_min + (1 - plddt_example_min) * torch.rand(L)  # prevent all residues from being filtered out
    seq = torch.randint(0, MAX_AMINO_ACID_IDX, (L,))
    xyz = torch.rand((L, MAX_NUM_ATOMS_PER_RESIDUE, NUM_EUCLIDEAN_DIMS))
    model = {
        'xyz': xyz,
        'Ls': L_s,
        'seq': seq,
        'plddt': plddt,
        'pae': pae,
    }
    counts = 1
    rosetta_density_dock('test_density_first.pdb', 'test_density_second.pdb', model, counts, 'emd_36027.map')


def test_pose_from_file():
    from pyrosetta import pose_from_file
    pose_from_file('S_00_pred.pdb')


def test_predict():
    import torch
    from network.predict import Predictor

    torch.backends.cuda.preferred_linalg_library(backend="magma")  # avoid issue with cuSOLVER when computing SVD
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args(['-inputs', 'a3m/test.a3m', '-n_recycles', '2', '-topk', '5'])

    pred = Predictor(args.model, torch.device("cuda:0"))

    pred.predict(
        inputs=args.inputs,
        out_prefix=args.prefix,
        symm=args.symm,
        n_recycles=args.n_recycles,
        n_models=args.n_models,
        subcrop=args.subcrop,
        topk=args.topk,
        low_vram=args.low_vram,
        nseqs=args.nseqs,
        nseqs_full=args.nseqs_full,
        ffdb=None
    )


@pytest.mark.parametrize("args_list", [
    # ['-inputs', 'a3m/8C2C_full.a3m', '-mapfile', 'emd_27094.map'],
    ['-inputs', 'a3m/test.a3m', '-n_recycles', '2', "-topk", '5', '-mapfile', 'emd_27094.map'],
    # ['-inputs', 'a3m/rcsb_pdb_8CZC.a3m', '-n_recycles', '2', "-topk", '5'],
    # ['-inputs', 'a3m/test.a3m', 'a3m/test.a3m'],
    # ['-inputs', 'a3m/test_two_chains.a3m']
])
def test_predict_with_density(args_list):
    import torch
    from network.predict import Predictor

    torch.manual_seed(1738)
    torch.backends.cuda.preferred_linalg_library(backend="magma")  # avoid issue with cuSOLVER when computing SVD
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args(args_list)

    pred = Predictor(args.model, torch.device("cuda:0"))

    pred.predict_w_dens(
        inputs=args.inputs,
        out_prefix=args.prefix,
        symm=args.symm,
        n_recycles=args.n_recycles,
        n_models=args.n_models,
        subcrop=args.subcrop,
        topk=args.topk,
        low_vram=args.low_vram,
        nseqs=args.nseqs,
        nseqs_full=args.nseqs_full,
        mapfile=args.mapfile,
        ffdb=None
    )


def test_c1_domain_duplication_is_noop():
    from network.symmetry import symm_subunit_matrix, find_symm_subs
    from network.chemical import INIT_CRDS

    B = 1
    best_xyz = torch.rand((B, L, MAX_NUM_ATOMS_PER_RESIDUE, NUM_EUCLIDEAN_DIMS))
    best_lddt = torch.rand((B, L))
    seq = torch.randint(0, MAX_AMINO_ACID_IDX, (B, L))

    symmids, symmRs, symmmeta, symmoffset = symm_subunit_matrix("C1")
    O = symmids.shape[0]
    n_templ = 4
    SYMM_OFFSET_SCALE = 1.0

    xyz_t = (
            INIT_CRDS.reshape(1, 1, 27, 3).repeat(n_templ, L, 1, 1)
            + torch.rand(n_templ, L, 1, 3) * 5.0 - 2.5
            + SYMM_OFFSET_SCALE * symmoffset * L ** (1 / 2)  # note: offset based on symmgroup
    )
    # template features
    maxtmpl = 1
    xyz_t = xyz_t[:maxtmpl].float().unsqueeze(0)
    xyz_prev = xyz_t[:, 0]
    xyz_prev, symmsub = find_symm_subs(xyz_prev[:, :L], symmRs, symmmeta)
    Osub = symmsub.shape[0]
    Lasu = L // Osub

    best_xyz = best_xyz.float().cpu()
    symmRs = symmRs.cpu()
    best_xyzfull = torch.zeros((B, O * Lasu, 27, 3))
    best_xyzfull[:, :Lasu] = best_xyz[:, :Lasu]
    seq_full = torch.zeros((B, O * Lasu), dtype=seq.dtype)
    seq_full[:, :Lasu] = seq[:, :Lasu]
    best_lddtfull = torch.zeros((B, O * Lasu))
    best_lddtfull[:, :Lasu] = best_lddt[:, :Lasu]
    for i in range(1, O):
        best_xyzfull[:, (i * Lasu):((i + 1) * Lasu)] = torch.einsum('ij,braj->brai', symmRs[i], best_xyz[:, :Lasu])
        seq_full[:, (i * Lasu):((i + 1) * Lasu)] = seq[:, :Lasu]
        best_lddtfull[:, (i * Lasu):((i + 1) * Lasu)] = best_lddt[:, :Lasu]

    assert torch.equal(best_xyz, best_xyzfull)
    assert torch.equal(best_lddt, best_lddtfull)
    assert torch.equal(seq, seq_full)


def test_parse_second_intermediate_pdb():
    from network.parsers import parse_pdb_w_seq
    test_predict_with_density(['-inputs', 'a3m/rcsb_pdb_8CZC.a3m', '-n_recycles', '2', "-topk", '5'])
    xyz_first_intermediate = torch.from_numpy(parse_pdb_w_seq("density_fit_first_intermediate.pdb")[0])
    xyz_second_intermediate = torch.from_numpy(parse_pdb_w_seq("density_fit_second_intermediate.pdb")[0])
    ic(xyz_first_intermediate.shape)
    ic(xyz_second_intermediate.shape)



def test_multidock():
    from network.density import multidock_model
    import rosetta
    pose: rosetta.core.pose.Pose = multidock_model("input.pdb", "emd_27094.map", 1)
    pose.pdb_info(rosetta.core.pose.PDBInfo(pose))
    pose.dump_pdb("output_fit_to_map.pdb")


def test_center_and_realign_missing():
    from network.util import center_and_realign_missing
    L = 1
    xyz = torch.rand(L, MAX_NUM_ATOMS_PER_RESIDUE, NUM_EUCLIDEAN_DIMS)
    mask_t = torch.bernoulli(torch.full((L,), 0.5)).expand(L, MAX_NUM_ATOMS_PER_RESIDUE)
    result = center_and_realign_missing(xyz, mask_t)
    assert result.shape == (L, MAX_NUM_ATOMS_PER_RESIDUE, NUM_EUCLIDEAN_DIMS)
    ic(torch.all(torch.eq(xyz, result), dim=2))
    assert torch.equal(torch.all(torch.eq(xyz, result), dim=2), mask_t)
    ic(xyz)
    ic(mask_t)
    ic(result)


@pytest.mark.parametrize(("use_template", "use_xyz_prev", "use_state_prev", "use_pair_prev"), [
    # (True, False, False, False),
    (False, True, False, False),
    # (False, False, True, False),
    # (False, False, False, True),
])
def test_predict_globin_w_rotated_template(use_template, use_xyz_prev, use_state_prev, use_pair_prev):
    import torch
    from network.predict import Predictor, merge_a3m_homo, get_striping_parameters, pae_unbin
    from network.symmetry import symm_subunit_matrix, find_symm_subs
    from network.chemical import INIT_CRDS
    from network.parsers import parse_a3m, parse_pdb_w_seq
    from network.data_loader import merge_a3m_hetero
    from network.kinematics import xyz_to_t2d
    from network import util
    import numpy as np
    from torch import nn

    torch.backends.cuda.preferred_linalg_library(backend="magma")  # avoid issue with cuSOLVER when computing SVD

    model = os.path.dirname(__file__) + "/weights/RF2_jan24.pt"
    pred = Predictor(model, torch.device("cuda:0"))
    symm = "C1"
    nseqs_full = 2048
    n_templ = 1
    n_recycles = 1
    nseqs = 256
    subcrop = -1
    topk = -1
    low_vram = False
    B = 1
    msa_concat_mode = "diag"
    pred.xyz_converter = pred.xyz_converter.cpu()
    out_prefix = "test_predict_globin_w_rotated_template"

    ###
    # pass 1, combined MSA
    Ls_blocked, Ls, msas, inss = [], [], [], []

    a3m_i = "a3m/myoglobin.a3m"
    msa_i, ins_i, Ls_i = parse_a3m(a3m_i)
    msa_i = torch.tensor(msa_i).long()
    ins_i = torch.tensor(ins_i).long()
    msas.append(msa_i)
    inss.append(ins_i)
    Ls.extend(Ls_i)
    Ls_blocked.append(msa_i.shape[1])

    msa_orig = {'msa': msas[0], 'ins': inss[0]}
    for i in range(1, len(Ls_blocked)):
        msa_orig = merge_a3m_hetero(msa_orig, {'msa': msas[i], 'ins': inss[i]}, [sum(Ls_blocked[:i]), Ls_blocked[i]])
    msa_orig, ins_orig = msa_orig['msa'], msa_orig['ins']

    symmids, symmRs, symmmeta, symmoffset = symm_subunit_matrix(symm)

    ###
    # pass 2, templates
    L = sum(Ls)
    xyz_globin = torch.from_numpy(parse_pdb_w_seq("pdb/rotated_structures/rotated_alpha000_beta000.pdb")[0]).unsqueeze(0)

    # dummy template
    SYMM_OFFSET_SCALE = 1.0
    xyz_t = (
            INIT_CRDS.reshape(1, 1, 27, 3).repeat(n_templ, L, 1, 1)
            + torch.rand(n_templ, L, 1, 3) * 5.0 - 2.5
            + SYMM_OFFSET_SCALE * symmoffset * L ** (1 / 2)  # note: offset based on symmgroup
    )


    mask_t = torch.full((n_templ, L, 27), False)
    t1d = torch.nn.functional.one_hot(torch.full((n_templ, L), 20).long(), num_classes=21).float()  # all gaps
    t1d = torch.cat((t1d, torch.zeros((n_templ, L, 1)).float()), -1)

    maxtmpl = 1

    same_chain = torch.zeros((1, L, L), dtype=torch.bool, device=xyz_t.device)
    stopres = 0
    for i in range(1, len(Ls)):
        startres, stopres = sum(Ls[:(i - 1)]), sum(Ls[:i])
        same_chain[:, startres:stopres, startres:stopres] = True
    same_chain[:, stopres:, stopres:] = True

    # template features
    xyz_t = xyz_t[:maxtmpl].float().unsqueeze(0)
    mask_t = mask_t[:maxtmpl].unsqueeze(0)
    t1d = t1d[:maxtmpl].float().unsqueeze(0)

    seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
    alpha, _, alpha_mask, _ = pred.xyz_converter.get_torsions(xyz_t.reshape(-1, L, 27, 3), seq_tmp,
                                                              mask_in=mask_t.reshape(-1, L, 27))
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
    if (Osub > 1):
        msa_orig, ins_orig = merge_a3m_homo(msa_orig, ins_orig, Osub, mode=msa_concat_mode)

    # index
    idx_pdb = torch.arange(Osub * L)[None, :]

    same_chain = torch.zeros((1, Osub * L, Osub * L)).long()
    i_start = 0
    for o_i in range(Osub):
        for li in Ls:
            i_stop = i_start + li
            idx_pdb[:, i_stop:] += 100
            same_chain[:, i_start:i_stop, i_start:i_stop] = 1
            i_start = i_stop

    mask_t_2d = mask_t[:, :, :, :3].all(dim=-1)  # (B, T, L)
    mask_t_2d = mask_t_2d[:, :, None] * mask_t_2d[:, :, :, None]  # (B, T, L, L)
    mask_t_2d = mask_t_2d.float() * same_chain.float()[:, None]  # (ignore inter-chain region)

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

        for i_cycle in range(n_recycles + 1):
            from network.featurizing import MSAFeaturize
            seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
                msa, ins, p_mask=0.0, params={'MAXLAT': nseqs, 'MAXSEQ': nseqs_full, 'MAXCYCLE': 1})

            seq = seq.unsqueeze(0)
            msa_seed = msa_seed.unsqueeze(0)
            msa_extra = msa_extra.unsqueeze(0)

            # fd memory savings
            msa_seed = msa_seed.half()  # GPU ONLY
            msa_extra = msa_extra.half()  # GPU ONLY

            xyz_prev_prev = xyz_prev.clone()

            with torch.cuda.amp.autocast(True):
                logit_s, _, _, logits_pae, p_bind, xyz_prev, alpha, symmsub, pred_lddt, msa_prev, pair_prev, state_prev = pred.model(
                    msa_seed, msa_extra,
                    seq, xyz_prev,
                    idx_pdb,
                    t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                    alpha_t=alpha_t, mask_t=mask_t_2d,
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
                    striping=STRIPE)
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

            print(f"recycle {i_cycle} plddt {pred_lddt.mean():.3f} pae {logits_pae.mean():.3f} rmsd: TODO")

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
            pred_lddt, logits_pae, logit_s = None, None, None

            remaining_residues = torch.tensor([0, 75, 150])
            if use_template:
                xyz_t = xyz_globin.to(xyz_t)[:, :, 1].unsqueeze(0)
                mask_t = torch.full_like(mask_t, False)
                mask_t[:, :, remaining_residues, :] = True
            if use_xyz_prev:
                xyz_prev = xyz_globin.to(xyz_prev)
                mask_recycle = torch.full((B, L, L), False, device=xyz_prev.device)
                mask_recycle[:, remaining_residues, remaining_residues] = True
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
    outdata['mean_plddt'] = best_lddt.mean().item()
    Lstarti = 0
    for i, li in enumerate(Ls):
        Lstartj = 0
        for j, lj in enumerate(Ls):
            if (j > i):
                outdata['pae_chain_' + str(i) + '_' + str(j)] = 0.5 * (
                        best_pae[:, Lstarti:(Lstarti + li), Lstartj:(Lstartj + lj)].mean()
                        + best_pae[:, Lstartj:(Lstartj + lj), Lstarti:(Lstarti + li)].mean()
                ).item()
            Lstartj += lj
        Lstarti += li

    outfile = "%s_pred.pdb" % (out_prefix)
    util.writepdb(outfile, best_xyz[0], seq[0], Ls, bfacts=100 * best_lddt[0])

    prob_s = [prob.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
    np.savez_compressed("%s.npz" % (out_prefix),
                        dist=prob_s[0].astype(np.float16),
                        lddt=best_lddt[0].detach().cpu().numpy().astype(np.float16),
                        pae=best_pae[0].detach().cpu().numpy().astype(np.float16))

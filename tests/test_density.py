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
    test_predict_with_density( ['-inputs', 'a3m/rcsb_pdb_8CZC.a3m', '-n_recycles', '2', "-topk", '5'])
    xyz_first_intermediate = torch.from_numpy(parse_pdb_w_seq("density_fit_first_intermediate.pdb")[0])
    xyz_second_intermediate = torch.from_numpy(parse_pdb_w_seq("density_fit_second_intermediate.pdb")[0])
    ic(xyz_first_intermediate.shape)
    ic(xyz_second_intermediate.shape)

    # TODO: first axis is 72 here but 100 for xyz_prev_prev??


def test_multidock():
    from network.density import multidock_model
    import rosetta
    pose: rosetta.core.pose.Pose = multidock_model("input.pdb", "emd_27094.map", 1)
    pose.pdb_info(rosetta.core.pose.PDBInfo(pose))
    pose.dump_pdb("output_fit_to_map.pdb")

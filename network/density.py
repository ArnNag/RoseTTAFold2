import os
import torch
import numpy as np
from icecream import ic

import util
import glob

from pyrosetta import rosetta, pose_from_pdb, get_fa_scorefxn, init, Pose

from network.scoring import HbPolyType
from parsers import parse_pdb_w_seq

# init("-beta -crystal_refine -mute core -unmute core.scoring.electron_density -multithreading:total_threads 4")

params = {
    "PLDDT_CUT": 0.6,  # remove residues below this plddt
    "MIN_RES_CUT": 3,  # do not keep segments shorter than this
}


def setup_docking_mover(counts) -> rosetta.protocols.electron_density.DockFragmentsIntoDensityMover:
    dock_into_dens = rosetta.protocols.electron_density.DockFragmentsIntoDensityMover()
    dock_into_dens.setB( 16 )
    dock_into_dens.setGridStep( 1 )
    dock_into_dens.setTopN( 500 , 50*counts , 1*counts )
    dock_into_dens.setMinDist( 3 )
    dock_into_dens.setNCyc( 1 )
    dock_into_dens.setClusterRadius( 3 )
    dock_into_dens.setFragDens( 0.9 )
    dock_into_dens.setMinBackbone( False )
    dock_into_dens.setDoRefine( True )
    dock_into_dens.setMaxRotPerTrans( 10 )
    dock_into_dens.setPointRadius( 5 )
    dock_into_dens.setConvoluteSingleR( False )
    dock_into_dens.setLaplacianOffset( 0 )
    return dock_into_dens


def plddt_trim(model):
    # trim low plddts
    plddt_mask = model['plddt'] > params['PLDDT_CUT']
    if torch.all(torch.logical_not(plddt_mask)):
        raise ValueError(
            f"All predicted pLDDT values were below the cutoff threshold. Lowest pLDDT: {torch.min(torch.nan_to_num(model['plddt'], nan=1e14))}. pLDDT cutoff: {params['PLDDT_CUT']}")
    # remove singletons
    mask, idx, ct = torch.torch.unique_consecutive(plddt_mask, dim=0, return_counts=True, return_inverse=True)
    mask = mask * ct >= params['MIN_RES_CUT']
    plddt_mask = mask[idx]

    pred = model['xyz'][plddt_mask]
    seq = model['seq'][plddt_mask]
    plddt = model['plddt'][plddt_mask]
    pae = model['pae'][plddt_mask][:, plddt_mask]
    L_s = []
    lstart = 0
    for li in model['Ls']:
        newl = torch.sum(plddt_mask[lstart:(lstart + li)])
        if newl > 0: L_s.append(newl)
        lstart += li
    return {
        'xyz': pred,
        'Ls': L_s,
        'seq': seq,
        'plddt': plddt,
        'pae': pae,
        'plddt_mask': plddt_mask,
    }


def multidock_model(allfiles: list[str], mapfile: str) -> rosetta.core.pose.Pose:
    rosetta.core.scoring.electron_density.getDensityMap(mapfile)
    dock_into_dens: rosetta.protocols.electron_density.DockFragmentsIntoDensityMover = setup_docking_mover(counts=1)

    all_poses: Pose = Pose()
    for filename in allfiles:
        pose_before_fit: Pose = pose_from_pdb(filename)
        # os.remove(filename) 
        dock_into_dens.apply(pose_before_fit)
        pose_after_fit: Pose = pose_from_pdb("EMPTY_JOB_use_jd2_000001.pdb")
        all_poses.append_pose_by_jump(pose_after_fit, 1)

    return all_poses


def compute_pae_enrichment(pae_array: torch.Tensor):
    chain_length = pae_array.shape[0]
    pae_cumsum_rows = torch.cumsum(pae_array, dim=0)
    pae_cumsum = torch.cumsum(pae_cumsum_rows, dim=1)
    from torch.nn import functional
    pae_cumsum = functional.pad(input=pae_cumsum, pad=(1, 0, 1, 0), mode='constant', value=0.)

    # The letters below represent the cumulative sum of the region as well as the regions to the top and left.
    # | A | B | C |
    # | D | E | F |
    # | G | H | I |

    all_idxs = torch.arange(chain_length + 1)
    all_slice_lens = all_idxs[None, :] - all_idxs[:, None]
    A = pae_cumsum.diag().expand(chain_length + 1, chain_length + 1)
    F = pae_cumsum[-1,:].expand(chain_length + 1, chain_length + 1)
    H = pae_cumsum[:,-1].expand(chain_length + 1, chain_length + 1)

    sum_inter_split_pae = A + A.T - pae_cumsum - pae_cumsum.T
    sum_intra_split_pae = F + H - F.T - H.T - 2 * sum_inter_split_pae

    return (sum_intra_split_pae * all_slice_lens) / (
            sum_inter_split_pae * (chain_length - all_slice_lens) + 1e-9)


def split_by_pae(
        pae_array: torch.Tensor,
        min_split_length: int,
) -> list[int]:

    chain_length = pae_array.shape[0]
    assert pae_array.shape[1] == chain_length
    assert chain_length >= min_split_length
    assert min_split_length > 0

    pae_enrichment = compute_pae_enrichment(pae_array)

    def split_by_pae_for_region(
            search_start_idx: int,
            search_end_idx: int,
    ) -> list[int]:

        assert chain_length >= search_end_idx
        assert search_end_idx >= search_start_idx
        assert search_start_idx >= 0

        if search_start_idx + min_split_length >= search_end_idx:
            return []

        best_pae_region_scale_factor = -float("inf")
        best_slice = slice(search_start_idx, search_end_idx)
        best_slice_length = -1
        for start_idx in range(search_start_idx, search_end_idx - min_split_length + 1):
            for end_idx in range(start_idx + min_split_length, search_end_idx + 1):
                test_slice = slice(start_idx, end_idx)
                test_slice_length = end_idx - start_idx
                pae_region_scale_factor: float = pae_enrichment[start_idx,end_idx]
                if pae_region_scale_factor > best_pae_region_scale_factor or (
                        pae_region_scale_factor == best_pae_region_scale_factor
                        and test_slice_length > best_slice_length
                ):
                    best_slice = test_slice
                    best_pae_region_scale_factor = pae_region_scale_factor
                    best_slice_length = test_slice_length

        best_slice_before = split_by_pae_for_region(
            search_start_idx=search_start_idx,
            search_end_idx=best_slice.start,
        )

        best_slice_after = split_by_pae_for_region(
            search_start_idx=best_slice.stop,
            search_end_idx=search_end_idx,
        )

        best_slice_before.append(best_slice.stop)
        best_slice_before.extend(best_slice_after)
        return best_slice_before

    return split_by_pae_for_region(
        search_start_idx=0, search_end_idx=chain_length
    )[:-1]


def test_split_by_pae():

    block_sizes = [7, 2, 7, 10]
    total_size = sum(block_sizes)
    block_matrix = torch.ones((total_size, total_size))
    current_index = 0
    for size in block_sizes:
        block_matrix[
            current_index : current_index + size, current_index : current_index + size
        ] = torch.zeros((size, size))
        current_index += size



    print(split_by_pae(block_matrix, min_split_length=1))


def check_clash(xyz: torch.Tensor, splits_with_ends: list[int], fit_scores_by_split: torch.Tensor, clash_threshold: float) -> torch.Tensor:
    """
    xyz: shape [L, MAX_NUMBER_OF_ATOMS, NUM_EUCLIDEAN_DIMS]
    fit_scores_by_split: shape [len(splits_with_ends) - 1]
    """
    import networkx as nx

    # Create a graph
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from(range(len(splits_with_ends) - 1))

    for split_idx_i in range(1, len(splits_with_ends) - 1):
        split_start_i = splits_with_ends[split_idx_i]
        split_end_i = splits_with_ends[split_idx_i + 1]
        for split_idx_j in range(split_idx_i):
            split_start_j = splits_with_ends[split_idx_j]
            split_end_j = splits_with_ends[split_idx_j + 1]
            all_inter_dist = torch.cdist(xyz[split_start_i:split_end_i, :, :].swapaxes(0, 1),
                                         xyz[split_start_j:split_end_j, :, :].swapaxes(0, 1))
            if torch.any(all_inter_dist < clash_threshold):
                G.add_edge(split_idx_i, split_idx_j)

    splits_to_mask: list[int] = []
    for component in nx.components.connected_components(G):
        if len(component) > 1:
            splits_to_mask.append(min((split for split in component), key=lambda split: fit_scores_by_split[split]))

    mask = torch.full((len(splits_with_ends) - 1,), True)

    for split_idx_i in range(len(splits_to_mask)):
        mask[split_idx_i] = False

    return mask


def rosetta_density_dock(before_dock_file, after_dock_file, model, mapfile):
    split_points: list[int] = split_by_pae(model["pae"].cpu(), min_split_length=100)
    ic(split_points)
    # trimmed_model = plddt_trim(model)
    trimmed_model = model
    # ic(trimmed_model["plddt_mask"])
    util.writepdb(before_dock_file, trimmed_model['xyz'], trimmed_model['seq'], trimmed_model['Ls'],
                  bfacts=100 * trimmed_model['plddt'])
    pose: rosetta.core.pose.Pose = multidock_model(before_dock_file, mapfile)
    pose.pdb_info(rosetta.core.pose.PDBInfo(pose))
    pose.dump_pdb(after_dock_file)
    xyz_with_dummy = torch.full_like(model['xyz'], torch.nan).unsqueeze(0)
    # TODO: better way to deal with batch axis than unsqueeze?
    xyz_with_dummy[0, trimmed_model['plddt_mask']] = torch.from_numpy(parse_pdb_w_seq(after_dock_file)[0]).to(
        xyz_with_dummy)

    return xyz_with_dummy, trimmed_model['plddt_mask']

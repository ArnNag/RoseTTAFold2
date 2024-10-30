import os
import torch
from icecream import ic

from network import util
import glob

from pyrosetta import rosetta, pose_from_pdb, get_fa_scorefxn, init, Pose

from network.parsers import parse_pdb_w_seq

init("-beta -crystal_refine -mute core -unmute core.scoring.electron_density -multithreading:total_threads 4")

params = {
    "PLDDT_CUT": 0.6,  # remove residues below this plddt
    "MIN_RES_CUT": 3,  # do not keep segments shorter than this
}


def setup_docking_mover(counts) -> rosetta.protocols.electron_density.DockFragmentsIntoDensityMover:
    dock_into_dens = rosetta.protocols.electron_density.DockFragmentsIntoDensityMover()
    dock_into_dens.setB(16)
    dock_into_dens.setGridStep(1)
    dock_into_dens.setTopN(500, 10 * counts, 5 * counts)
    dock_into_dens.setMinDist(3)
    dock_into_dens.setNCyc(1)
    dock_into_dens.setClusterRadius(3)
    dock_into_dens.setFragDens(0.9)
    dock_into_dens.setMinBackbone(False)
    dock_into_dens.setDoRefine(True)
    dock_into_dens.setMaxRotPerTrans(10)
    dock_into_dens.setPointRadius(5)
    dock_into_dens.setConvoluteSingleR(False)
    dock_into_dens.setLaplacianOffset(0)
    return dock_into_dens


def plddt_trim(model):
    # trim low plddts
    plddt_mask = model['plddt'] > params['PLDDT_CUT']
    if torch.all(torch.logical_not(plddt_mask)):
        raise ValueError(f"All predicted pLDDT values were below the cutoff threshold. Lowest pLDDT: {torch.min(torch.nan_to_num(model['plddt'], nan=1e14))}. pLDDT cutoff: {params['PLDDT_CUT']}")
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

def domains_from_pae_matrix_igraph(pae_matrix, pae_power=1, pae_cutoff=5, graph_resolution=1):
    '''
    Takes a predicted aligned error (PAE) matrix representing the predicted error in distances between each 
    pair of residues in a model, and uses a graph-based community clustering algorithm to partition the model
    into approximately rigid groups.

    Arguments:

        * pae_matrix: a (n_residues x n_residues) numpy array. Diagonal elements should be set to some non-zero
          value to avoid divide-by-zero warnings
        * pae_power (optional, default=1): each edge in the graph will be weighted proportional to (1/pae**pae_power)
        * pae_cutoff (optional, default=5): graph edges will only be created for residue pairs with pae<pae_cutoff
        * graph_resolution (optional, default=1): regulates how aggressively the clustering algorithm is. Smaller values
          lead to larger clusters. Value should be larger than zero, and values larger than 5 are unlikely to be useful.

    Returns: a series of lists, where each list contains the indices of residues belonging to one cluster.
    '''
    try:
        import igraph
    except ImportError:
        print('ERROR: This method requires python-igraph to be installed. Please install it using "pip install python-igraph" '
            'in a Python >=3.6 environment and try again.')
        import sys
        sys.exit()
    import numpy
    weights = 1/pae_matrix**pae_power

    g = igraph.Graph()
    size = weights.shape[0]
    g.add_vertices(range(size))
    edges = numpy.argwhere(pae_matrix < pae_cutoff)
    ic(edges)
    if len(edges) == 0:
        raise ValueError("All pAE values were below the pae_cutoff threshold.")
    sel_weights = weights[edges.T[0], edges.T[1]]
    g.add_edges(list(zip(edges.T[0], edges.T[1])))
    g.es['weight'] = list(sel_weights)

    vc = g.community_leiden(weights='weight', resolution_parameter=graph_resolution/100, n_iterations=-1)
    membership = numpy.array(vc.membership)
    from collections import defaultdict
    clusters = defaultdict(list)
    for i, c in enumerate(membership):
        clusters[c].append(i)
    clusters = list(sorted(clusters.values(), key=lambda l:(len(l)), reverse=True))
    return clusters



def multidock_model(allfiles: list[str], mapfile) -> rosetta.core.pose.Pose:
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


def rosetta_density_dock(before_dock_file, after_dock_file, model, counts, mapfile):
    clusters = domains_from_pae_matrix_igraph(model["pae"].cpu(), pae_cutoff=100, graph_resolution=0.005)
    ic(clusters)
    # TODO: split model into clusters and pass into multidock model
    ic(model["pae"])
    trimmed_model = plddt_trim(model)
    ic(trimmed_model["plddt_mask"])
    util.writepdb(before_dock_file, trimmed_model['xyz'], trimmed_model['seq'], trimmed_model['Ls'], bfacts=100 * trimmed_model['plddt'])
    pose: rosetta.core.pose.Pose = multidock_model(before_dock_file, mapfile, counts)
    pose.pdb_info(rosetta.core.pose.PDBInfo(pose))
    pose.dump_pdb(after_dock_file)
    xyz_with_dummy = torch.full_like(model['xyz'], torch.nan).unsqueeze(0)
    # TODO: better way to deal with batch axis than unsqueeze?
    xyz_with_dummy[0][trimmed_model['plddt_mask']] = torch.from_numpy(parse_pdb_w_seq(after_dock_file)[0]).to(xyz_with_dummy)

    return xyz_with_dummy, trimmed_model['plddt_mask']

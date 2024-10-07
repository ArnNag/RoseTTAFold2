import os
import torch
from icecream import ic

from network import util
import glob

from pyrosetta import rosetta, pose_from_pdb, get_fa_scorefxn, init

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


def multidock_model(pdbfile, mapfile, counts) -> rosetta.core.pose.Pose:
    pose: rosetta.core.pose.Pose = pose_from_pdb(pdbfile)
    rosetta.core.scoring.electron_density.getDensityMap(mapfile)
    dock_into_dens: rosetta.protocols.electron_density.DockFragmentsIntoDensityMover = setup_docking_mover(counts)
    dock_into_dens.apply(pose)

    # grab top 'count' poses
    allfiles = glob.glob('EMPTY_JOB_use_jd2_*.pdb')
    allfiles.sort()
    for i, filename in enumerate(allfiles):
        if i == 0:
            pose = pose_from_pdb(filename)
        elif i < counts:
            pose.append_pose_by_jump(pose_from_pdb(filename), 1)
        #os.remove(filename) 
    return pose


def rosetta_density_dock(before_dock_file, after_dock_file, model, counts, mapfile):
    trimmed_model = plddt_trim(model)
    util.writepdb(before_dock_file, trimmed_model['xyz'], trimmed_model['seq'], trimmed_model['Ls'], bfacts=100 * trimmed_model['plddt'])
    pose: rosetta.core.pose.Pose = multidock_model(before_dock_file, mapfile, counts)
    pose.pdb_info(rosetta.core.pose.PDBInfo(pose))
    pose.dump_pdb(after_dock_file)
    xyz_with_dummy = torch.full_like(model['xyz'], torch.nan).unsqueeze(0)
    # TODO: better way to deal with batch axis than unsqueeze?
    xyz_with_dummy[0][trimmed_model['plddt_mask']] = torch.from_numpy(parse_pdb_w_seq(after_dock_file)[0]).to(xyz_with_dummy)

    return xyz_with_dummy, trimmed_model['plddt_mask']

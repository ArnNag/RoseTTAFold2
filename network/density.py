import os
import torch
from network import util
import glob

from pyrosetta import *
init("-beta -crystal_refine -mute core -unmute core.scoring.electron_density -multithreading:total_threads 4")

params = {
    "PLDDT_CUT": 0.6, # remove residues below this plddt
    "MIN_RES_CUT": 3, # do not keep segments shorter than this
}

def setup_docking_mover(counts):
    dock_into_dens = rosetta.protocols.electron_density.DockFragmentsIntoDensityMover()
    dock_into_dens.setB( 16 )
    dock_into_dens.setGridStep( 1 )
    dock_into_dens.setTopN( 500 , 10*counts , 5*counts )
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

def rosetta_density_relax(posein):
    scorefxn = get_fa_scorefxn()
    scorefxn.set_weight( rosetta.core.scoring.elec_dens_fast, 50 )
    scorefxn.set_weight( rosetta.core.scoring.cart_bonded, 0.5 )
    scorefxn.set_weight( rosetta.core.scoring.cart_bonded_angle, 1.0 )
    scorefxn.set_weight( rosetta.core.scoring.pro_close, 0.0 )
    setup = rosetta.protocols.electron_density.SetupForDensityScoringMover()
    relax = rosetta.protocols.relax.FastRelax(scorefxn,1)
    relax.cartesian(True)
    relax.max_iter(100)
    setup.apply(posein)
    relax.apply(posein)

def plddt_trim(model):
    # trim low plddts
    plddt_mask = model['plddt']>params['PLDDT_CUT']
    # remove singletons
    mask,idx,ct = torch.torch.unique_consecutive(plddt_mask,dim=0,return_counts=True,return_inverse=True)
    mask = mask*ct>=params['MIN_RES_CUT']
    plddt_mask = mask[idx]

    pred = model['xyz'][plddt_mask]
    seq = model['seq'][plddt_mask]
    plddt = model['plddt'][plddt_mask]
    pae = model['pae'][plddt_mask][:,plddt_mask]
    L_s = []
    lstart=0
    for li in model['Ls']:
        newl = torch.sum(plddt_mask[lstart:(lstart+li)])
        if newl>0: L_s.append(newl)
        lstart += li
    return {
        'xyz': pred,
        'Ls': L_s,
        'seq': seq,
        'plddt': plddt,
        'pae': pae,
    }

def multidock_model(pdbfile,mapfile, counts):
    pose = pose_from_pdb(pdbfile)
    rosetta.core.scoring.electron_density.getDensityMap(mapfile)
    dock_into_dens = setup_docking_mover(counts)
    dock_into_dens.apply(pose)

    # grab top 'count' poses
    allfiles = glob.glob('EMPTY_JOB_use_jd2_*.pdb')
    allfiles.sort()
    for i,filename in enumerate(allfiles):
        if i==0:
            pose = pose_from_pdb(filename)
        elif i<counts:
            pose.append_pose_by_jump( pose_from_pdb(filename), 1 )
        #os.remove(filename) 
    return pose

def rosetta_density_dock(pdbfile, preds, mapfile ):
    for i,(outfile,model,counts) in enumerate(preds):
        model = plddt_trim(model)
        util.writepdb(outfile, model['xyz'], model['seq'], model['Ls'], bfacts=100*model['plddt'])
        pose_i = multidock_model(outfile, mapfile, counts)
        if i==0:
            pose = pose_i
        else:
            pose.append_pose_by_jump( pose_i, 1 )

    #rosetta_density_relax(pose)

    pose.pdb_info(rosetta.core.pose.PDBInfo(pose))
    pose.dump_pdb(pdbfile) # overwrite
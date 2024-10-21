from pyrosetta import pose_from_file, init, Pose, rosetta
from density import setup_docking_mover
init("-beta -crystal_refine -mute core -unmute core.scoring.electron_density -multithreading:total_threads 4")
pose_orig: Pose = pose_from_file('pdb/AF-P03960-F1-model_v4.cif')

mapfile = "map/emd_14914.map"
rosetta.core.scoring.electron_density.getDensityMap(mapfile)
dock_into_dens: rosetta.protocols.electron_density.DockFragmentsIntoDensityMover = setup_docking_mover(counts=1)

dock_into_dens.apply(pose_orig)

pose_orig.pdb_info(rosetta.core.pose.PDBInfo(pose_orig))
pose_orig.dump_pdb(f"af2_after_dock_movable_region.pdb")


def test_density():
    from network.density import rosetta_density_dock
    rosetta_density_dock('model_00_pred.pdb', 'emd_36027.map')

def test_predict():
    from collections import namedtuple
    import torch
    from network.predict import read_index, read_data, Predictor
    import os
    import argparse

    torch.backends.cuda.preferred_linalg_library(backend="magma")
    default_model = os.path.dirname(__file__)+"/weights/RF2_jan24.pt"
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
    parser.add_argument("-symm", default="C1", help="Symmetry group (Cn,Dn,T,O, or I).  If provided, 'input' should cover the asymmetric unit. [C1]")
    parser.add_argument("-model", default=default_model, help="Model weights. [weights/RF2_jan24.pt]")
    parser.add_argument("-n_recycles", default=3, type=int, help="Number of recycles to use [3].")
    parser.add_argument("-n_models", default=1, type=int, help="Number of models to predict [1].")
    parser.add_argument("-subcrop", default=-1, type=int, help="Subcrop pair-to-pair updates. A value of -1 means no subcropping. [-1]")
    parser.add_argument("-topk", default=1536, type=int, help="Limit number of residue-pair neighbors in structure updates. A value of -1 means no subcropping. [2048]")
    parser.add_argument("-low_vram", default=False, help="Offload some computations to CPU to allow larger systems in low VRAM. [False]", action='store_true')
    parser.add_argument("-nseqs", default=256, type=int, help="The number of MSA sequences to sample in the main 1D track [256].")
    parser.add_argument("-nseqs_full", default=2048, type=int, help="The number of MSA sequences to sample in the wide-MSA 1D track [2048].")
    args: argparse.Namespace = parser.parse_args(['-inputs', 'test.a3m'])

    if (args.db is not None):
        FFDB = args.db
        FFindexDB = namedtuple("FFindexDB", "index, data")
        ffdb = FFindexDB(read_index(FFDB+'_pdb.ffindex'),
                         read_data(FFDB+'_pdb.ffdata'))
    else:
        ffdb = None

    if (torch.cuda.is_available()):
        print ("Running on GPU")
        pred = Predictor(args.model, torch.device("cuda:0"))
    else:
        print ("Running on CPU")
        pred = Predictor(args.model, torch.device("cpu"))

    if args.mapfile is not None and args.mapfile != "None":
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
            ffdb=ffdb)
    else:
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
            ffdb=ffdb)

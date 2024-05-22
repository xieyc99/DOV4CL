from .SimCLR.simclr import SimCLRModel
from .BYOL.byol import BYOL
from .SIMSIAM.simsiam import SimSiamModel
from .MOCO.moco import MoCoModel
from .MOCOv3.mocov3 import MoCov3

def set_model(args):
    if args.method == 'simclr':
        return SimCLRModel(args)
    elif args.method == 'byol':
        return BYOL(args)
    elif args.method == 'simsiam':
        return SimSiamModel(args)
    elif args.method == 'moco':
        return MoCoModel(args)
    elif args.method == 'mocov3':
        return MoCov3(args)
    else:
        raise  NotImplementedError

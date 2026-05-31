import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from spandrel import ModelLoader


def load_model(args):
    model_pth = args.model
    if args.mode == "realesrgan":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        checkpoint = torch.load(model_pth, map_location="cpu")
        model.load_state_dict(checkpoint.get("params_ema", checkpoint), strict=True)
        model.body = torch.nn.Sequential(
            *[model.body[i] for i in range(args.keep_layers)]
        )
    elif args.mode == "spandrel":
        model = ModelLoader().load_from_file(model_pth).model

    model.eval()

    return model

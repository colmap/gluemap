import os

# Import VPRModel with salad on sys.path, temporarily swapping 'models' in
# sys.modules to avoid namespace collision with croco's models directory.
import sys

import thirdparty.path_to_thirdparty  # noqa: F401  (adds all thirdparty submodules to sys.path)

from mapanything.models.mapanything.model import MapAnything  # noqa: E402
from mast3r.model import AsymmetricMASt3R  # noqa: E402
from pi3.models.pi3 import Pi3  # noqa: E402
from pi3.models.pi3x import Pi3X  # noqa: E402
from vggsfm.vggsfm_tracker import TrackerPredictor  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402

_salad_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../thirdparty/salad")
)
_saved_models = {
    k: v
    for k, v in sys.modules.items()
    if k == "models" or k.startswith("models.")
}
for k in _saved_models:
    del sys.modules[k]
sys.path.insert(0, _salad_path)
from vpr_model import VPRModel  # noqa: E402

sys.path.remove(_salad_path)
_salad_models = {
    k: v
    for k, v in sys.modules.items()
    if k == "models" or k.startswith("models.")
}
for k in _salad_models:
    del sys.modules[k]
sys.modules.update(_saved_models)

import numpy as np
import torch
from safetensors.torch import load_file


def load_models(args, keys=set()):
    chosen_model = getattr(args, "chosen_model", "pi3")
    keys.add(chosen_model)

    models = {}

    # Load the chosen multi-view model
    if chosen_model == "pi3" and chosen_model in keys:
        models["pi3"] = Pi3()
        models["pi3"].load_state_dict(load_file(args.path_feedforward))
    elif chosen_model == "pi3x" and chosen_model in keys:
        models["pi3x"] = Pi3X(use_multimodal=False)
        models["pi3x"].load_state_dict(
            load_file(args.path_feedforward), strict=False
        )
    elif chosen_model == "vggt" and chosen_model in keys:
        models["vggt"] = VGGT()
        models["vggt"].load_state_dict(
            torch.load(
                args.path_feedforward, map_location="cpu", weights_only=False
            )
        )
    elif chosen_model == "map_anything" and chosen_model in keys:
        models["map_anything"] = MapAnything.from_pretrained(
            args.path_feedforward
        )

    # Load Doppelganger++
    if "dg" in keys:
        models["dg"] = AsymmetricMASt3R(
            pos_embed="RoPE100",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
            head_type="catmlp+dpt",
            head_type_dg="transformer",
            output_mode="pts3d+desc24",
            output_mode_dg="dg_score",
            depth_mode=("exp", -np.inf, np.inf),
            conf_mode=("exp", 1, np.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            desc_conf_mode=("exp", 0, np.inf),
            add_dg_pred_head=True,
            freeze=["mask", "encoder", "decoder", "head"],
        ).from_pretrained(args.path_dg)

    # Load VGG-SfM Tracker
    if "vggsfm" in keys:
        models["vggsfm"] = TrackerPredictor()
        models["vggsfm"].load_state_dict(
            torch.load(
                args.path_tracker, map_location="cpu", weights_only=False
            )
        )

    # Load SALAD
    if "salad" in keys:
        models["salad"] = VPRModel(
            backbone_arch="dinov2_vitb14",
            backbone_config={
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
            agg_arch="SALAD",
            agg_config={
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        )

        models["salad"].load_state_dict(
            torch.load(
                args.path_retrieval, map_location="cpu", weights_only=False
            ),
            strict=False,
        )

    if getattr(args, "cpu", False):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name, model in models.items():
        model.eval()
        model.to(device)

    if args.distributed:
        for model_name, model in models.items():
            models[model_name] = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.gpu],
                find_unused_parameters=True,
                static_graph=True,
            )

    # Create uniform alias for the chosen multi-view model
    if chosen_model in models:
        models["mv"] = models[chosen_model]

    return models, device


def load_all_models(args):
    """Load all pipeline models at once for reuse across datasets.

    Respects args.disable_tracking to skip VGGSfM if not needed.
    When direct_inference is enabled, only loads the backbone model.
    """
    chosen_model = getattr(args, "chosen_model", "pi3")
    if getattr(args, "direct_inference", False):
        keys = {chosen_model}
    else:
        keys = {"salad", "dg", chosen_model}
        if not getattr(args, "disable_tracking", False):
            keys.add("vggsfm")
    return load_models(args, keys=keys)

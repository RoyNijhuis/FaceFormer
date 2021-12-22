import sys
sys.path.extend(['.', '..'])

from utils.inference_utils import run_inversion
from utils import restyle_inference_utils
from utils.model_utils import load_model, load_generator
import torch
from notebooks.notebook_utils import FINETUNED_MODELS, RESTYLE_E4E_MODELS
import os

hyperstyle_path = "./pretrained_models/hyperstyle_ffhq.pt" #@param {type:"string"}
w_encoder_path = "./pretrained_models/faces_w_encoder.pt" #@param {type:"string"}
restyle_e4e_path = os.path.join("./pretrained_models", RESTYLE_E4E_MODELS['name'])

def run_domain_adaptation(inputs, generator_type='toonify'):
    """ Combine restyle e4e's latent code with HyperStyle's predicted weight offsets. """
    generator_path = f"./pretrained_models/{FINETUNED_MODELS[generator_type]['name']}"

    restyle_e4e, restyle_e4e_opts = load_model(restyle_e4e_path, device='cuda', is_restyle_encoder=True)
    restyle_e4e_opts.n_iters_per_batch = 5
    restyle_e4e_opts.resize_outputs = False

    y_hat, latents = restyle_inference_utils.run_on_batch(inputs, restyle_e4e, restyle_e4e_opts)
    restyle_e4e = None
    restyle_e4e_opts = None

    torch.cuda.empty_cache()

    net, opts = load_model(hyperstyle_path, update_opts={"w_encoder_checkpoint_path": w_encoder_path})
    opts.n_iters_per_batch = 5
    opts.resize_outputs = False  # generate outputs at full resolution

    y_hat2, _, weights_deltas, _ = run_inversion(inputs, net, opts)
    net = None
    opts = None
    torch.cuda.empty_cache()
    #weights_deltas = filter_non_ffhq_layers_in_toonify_model(weights_deltas)
    fine_tuned_generator = load_generator(generator_path)
    return fine_tuned_generator([latents],
                                input_is_latent=True,
                                randomize_noise=True,
                                return_latents=True,
                                weights_deltas=weights_deltas)


def filter_non_ffhq_layers_in_toonify_model(weights_deltas):
    toonify_ffhq_layer_idx = [14, 15, 17, 18, 20, 21, 23, 24]  # convs 8-15 according to model_utils.py
    for i in range(len(weights_deltas)):
        if weights_deltas[i] is not None and i not in toonify_ffhq_layer_idx:
            weights_deltas[i] = None
    return weights_deltas

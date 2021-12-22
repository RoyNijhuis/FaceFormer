import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from editing.face_editor import FaceEditor
from options.test_options import TestOptions
from utils.common import tensor2im
from utils.inference_utils import run_inversion
from utils.model_utils import load_model

test_opts = {
    "data_path": "./input_images",
    "exp_dir": "./edit_results",
    "dataset_type": "ffhq_hypernet",
    "test_batch_size": 1,
    "test_workers": 1,
    "n_images": None,
    "directions": ["age", "pose", "smile"]
}

hyperstyle_path = "./pretrained_models/hyperstyle_ffhq.pt"


def run():
    out_path_results = os.path.join(test_opts["exp_dir"], 'editing_results')
    out_path_coupled = os.path.join(test_opts["exp_dir"], 'editing_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    net, opts = load_model(hyperstyle_path)
    opts.resize_outputs = False
    opts.data_path = "./input_images"
    opts.n_images = 1
    opts.directions = ["age"]
    opts.test_batch_size = 1
    opts.test_workers = 1
    opts.factor_range = 1

    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'])
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    latent_editor = FaceEditor(net.decoder)

    global_i = 0
    for input_batch in tqdm(dataloader):

        if global_i >= opts.n_images:
            break

        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            result_batch = run_on_batch(input_cuda, net, latent_editor, opts)

        resize_amount = (256, 256)
        for i in range(input_batch.shape[0]):

            im_path = dataset.paths[global_i]
            results = result_batch[i]

            inversion = results.pop('inversion')
            input_im = tensor2im(input_batch[i])

            all_edit_results = []
            for edit_name, edit_res in results.items():
                # set the input image
                res = np.array(input_im.resize(resize_amount))
                # set the inversion
                res = np.concatenate([res, np.array(inversion.resize(resize_amount))], axis=1)
                # add editing results side-by-side
                for result in edit_res:
                    res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
                res_im = Image.fromarray(res)
                all_edit_results.append(res_im)

                edit_save_dir = os.path.join(out_path_results, edit_name)
                os.makedirs(edit_save_dir, exist_ok=True)
                res_im.save(os.path.join(edit_save_dir, os.path.basename(im_path)))

            # save final concatenated result of all edits
            coupled_res = np.concatenate(all_edit_results, axis=0)
            im_save_path = os.path.join(out_path_coupled, os.path.basename(im_path))
            Image.fromarray(coupled_res).save(im_save_path)
            global_i += 1


def run_on_batch(inputs, net, latent_editor, opts):
    y_hat, _, weights_deltas, codes = run_inversion(inputs, net, opts)

    net = None
    torch.cuda.empty_cache()

    edit_directions = opts.directions
    # store all results for each sample, split by the edit direction
    print("Inversed!")
    #results = {idx: {'inversion': tensor2im(y_hat[idx])} for idx in range(len(inputs))}
    for edit_direction in edit_directions:
        edit_res = latent_editor.apply_interfacegan(latents=codes,
                                                    weights_deltas=weights_deltas,
                                                    direction=edit_direction,
                                                    factor=1)
        # store the results for each sample
        for idx, sample_res in edit_res.items():
            #results[idx][edit_direction] = sample_res
            pass
    #return results


if __name__ == '__main__':
    run()

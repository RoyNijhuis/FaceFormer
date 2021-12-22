import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

sys.path.append(".")
sys.path.append("..")

from notebooks.notebook_utils import Downloader, HYPERSTYLE_PATHS, W_ENCODERS_PATHS, FINETUNED_MODELS, RESTYLE_E4E_MODELS, run_alignment
from utils.domain_adaptation_utils import run_domain_adaptation
from utils.model_utils import load_model
from utils.common import tensor2im

def generate_styled_image(image_path, style_name):
    input_image = run_alignment(image_path)
    input_image.resize((256, 256))

    img_transforms = transforms.Compose([transforms.Resize((256, 256)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transformed_image = img_transforms(input_image)

    with torch.no_grad():
        tic = time.time()
        result, _ = run_domain_adaptation(transformed_image.unsqueeze(0).cuda(), generator_type=style_name)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))


    resize_amount = (256, 256)

    final_res = tensor2im(result[0]).resize(resize_amount)
    input_im = tensor2im(transformed_image).resize(resize_amount)
    res = np.concatenate([np.array(input_im), np.array(final_res)], axis=1)
    res = Image.fromarray(res)
    return res

if __name__ == "__main__":
    image_path = "roy.jpg" #@param {type:"string"}
    result_img = generate_styled_image(image_path, 'toonify')
    result_img.show()

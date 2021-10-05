from PIL import Image
import numpy as np
import fasttext
import pdb

import random

import torch
import torchvision.transforms as transforms

from tagan import Generator, split_sentence_into_words

def tensor_to_img(tensor_image, denorm=False):
    if denorm:
        tensor_image = invTrans(tensor_image)
    img = transforms.ToPILImage()(tensor_image)
#     img = tensor_image.permute(1,2,0)
    return img


def tensor_to_np(tensor_image):
    return tensor_image.mul(0.5).add(0.5).mul(255).add_(0.5).permute(1, 2, 0).to('cpu', torch.uint8).numpy()


class TripletTAGAN():
    def __init__(self, trained_model, word_embedding, device):
        self.G = G = torch.load(trained_model).to(device)
        self.G.eval()
        self.word_embedding = word_embedding
        self.device = device
        
    def txt_to_emb(self, text):
        words = split_sentence_into_words(text)
        txt = torch.tensor([self.word_embedding.get_word_vector(w) for w in words], device=self.device)
        txt = txt.unsqueeze(1)
        len_txt = torch.tensor([len(words)], dtype=torch.long, device=self.device)
        return txt, len_txt
    
    def gen_np(self, input_img, txt, len_txt):
        # Generates image and converts to numpy
        with torch.no_grad():
            if input_img.shape[0] == 1:
                output, _ = self.G(input_img.cuda(), (txt, len_txt))
                img_np = tensor_to_np(output[0])
                return img_np
            else:
                bs = input_img.shape[0]
                cat_txt = torch.cat([txt]*bs, axis=1)
                cat_len_txt = torch.cat([len_txt]*bs)
                output, _ = self.G(input_img.cuda(), (cat_txt, cat_len_txt))
                imgs_np = [tensor_to_np(im) for im in output]
                return imgs_np
        
    def gen_from_text(self, input_img, text):
        # Accepts tensor image/images and string text.        
        if len(input_img.shape) < 4:
            input_img = input_img.unsqueeze(0)
                        
        txt, len_txt = self.txt_to_emb(text)
                
        return self.gen_np(input_img, txt, len_txt)
            
    
    def interpolate_between_txts(self, input_img, text1, text2, bins=10):
        
        # Accepts tensor image/images and string text.        
        if len(input_img.shape) < 4:
            input_img = input_img.unsqueeze(0)
            
            
        txt1, len_txt1 = self.txt_to_emb(text1)
        txt2, len_txt2 = self.txt_to_emb(text2)
        
        alphas = [(1/bins)*k for k in range(bins+1)]
        
        interpolated_images = []
        
        for alpha in alphas:
            text_vector = (1-alpha)*txt1 + (alpha)*txt2
            length = (1-alpha)*len_txt1 + (alpha)*len_txt2
            int_img = self.gen_np(input_img, text_vector, length)
            interpolated_images.append(int_img)
            
        return interpolated_images
        
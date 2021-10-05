import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import random
import nltk
from PIL import Image
import numpy as np
import json
from utils import img_load_and_transform


class ImageDataset(data.Dataset):
    pass


class Dataset(data.Dataset):

    def __init__(self, root, data_file_name, vocab, transform=None, return_target=True, transform_target=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            data: index file name.
            transform: image transformer.
            vocab: pre-processed vocabulary.
        """
        self.root = root
        with open(data_file_name, 'r') as f:
            self.data = json.load(f)
        self.ids = range(len(self.data))
        self.vocab = vocab
        self.transform = transform
        self.return_target = return_target
        if transform_target is None:
            self.transform_target = transform
        else:
            self.transform_target = transform_target

    def __getitem__(self, index):
        """Returns one data pair (image and concatenated captions)."""
        data = self.data
        vocab = self.vocab
        id = self.ids[index]

        candidate_asin = data[id]['candidate']
        candidate_img_name = candidate_asin + '.jpg'
        candidate_image = Image.open(os.path.join(
            self.root, candidate_img_name)).convert('RGB')
        if self.transform is not None:
            candidate_image = self.transform(candidate_image)

        if self.return_target:
            target_asin = data[id]['target']
            target_img_name = target_asin + '.jpg'
            target_image = Image.open(os.path.join(
                self.root, target_img_name)).convert('RGB')
            if self.transform_target is not None:
                target_image = self.transform_target(target_image)
        else:
            target_image = candidate_image
            target_asin = ''

        caption_texts = data[id]['captions']
        # Convert caption (string) to word ids.
        tokens = []
        for capt_ in caption_texts:
            tokens += nltk.tokenize.word_tokenize(str(capt_).lower())
#         tokens = nltk.tokenize.word_tokenize(str(caption_texts[0]).lower()) + ['<and>'] + \
#             nltk.tokenize.word_tokenize(str(caption_texts[1]).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        caption = torch.Tensor(caption)

        return target_image, candidate_image, caption, {'target': target_asin, 'candidate': candidate_asin, 'caption': caption_texts}

    def __len__(self):
        return len(self.ids)

    
class EmbeddingDataset(data.Dataset):

    def __init__(self, img_root, emb_root, data_file_name, vocab, transform=None, tokenize=True, join_captions=False):
        self.root = img_root
        self.emb_root = emb_root
        with open(data_file_name, 'r') as f:
            data_all = json.load(f)
        data = []
        for items in data_all:
            if os.path.exists(os.path.join(img_root, items['candidate']+'.jpg')) and os.path.exists(os.path.join(img_root, items['target']+'.jpg')):
                data.append(items)
        self.data = data
        self.ids = range(len(self.data))
        self.vocab = vocab
        self.transform = transform
        self.max_length = 45
        self.tokenize = tokenize
        self.join_captions = join_captions


    def __getitem__(self, index):
        """Returns one data pair (image and concatenated captions)."""
        data = self.data
        vocab = self.vocab
        id = self.ids[index]
        
        # Target image for display purposes
        target_asin = data[id]['target']
        target_img_name = target_asin + '.jpg'
        target_image = Image.open(os.path.join(
            self.root, target_img_name)).convert('RGB')
        if self.transform is not None:
            target_image = self.transform(target_image)
        
        # Read and return image
        candidate_asin = data[id]['candidate']
        candidate_img_name = candidate_asin + '.jpg'
        candidate_image = Image.open(os.path.join(
            self.root, candidate_img_name)).convert('RGB')
        if self.transform is not None:
            candidate_image = self.transform(candidate_image)

        # Read caption
        caption_texts = data[id]['captions']
        # if there is more than one caption, choose one random caption
        if self.join_captions:
            one_caption = caption_texts[0] + ' and ' + caption_texts[1] 
        else:
            one_caption = random.choice(caption_texts)
        
#         # Convert caption (string) to word ids.
        if self.tokenize:
            tokens = []
    #         for capt_ in caption_texts:
    #             tokens += nltk.tokenize.word_tokenize(str(capt_).lower())
            tokens += nltk.tokenize.word_tokenize(str(one_caption).lower())
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))

            caption = torch.Tensor(caption)
            length = len(caption)
            caption_out = torch.zeros(self.max_length).long()
            caption_out[:length] = caption
            caption_out_ = caption_out
        else:
            caption_out_ = one_caption
        
        # Read latent code of target image
        target_img = data[id]['target']
        latent_code_path = os.path.join(self.emb_root, target_img + '.npz')
        latent_code = np.load(latent_code_path)['dlatents']
        latent_code = torch.Tensor(latent_code)

        return target_image, candidate_image, caption_out_, length, latent_code, {'target': data[id]['target'], 'candidate': data[id]['candidate'], 'caption': one_caption}

    def __len__(self):
        return len(self.ids)
 

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of images.
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    target_images, candidate_images, captions, meta = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    target_images = torch.stack(target_images, 0)
    candidate_images = torch.stack(candidate_images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    captions_out = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        captions_out[i, :end] = cap[:end]
#     print(meta.shape)
    return target_images, candidate_images, captions_out, lengths, meta


def get_loader(root, data_file_name, vocab, transform, batch_size, shuffle, return_target, num_workers, transform_target=None):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    # relative caption dataset
    dataset = Dataset(root=root,
                      data_file_name=data_file_name,
                      vocab=vocab,
                      transform=transform,
                      return_target=return_target,
                      transform_target=transform_target)

    # Data loader for the dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader

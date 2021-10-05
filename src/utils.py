import os
import torch
import shutil
import json
import math
import numpy as np
from PIL import Image
from torchvision import transforms
from joblib import Parallel, delayed


import pdb

import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def recall(actual, predicted, k):
    act_set = set([actual])
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result


def compute_score(solution, prediction):
    n = len(solution)
    scores_r_10 = []
    scores_r_50 = []
    scores_r_1000 = []
    scores_r_2000 = []
    scores_r_5000 = []
    for i in range(n):

        assert solution[i]["candidate"] == prediction[i]["candidate"]

        scores_r_10.append(recall(solution[i]["target"], prediction[i]["ranking"], 10))
        scores_r_50.append(recall(solution[i]["target"], prediction[i]["ranking"], 50))
        scores_r_1000.append(recall(solution[i]["target"], prediction[i]["ranking"], 1000))
        scores_r_2000.append(recall(solution[i]["target"], prediction[i]["ranking"], 2000))
        scores_r_5000.append(recall(solution[i]["target"], prediction[i]["ranking"], 5000))

    return sum(scores_r_10) / n, sum(scores_r_50) / n, sum(scores_r_1000) / n,  sum(scores_r_2000) / n, sum(scores_r_5000) / n


def mean_ranking(output):
    ranks = []
    for r in output:
        if r['target'] in r['ranking']:
            curr_r = r['ranking'].index(r['target'])
            ranks.append(curr_r)
        else:
            ranks.append(len(r['ranking']))
    return np.mean(ranks)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        if not os.path.exists(os.path.join(path, 'images')):
            os.mkdir(os.path.join(path, 'images'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    return


def tensor_to_img(tensor_image, denorm=True):
    if denorm:
        inv_tensor = invTrans(tensor_image)
    img = transforms.ToPILImage()(tensor_image)
    img = tensor_image.permute(1, 2, 0)
    return img


def img_load_and_transform(img_path, img_transform=None):
    img = Image.open(img_path)
    if img_transform == None:
        img_transform = transforms.ToTensor()
    img = img_transform(img)
    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)
    return img


def caption_to_description(c, vocab):
    description = [vocab.idx2word[str(word)] for word in c.cpu().numpy()]
    return ' '.join(description)


def to_np(x):
    "Convert a tensor to a numpy array."
    return x.data.cpu().numpy()


class Ranker():
    def __init__(self, root, image_split_file=None, transform=None, num_workers=0):
        self.num_workers = num_workers
        self.root = root
        data = []
        if image_split_file is not None:
            with open(image_split_file, 'r') as f:
                data_all = json.load(f)
            for items in data_all:
                if os.path.exists(os.path.join(root, items+'.jpg')):
                    data.append(items)    
        self.data = data
        self.ids = range(len(self.data))
        self.transform = transform
        return

    def get_item(self, index):
        data = self.data
        id = self.ids[index]
        img_name = data[id] + '.jpg'
        image = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, data[id]

    def get_items(self, indexes):
        items = Parallel(n_jobs=1)(
            delayed(self.get_item)(
                i) for i in indexes)
        images, meta_info = zip(*items)
        images = torch.stack(images, dim=0)
        return images, meta_info

    def embed(self, v, image_encoder=None, crop_size=224):
        if image_encoder is None:
            vector_size = v.shape[1] * v.shape[2] * v.shape[3]
            feat = v.view(-1, vector_size)
        else:
            feat = image_encoder(v)
        return feat

    def update_emb(self, image_encoder, batch_size=64, crop_size=224):
        data_emb = []
        data_asin = []
        num_data = len(self.data)
        num_batch = math.floor(num_data / batch_size)
        print('updating emb')
        for i in range(num_batch):
            batch_ids = torch.LongTensor(
                [i for i in range(i * batch_size, (i + 1) * batch_size)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)

            with torch.no_grad():
                feat = self.embed(images)

            data_emb.append(feat)
            data_asin.extend(asins)

        if num_batch * batch_size < num_data:
            batch_ids = torch.LongTensor(
                [i for i in range(num_batch * batch_size, num_data)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)

            feat = self.embed(images)

            data_emb.append(feat)
            data_asin.extend(asins)

        self.data_emb = torch.cat(data_emb, dim=0)
        self.data_asin = data_asin
        print('emb updated')
        return

    def compute_rank(self, inputs, target_ids):
        rankings = []
        for i in range(inputs.size(0)):
            distances = (self.data_emb - inputs[i]).pow(2).sum(dim=1)
            ranking = (
                distances < distances[self.data_asin.index(target_ids[i])]).sum(dim=0)
            rankings.append(ranking)
        return torch.FloatTensor(rankings).to(device)

    def get_nearest_neighbors(self, inputs, topK=50):
        neighbors = []
        for i in range(inputs.size(0)):
            [_, neighbor] = (self.data_emb - inputs[i]).pow(2).sum(
                dim=1).topk(dim=0, k=topK, largest=False, sorted=True)
            neighbors.append(neighbor)
        return torch.stack(neighbors, dim=0).to(device)

    def get_nearest_distances(self, inputs, topK=50):
        dists = []
        for i in range(inputs.size(0)):
            (d, _) = (self.data_emb - inputs[i]).pow(2).sum(
                dim=1).topk(dim=0, k=topK, largest=False, sorted=True)
            dists.append(d)
        return torch.stack(dists, dim=0).to(device)


class EncoderRanker(Ranker):
    # previously called GANRanker
    def __init__(self, root, image_split_file, transform=None, num_workers=0):
        super(EncoderRanker, self).__init__(root, image_split_file,
                                            transform=transform, num_workers=num_workers)

    def embed(self, v, image_encoder, crop_size=224):
        feat = image_encoder.forward(F.interpolate(v, size=crop_size))
        return feat

    def update_emb(self, image_encoder, batch_size=64, crop_size=224):
        data_emb = []
        data_asin = []
        num_data = len(self.data)
        num_batch = math.floor(num_data / batch_size)
        print('updating emb')
        for i in range(num_batch):
            batch_ids = torch.LongTensor(
                [i for i in range(i * batch_size, (i + 1) * batch_size)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)

            with torch.no_grad():
                feat = image_encoder(F.interpolate(images, size=crop_size))

            data_emb.append(feat)
            data_asin.extend(asins)

        if num_batch * batch_size < num_data:
            batch_ids = torch.LongTensor(
                [i for i in range(num_batch * batch_size, num_data)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)

            feat = image_encoder(F.interpolate(images, size=crop_size))

            data_emb.append(feat)
            data_asin.extend(asins)

        self.data_emb = torch.cat(data_emb, dim=0)
        self.data_asin = data_asin
        print('emb updated')
        return


class ResnetRanker(Ranker):
    def __init__(self, root, image_split_file=None, transform=None, num_workers=0):
        super(ResnetRanker, self).__init__(root, image_split_file=image_split_file,
                                           transform=transform, num_workers=num_workers)
        resnet152 = models.resnet152(pretrained=True).cuda()
        resnet152.eval()
        for param in resnet152.parameters():
            param.requires_grad = False
        modules = list(resnet152.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def embed(self, v, image_encoder=None, crop_size=224):
        if len(v.shape) < 4 : 
            v = v.unsqueeze(dim=0)
        with torch.no_grad():
            feat = self.resnet.forward(F.interpolate(
                v, size=(crop_size, crop_size))).view(-1, 2048)
        return feat

    def update_emb(self, image_encoder=None, batch_size=64, crop_size=224):
        data_emb = []
        data_asin = []
        num_data = len(self.data)
        num_batch = math.floor(num_data / batch_size)
        print('updating emb')
        for i in range(num_batch):
            batch_ids = torch.LongTensor(
                [i for i in range(i * batch_size, (i + 1) * batch_size)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)

            with torch.no_grad():
                feat = self.embed(images)

            data_emb.append(feat)
            data_asin.extend(asins)

        if num_batch * batch_size < num_data:
            batch_ids = torch.LongTensor(
                [i for i in range(num_batch * batch_size, num_data)])
            images, asins = self.get_items(batch_ids)
            images = images.to(device)

            with torch.no_grad():
                feat = self.embed(images)

            data_emb.append(feat)
            data_asin.extend(asins)

        self.data_emb = torch.cat(data_emb, dim=0)
        self.data_asin = data_asin
        print('emb updated')
        return
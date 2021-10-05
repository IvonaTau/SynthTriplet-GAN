import argparse
from comet_ml import Experiment
import time
import os
import numpy as np
import json
import fasttext
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch import autograd
from torch.autograd import Variable
import torchvision

from data_loader import get_loader, ImageDataset
from build_vocab import Vocabulary
from models import DummyImageEncoder, DummyCaptionEncoder, CNN_ENCODER, RNN_ENCODER
from utils import create_exp_dir, Ranker, EncoderRanker, ResnetRanker, tensor_to_img, to_np, recall, compute_score, caption_to_description, mean_ranking

from tagan import Generator, Discriminator, ReadFromVec, split_sentence_into_words

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Paths to data
# IMAGE_ROOT = '/home/datasets/style_search-yosh_ai/images/asos'

# image_encoder_path = '../models/start-kit/toptee-20191115-110235/image-1024.th'
# text_encoder_path = '../models/start-kit/toptee-20191115-110235/cap-1024.th'

# TOP_K = 500


def label_like(label, x):
    assert label == 0 or label == 1
    v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
    v = v.to(x.device)
    return v


def zeros_like(x):
    return label_like(0, x)


def ones_like(x):
    return label_like(1, x)


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)


def train(args):

    tagan_transform = transforms.Compose([
        transforms.Resize(int(1.06*args.gen_img_size)),

        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(args.gen_img_size),
        transforms.ToTensor()
    ])

    print('Loading a dataset...')
    train_data = ImageDataset(args.img_root ,args.shop)

    save_folder = '{}/{}-{}'.format(args.save,
                                    args.data_set, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(save_folder, scripts_to_save=['train-tagan.py', 'tagan.py'])

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(save_folder, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging(str(args))
    hyper_params = vars(args)
    hyper_params['save_dir'] = save_folder

    experiment = Experiment(api_key="njYbdzqt7zdPd6vz9qCklcuMH",
                            project_name="multimodal-gan-paper", workspace="ivonatau")
    experiment.log_parameters(hyper_params)

    D = Discriminator()
    G = Generator()
    
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    
    
    G, D = G.to(device), D.to(device)

    g_optimizer = torch.optim.Adam(G.parameters(),
                                   lr=args.lr_G, betas=(args.beta1, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(),
                                   lr=args.lr_D, betas=(args.beta1, 0.999))
    g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, 100, args.lr_decay)
    d_lr_scheduler = lr_scheduler.StepLR(d_optimizer, 100, args.lr_decay)
    
    # ----------------------
    #  Training
    # ----------------------

    total_step = len(train_loader)
    LOG_STEP = 100
    SAMPLE_INTERVAL = 500

    batches_done = 0
    best_score = float('-inf')

    for epoch in range(args.epochs):
        print('Epoch:', epoch)

        d_lr_scheduler.step()
        g_lr_scheduler.step()

        avg_D_real_loss = 0
        avg_D_real_c_loss = 0
        avg_D_fake_loss = 0
        avg_G_fake_loss = 0
        avg_G_fake_c_loss = 0
        avg_G_recon_loss = 0
        avg_kld = 0

        for i, (img) in enumerate(train_loader):
            
            G.train()

            img = img.to(device)
            img = img.mul(2).sub(1)
            # BTC to TBC
            txt =
            len_txt = 
            # negative text
            txt_m = torch.cat((txt[:, -1, :].unsqueeze(1), txt[:, :-1, :]), 1)
            len_txt_m = torch.cat((len_txt[-1].unsqueeze(0), len_txt[:-1]))

            # UPDATE DISCRIMINATOR
            D.zero_grad()

            # real images
            real_logit, real_c_prob, real_c_prob_n = D(
                img, txt, len_txt, negative=True)

            real_loss = F.binary_cross_entropy_with_logits(
                real_logit, ones_like(real_logit))
            avg_D_real_loss += real_loss.item()

            real_c_loss = (F.binary_cross_entropy(real_c_prob, ones_like(real_c_prob)) +
                           F.binary_cross_entropy(real_c_prob_n, zeros_like(real_c_prob_n))) / 2
            avg_D_real_c_loss += real_c_loss.item()

            real_loss = real_loss + args.lambda_cond_loss * real_c_loss

            real_loss.backward()

            # synthesized images
            fake, _ = G(img, (txt_m, len_txt_m))
            fake_logit, _ = D(fake.detach(), txt_m, len_txt_m)

            fake_loss = F.binary_cross_entropy_with_logits(
                fake_logit, zeros_like(fake_logit))
            avg_D_fake_loss += fake_loss.item()

            fake_loss.backward()

            d_optimizer.step()

            # UPDATE GENERATOR
            G.zero_grad()

            fake, (z_mean, z_log_stddev) = G(img, (txt_m, len_txt_m))

            kld = torch.mean(-z_log_stddev + 0.5 *
                             (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1))
            avg_kld += 0.5 * kld.item()

            fake_logit, fake_c_prob = D(fake, txt_m, len_txt_m)
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_logit, ones_like(fake_logit))
            avg_G_fake_loss += fake_loss.item()
            fake_c_loss = F.binary_cross_entropy(
                fake_c_prob, ones_like(fake_c_prob))
            avg_G_fake_c_loss += fake_c_loss.item()
            
            
            # Triplet loss
            neutral_img_feats = ranker.embed(fake, crop_size=args.crop_size)
            positive_img_feats = ranker.embed(img, crop_size=args.crop_size)
            negative_img_feats = ranker.embed(source_img, crop_size=args.crop_size)
            loss_G_triplet = triplet_avg(anchor=neutral_img_feats, positive=positive_img_feats, negative=negative_img_feats)
            
            

            G_loss = fake_loss + args.lambda_cond_loss * fake_c_loss + 0.5 * kld + args.a_t*loss_G_triplet

            G_loss.backward()

            # reconstruction for matching input
            recon, (z_mean, z_log_stddev) = G(img, (txt, len_txt))

            kld = torch.mean(-z_log_stddev + 0.5 *
                             (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1))
            avg_kld += 0.5 * kld.item()

            recon_loss = F.l1_loss(recon, img)
            avg_G_recon_loss += recon_loss.item()

            G_loss = args.lambda_recon_loss * recon_loss + 0.5 * kld

            G_loss.backward()

            g_optimizer.step()

            experiment.log_metric('loss_D_fake', avg_D_fake_loss)
            experiment.log_metric('loss_D_real', avg_D_real_c_loss)
            experiment.log_metric('loss_G', avg_G_fake_loss)

            print('Epoch [%03d/%03d], Iter [%03d/%03d], D_real: %.4f, D_real_c: %.4f, D_fake: %.4f, G_fake: %.4f, G_fake_c: %.4f, G_recon: %.4f, KLD: %.4f'
                  % (epoch + 1, args.epochs, i + 1, len(train_loader), avg_D_real_loss / (i + 1),
                     avg_D_real_c_loss / (i + 1), avg_D_fake_loss / (i + 1),
                     avg_G_fake_loss / (i + 1), avg_G_fake_c_loss / (i + 1),
                     avg_G_recon_loss / (i + 1), avg_kld / (i + 1)))

            if batches_done % SAMPLE_INTERVAL == 0:
                gen_images = fake.mul(0.5).add(0.5)
                save_image(gen_images, "%s/images/gan_img_%d.png" %
                           (save_folder, batches_done), nrow=8, normalize=True)

                np_images = torchvision.utils.make_grid(
                    gen_images).detach().cpu().numpy().transpose(1, 2, 0)
                experiment.log_image(np_images)

                torch.save(G, os.path.join(
                    save_folder,
                    str(batches_done) + '_generator.pt'))

                torch.save(D, os.path.join(
                    save_folder,
                    str(batches_done) +  '_discriminator.pt'))

            batches_done += 1

        # Evaluation of recall score

        metrics = eval_gan_batch(data_loader_dev, G, ranker, args.data_set, word_embedding)
        experiment.log_metric('r10', metrics['r10'])
        experiment.log_metric('r50', metrics['r50'])
        experiment.log_metric('r1000', metrics['r1000'])
        experiment.log_metric('ranking', metrics['mean_rank'])
        experiment.log_metric('target_distance', metrics['mean_dist_from_target'])

        if metrics['r50'] > best_score:
            best_score = metrics['r50']
            experiment.log_metric('best_score', best_score)
#             # save best model
            torch.save(G, os.path.join(
                save_folder,
                'generator_{}.pt'.format(batches_done)))

            torch.save(D, os.path.join(
                save_folder,
                'discriminator_{}.pt'.format(batches_done)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='../models/tagan',
                        help='path for saving trained models')
    parser.add_argument('--img_root', type=str, default='/home/datasets/style_search-yosh_ai/images/',
                        help='root directory that contains images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--gen_img_size', type=int, default=128,
                        help='size for generated images')
    parser.add_argument('--log_step', type=int, default=3,
                        help='step size for printing log info')
    parser.add_argument('--shop', type=int, choices=['answear', 'asos', 'zalando', 'gerryweber', 'all'])

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=1024,
                        help='dimension of word embedding vectors')
    # Learning parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--optimizer', type=str,
                        default='adam', choices=['rmsprop', 'adam'])
    parser.add_argument('--gan_type', type=str, default='tagan')
    parser.add_argument('--lr_D', type=float, default=0.0002)
    parser.add_argument('--lr_G', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='learning rate decay (dafault: 0.5)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_cond_loss', type=float, default=10,
                        help='lambda of conditional loss (default: 10)')
    parser.add_argument('--lambda_recon_loss', type=float, default=0.2,
                        help='lambda of reconstruction loss (default: 0.2)')

    args = parser.parse_args()

    IMG_SIZE = args.crop_size
    DATA_SET = args.data_set
    BATCH_SIZE = args.batch_size
    EMBEDDING_SIZE = args.embed_size
    GEN_IMG_SIZE = args.gen_img_size

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    train(args)

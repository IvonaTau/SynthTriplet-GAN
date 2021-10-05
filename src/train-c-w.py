import argparse
from comet_ml import Experiment
import time
import os
import numpy as np
import json
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch import autograd
from torch.autograd import Variable
import torchvision

from data_loader import get_loader
from build_vocab import Vocabulary
from models import DummyImageEncoder, DummyCaptionEncoder, CNN_ENCODER, RNN_ENCODER
from utils import create_exp_dir, Ranker, EncoderRanker, ResnetRanker, tensor_to_img, to_np, recall, compute_score, caption_to_description, mean_ranking
import wgan
from wgan import create_noise, calc_gradient_penalty, generate_images

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Paths to data
CAPT = '../data/captions/cap.{}.{}.json'
DICT = '../data/captions/dict.{}.json'
SPLIT = '../data/image_splits/split.{}.{}.json'
image_encoder_path = '../models/start-kit/toptee-20191115-110235/image-1024.th'
text_encoder_path = '../models/start-kit/toptee-20191115-110235/cap-1024.th'

TOP_K = 500


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)


def eval_gan_batch(data_loader, image_encoder, caption_encoder, netG,  ranker, dataset, vocab):
    ranker.update_emb(image_encoder, batch_size=args.batch_size,
                      crop_size=args.crop_size)
    rankings = []
    loss = []
    distances = []
    output = json.load(open(CAPT.format(dataset, 'val')))
    index = 0

    for i, (target_images, candidate_images, captions, lengths, meta_info) in enumerate(data_loader):
        with torch.no_grad():
            candidate_images = candidate_images.to(device)
            target_images = target_images.to(device)
            candidate_ft = image_encoder.forward(candidate_images)
            captions = captions.to(device)
            descriptions = [caption_to_description(c, vocab) for c in captions]
            caption_ft = caption_encoder(captions, lengths)
            concatenated = torch.cat((candidate_ft, caption_ft), 1)
            concatenated = concatenated.view(-1, EMBEDDING_SIZE*2)
            try:
                gen_img = netG(concatenated).view(-1, 3, 64, 64)
            except:
                gen_img = netG(concatenated)[0]

            gen_imgs_feats = ranker.embed(
                gen_img, image_encoder, crop_size=args.crop_size).detach()
            target_imgs_feats = ranker.embed(
                target_images, image_encoder, crop_size=args.crop_size).detach()

            rankings = ranker.get_nearest_neighbors(gen_imgs_feats, TOP_K)

            curr_dist = (target_imgs_feats -
                         gen_imgs_feats).pow(2).sum(dim=1).mean()
#             curr_dist = torch.norm(target_imgs_feats - gen_imgs_feats, dim=1, p=2).mean()
            distances.append(curr_dist.cpu().numpy())

            for j in range(rankings.size(0)):
                output[index]['candidate'] = meta_info[j]['candidate']
                output[index]['target'] = meta_info[j]['target']
                output[index]['ranking'] = [ranker.data_asin[rankings[j, m].item()]
                                            for m in range(rankings.size(1))]
                index += 1
    recall = compute_score(output, output)
    mean_rank = mean_ranking(output)
    metrics = {}

    metrics['r10'] = recall[0]
    metrics['r50'] = recall[1]
    metrics['r1000'] = recall[2]
    metrics['mean_rank'] = mean_rank
    metrics['mean_dist_from_target'] = np.mean(distances)
    return metrics


def initialize_models(resume_path, loss_type, args, vocab):

    # ----------------------
    #  Data loaders
    # ----------------------

    transform = transforms.Compose([
        transforms.Resize(int(GEN_IMG_SIZE*1.3)),
        transforms.RandomCrop(GEN_IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    transform_dev = transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    transform_dev_gan = transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_target = transforms.Compose([
        transforms.Resize(int(GEN_IMG_SIZE*1.3)),
        transforms.RandomCrop(GEN_IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # ---------------------
    #  Initialize models
    # ---------------------

    if resume_path == '':
        if loss_type == 'wgan':
            netD = wgan.Discriminator(3, GEN_IMG_SIZE).cuda()
            netG = wgan.Generator(EMBEDDING_SIZE*2, 64, GEN_IMG_SIZE).cuda()

            data_loader = get_loader(IMAGE_ROOT,
                                     CAPT.format(DATA_SET, 'train'),
                                     vocab, transform,
                                     BATCH_SIZE, shuffle=True, return_target=True,
                                     num_workers=0, transform_target=transform_target)

            data_loader_dev = get_loader(IMAGE_ROOT,
                                         CAPT.format(args.data_set, 'val'),
                                         vocab, transform_dev,
                                         BATCH_SIZE, shuffle=True, return_target=True, num_workers=0, transform_target=transform_target)

        elif loss_type == 'cgan':
            netD = wgan.Discriminator_cgan(3, GEN_IMG_SIZE).cuda()
            netG = wgan.Generator(EMBEDDING_SIZE*2, 64, GEN_IMG_SIZE).cuda()

            data_loader = get_loader(IMAGE_ROOT,
                                     CAPT.format(DATA_SET, 'train'),
                                     vocab, transform,
                                     BATCH_SIZE, shuffle=True, return_target=True,
                                     num_workers=0, transform_target=transform_target)

            data_loader_dev = get_loader(IMAGE_ROOT,
                                         CAPT.format(args.data_set, 'val'),
                                         vocab, transform_dev,
                                         BATCH_SIZE, shuffle=True, return_target=True, num_workers=0, transform_target=transform_target)

        elif loss_type == 'sagan':
            netD = sagan.Discriminator(
                args.batch_size, GEN_IMG_SIZE, 64).cuda()
            netG = sagan.Generator(
                args.batch_size, GEN_IMG_SIZE, EMBEDDING_SIZE*2, 64).cuda()

            data_loader = get_loader(IMAGE_ROOT,
                                     CAPT.format(DATA_SET, 'train'),
                                     vocab, transform_gan,
                                     BATCH_SIZE, shuffle=True, return_target=True,
                                     num_workers=0, transform_target=transform_target)

            data_loader_dev = get_loader(IMAGE_ROOT,
                                         CAPT.format(args.data_set, 'val'),
                                         vocab, transform_dev_gan,
                                         BATCH_SIZE, shuffle=True, return_target=True, num_workers=0, transform_target=transform_target)

        else:
            raise ValueError('Wrong loss type.')
    else:
        # resume training
        print('Resuming training')
        gen_pth = os.path.join(resume_path, 'generator_last.pt')
        netG = torch.load(gen_pth)
        d_pth = os.path.join(resume_path, 'discriminator_last.pt')
        netD = torch.load(d_pth)

        data_loader = get_loader(IMAGE_ROOT,
                                 CAPT.format(DATA_SET, 'train'),
                                 vocab, transform,
                                 BATCH_SIZE, shuffle=True, return_target=True,
                                 num_workers=0, transform_target=transform_target)

        data_loader_dev = get_loader(IMAGE_ROOT,
                                     CAPT.format(args.data_set, 'val'),
                                     vocab, transform_dev,
                                     BATCH_SIZE, shuffle=True, return_target=True, num_workers=0, transform_target=transform_target)

    return netD, netG, data_loader, data_loader_dev


def train(args):

    transform_gan = transforms.Compose([
        transforms.Resize(int(GEN_IMG_SIZE*1.3)),
        transforms.RandomCrop(GEN_IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    vocab = Vocabulary()
    vocab.load(DICT.format(DATA_SET))

    if args.ranker == 'encoder':
        ranker = EncoderRanker(root=IMAGE_ROOT, image_split_file=SPLIT.format(args.data_set, 'val'),
                               transform=transform_gan, num_workers=0)
    elif args.ranker == 'resnet':
        ranker = ResnetRanker(root=IMAGE_ROOT, image_split_file=SPLIT.format(args.data_set, 'val'),
                              transform=transform_gan, num_workers=0)
    elif args.ranker == 'simple':
        ranker = Ranker(root=IMAGE_ROOT, image_split_file=SPLIT.format(args.data_set, 'val'),
                        transform=transform_gan, num_workers=0)

    save_folder = '{}/{}-{}'.format(args.save,
                                    args.data_set, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(save_folder, scripts_to_save=[
                   'train-gan.py', 'models.py', 'wgan.py'])

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

    netD, netG, data_loader, data_loader_dev = initialize_models(
        args.resume, args.loss_type, args, vocab)

    image_encoder = DummyImageEncoder(
        EMBEDDING_SIZE, backbone=args.backbone).to(device)
    image_encoder.load_state_dict(torch.load(image_encoder_path), strict=False)

    caption_encoder = DummyCaptionEncoder(vocab_size=len(vocab), vocab_embed_size=EMBEDDING_SIZE * 2, embed_size=EMBEDDING_SIZE).to(device)
    
    caption_encoder.load_state_dict(torch.load(text_encoder_path))

    for param in image_encoder.parameters():
        param.requires_grad = False

    for param in caption_encoder.parameters():
        param.requires_grad = False

    image_encoder.eval()
    caption_encoder.eval()

    # ----------------------
    #  Optimizers and losses
    # ----------------------

    if args.optimizer == 'rmsprop':
        optimizer_D = torch.optim.RMSprop(netD.parameters(), lr=args.lr_D)
        optimizer_G = torch.optim.RMSprop(netG.parameters(), lr=args.lr_G)
    elif args.optimizer == 'adam':
        optimizer_D = torch.optim.Adam(netD.parameters(), args.lr_D, [
                                       args.beta1, args.beta2])
        optimizer_G = torch.optim.Adam(netG.parameters(), args.lr_G, [
                                       args.beta1, args.beta2])

    if args.loss_type == 'sagan':
        optimizer_G = torch.optim.Adam(filter(
            lambda p: p.requires_grad, netG.parameters()), args.lr_G, [args.beta1, args.beta2])
        optimizer_D = torch.optim.Adam(filter(
            lambda p: p.requires_grad, netG.parameters()), args.lr_D, [args.beta1, args.beta2])
        args.optimizer = 'adam'

    triplet_avg = nn.TripletMarginLoss(reduction='elementwise_mean', margin=1)
    c_loss = torch.nn.CrossEntropyLoss()

    # ----------------------
    #  Training
    # ----------------------

    D_ITERS = 2
    total_step = len(data_loader)
    LOG_STEP = 100

    total_step
    EPOCHS = args.epochs
    W_CLIP = 0.01
    N_CRITIC = args.dicr_iter
    SAMPLE_INTERVAL = 200

    batches_done = 0
    best_score = float('-inf')

    for epoch in range(EPOCHS):
        print('Epoch:', epoch)

        for i, (target_images,
                candidate_images, captions,
                lengths, meta_info) in enumerate(data_loader):

            # Set to training mode
            netD.train()
            netG.train()

            # Image input
            target_images = target_images.to(device)
            candidate_images = candidate_images.to(device)
            candidate_ft = image_encoder.forward(
                F.interpolate(candidate_images, size=args.crop_size))
            # Text input
            captions = captions.to(device)
            caption_ft = caption_encoder(captions, lengths)
            # Concatenate image and text features
            concatenated = torch.cat((candidate_ft, caption_ft), 1)
            concatenated = concatenated.view(-1, EMBEDDING_SIZE*2)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            fake_imgs = netG(concatenated).detach()

            if args.loss_type == 'cgan':
                disc_fake = netD(fake_imgs, caption_ft)
                disc_real = netD(target_images, caption_ft)
                # Weserstein
                gradient_penalty = calc_gradient_penalty(netD, target_images, fake_imgs, len(
                    lengths), device, args.gen_img_size, caption_ft=caption_ft)
            else:
                disc_fake = netD(fake_imgs)
                disc_real = netD(target_images)
                # Weserstein
                gradient_penalty = calc_gradient_penalty(
                    netD, target_images, fake_imgs, len(lengths), device, args.gen_img_size)

            disc_fake = disc_fake.mean()
            disc_real = disc_real.mean()

            # Triplet loss
            gen_imgs = fake_imgs.view(-1, 3,
                                      args.gen_img_size, args.gen_img_size)
            m = target_images.size(0)
            random_index = [m - 1 - n for n in range(m)]
            random_index = torch.LongTensor(random_index)
            negative_images = target_images[random_index]

            target_images_feats = ranker.embed(
                target_images, image_encoder, crop_size=args.crop_size)
            gen_imgs_feats = ranker.embed(
                gen_imgs, image_encoder, crop_size=args.crop_size)
            negative_images_feats = ranker.embed(
                negative_images, image_encoder, crop_size=args.crop_size)
            source_images_feats = ranker.embed(
                candidate_images, image_encoder, crop_size=args.crop_size)

            loss_D_triplet = triplet_avg(anchor=gen_imgs_feats,
                                         positive=target_images_feats,
                                         negative=source_images_feats)

            # Adversarial loss
            loss_D = disc_fake - disc_real + gradient_penalty + args.b_t*loss_D_triplet
            loss_D.backward()
            optimizer_D.step()

            experiment.log_metric('loss_D', loss_D.item())
            experiment.log_metric('loss_D_fake', disc_fake.item())
            experiment.log_metric('loss_D_real', disc_real.item())

            # Train the generator every n_critic iterations
            if i % N_CRITIC == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()
                gen_imgs = netG(concatenated).view(-1, 3,
                                                   args.gen_img_size, args.gen_img_size)
                # Adversarial loss
                if args.loss_type == 'cgan':
                    loss_G_adversarial = - \
                        torch.mean(netD(gen_imgs, caption_ft))
                else:
                    loss_G_adversarial = -torch.mean(netD(gen_imgs))

                # Triplet loss
                m = target_images.size(0)
                random_index = [m - 1 - n for n in range(m)]
                random_index = torch.LongTensor(random_index)
                negative_images = target_images[random_index]

                target_images_feats = ranker.embed(
                    target_images, image_encoder, crop_size=args.crop_size)
                gen_imgs_feats = ranker.embed(
                    gen_imgs, image_encoder, crop_size=args.crop_size)
                negative_images_feats = ranker.embed(
                    negative_images, image_encoder, crop_size=args.crop_size)
                source_images_feats = ranker.embed(
                    candidate_images, image_encoder, crop_size=args.crop_size)

                loss_G_triplet = triplet_avg(anchor=gen_imgs_feats,
                                             positive=target_images_feats,
                                             negative=negative_images_feats)
                loss_G = loss_G_adversarial + args.a_t*loss_G_triplet

                loss_G.backward()

                optimizer_G.step()
                experiment.log_metric(
                    'loss_G_adversarial', loss_G_adversarial.item())
                experiment.log_metric('loss_G_triplet', loss_G_triplet.item())

                experiment.log_metric('loss_G', loss_G.item())

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss fake: %f] [G loss: %f]"
                    % (epoch, EPOCHS, batches_done % len(data_loader), len(data_loader), disc_fake.item(), loss_G.item())
                )

            if batches_done % SAMPLE_INTERVAL == 0:
                gen_images = generate_images(netG, concatenated, BATCH_SIZE)
#                 save_image(gen_images, "../gan_output/%s/gan_%d.png" % (DATA_SET, batches_done), nrow=8, normalize=True)
                save_image(gen_images, "%s/images/gan_img_%d.png" %
                           (save_folder, batches_done), nrow=8, normalize=True)

                np_images = torchvision.utils.make_grid(
                    gen_images).detach().cpu().numpy().transpose(1, 2, 0)
                experiment.log_image(np_images)

                torch.save(netG, os.path.join(
                    save_folder,
                    'generator_last.pt'))

                torch.save(netD, os.path.join(
                    save_folder,
                    'discriminator_last.pt'))

            batches_done += 1

        # Evaluation of recall score
        image_encoder.eval()
        caption_encoder.eval()

        metrics = eval_gan_batch(data_loader_dev, image_encoder, caption_encoder,
                                 netG, ranker, dataset=args.data_set, vocab=vocab)
        experiment.log_metric('r10', metrics['r10'])
        experiment.log_metric('r50', metrics['r50'])
        experiment.log_metric('r1000', metrics['r1000'])
        experiment.log_metric('ranking', metrics['mean_rank'])
        experiment.log_metric(
            'target_distance', metrics['mean_dist_from_target'])

        if metrics['r50'] > best_score:
            best_score = metrics['r50']
            experiment.log_metric('best_score', best_score)
            # save best model
            resnet = image_encoder.delete_resnet()
            torch.save(image_encoder.state_dict(), os.path.join(
                save_folder,
                'image-{}.th'.format(args.embed_size)))
            image_encoder.load_resnet(resnet)

            torch.save(caption_encoder.state_dict(), os.path.join(
                save_folder,
                'cap-{}.th'.format(args.embed_size)))

            torch.save(netG, os.path.join(
                save_folder,
                'generator_{}.pt'.format(batches_done)))

            torch.save(netD, os.path.join(
                save_folder,
                'discriminator_{}.pt'.format(batches_done)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='/home/datasets/fashion-iq/models/gan',
                        help='path for saving trained models')
    parser.add_argument('--img_root', type=str, default='../data/resized/',
                        help='root directory that contains images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--gen_img_size', type=int, default=64,
                        help='size for generated images')

    parser.add_argument('--data_set', type=str, default='toptee')
    parser.add_argument('--log_step', type=int, default=3,
                        help='step size for printing log info')
    parser.add_argument('--patient', type=int, default=3,
                        help='patient for reducing learning rate')
    parser.add_argument('--a_t', type=int, default=1,
                        help='weights for triplet loss in generator')
    parser.add_argument('--b_t', type=int, default=0,
                        help='weights for triplet loss in discriminator')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=1024,
                        help='dimension of word embedding vectors')
    # Learning parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument(
        '--encoder', choices=['dummy', 'DAMSM', 'resnet50', 'resnet152'], default='dummy')
    parser.add_argument(
        '--backbone', choices=['resnet152', 'resnet50', 'resnet18'], default='resnet152')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--optimizer', type=str,
                        default='rmsprop', choices=['rmsprop', 'adam'])
    parser.add_argument('--loss_type', type=str, default='cgan',
                        choices=['wgan', 'cgan', 'sagan'])
#     parser.add_argument('--adv_loss', type=str, default='hinge', choices=['wgan-gp', 'hinge'], help='Only used in SAGAN. Type of adversarial loss.')
    parser.add_argument('--lr_D', type=float, default=4e-5)
    parser.add_argument('--lr_G', type=float, default=1e-5)
    parser.add_argument('--dicr_iter', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--ranker', type=str, default='encoder',
                        choices=['simple', 'encoder', 'resnet'])

    args = parser.parse_args()

    IMG_SIZE = args.crop_size
    DATA_SET = args.data_set
    BATCH_SIZE = args.batch_size
    EMBEDDING_SIZE = args.embed_size
    GEN_IMG_SIZE = args.gen_img_size
    IMAGE_ROOT = args.img_root

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    train(args)

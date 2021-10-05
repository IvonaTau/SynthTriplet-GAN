# python3.7 train-latents.py
import argparse
from comet_ml import Experiment
import time
import os
import numpy as np
import json
import fasttext
import pdb
from bert_serving.client import BertClient

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

from data_loader import get_loader, EmbeddingDataset, collate_fn
from build_vocab import Vocabulary
from models import DummyImageEncoder, DummyCaptionEncoder, CNN_ENCODER, RNN_ENCODER, MLP
from utils import create_exp_dir, Ranker, EncoderRanker, ResnetRanker, tensor_to_img, to_np, recall, compute_score, caption_to_description, mean_ranking

import warnings
warnings.simplefilter("ignore")

DICT = '../data/captions/dict.{}.json'
CAPT = '../data/captions/cap.{}.{}.json'


def train(args):
    
    vocab = Vocabulary()
    vocab.load(DICT.format(args.data_set))
    
    img_transform = transforms.Compose([
        transforms.Resize(int(1.06*args.crop_size)),

        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(args.crop_size),
        transforms.ToTensor()
    ])
    
    print('Loading a dataset...')
    
    train_data = EmbeddingDataset(args.img_root,
                                  args.embeddings_root,
                                  CAPT.format(args.data_set, 'val'),
                                  vocab, 
                                  transform=img_transform)
    print('Training data length:', len(train_data))
    val_data =  EmbeddingDataset(args.img_root,
                                  args.embeddings_root,
                                  CAPT.format(args.data_set, 'val'),
                                  vocab, 
                                  transform=img_transform)
    print('Validation data length:', len(val_data))
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True)
    
    img_encoder = ResnetRanker(root=args.img_root, num_workers=0)
        
    caption_encoder = DummyCaptionEncoder(vocab_size=len(vocab), vocab_embed_size=args.embed_size*2, embed_size=args.embed_size).to(device)
    caption_encoder = DummyCaptionEncoder(vocab_size=len(vocab), vocab_embed_size=1024*2, embed_size=1024).to(device)
    caption_encoder.load_state_dict(torch.load(args.text_encoder_path))
    for param in caption_encoder.parameters():
        param.requires_grad = False
    caption_encoder.eval()
    
    save_folder = '{}/{}-{}'.format(args.save,
                                    args.data_set, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(save_folder, scripts_to_save=['train-latents.py'])
    
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(save_folder, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')
                
    def emb_distance(x1, x2):
            return torch.mean((x1-x2).pow(2).sum(dim=1))

    logging(str(args))
    hyper_params = vars(args)
    hyper_params['save_dir'] = save_folder

    experiment = Experiment(api_key="njYbdzqt7zdPd6vz9qCklcuMH",
                            project_name="multimodal-latents", workspace="ivonatau")
    experiment.log_parameters(hyper_params)

    model = MLP(hidden_layers= [args.embed_size_1, args.embed_size_2],
               in_dim = args.mlp_input,
               out_dim = args.gan_latent_dim,
               dropout_prob=args.dropout_prob).cuda()
    
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, [args.beta1, args.beta2])
    
    if args.loss == 'mse':
        loss = nn.MSELoss()
    elif args.loss == 'me':
        loss = torch.nn.L1Loss()
    elif args.loss == 'dist':
        loss = emb_distance
    else:
        raise ValueError('Wrong loss type!')
    min_val_loss = float('inf')
    
    # ----------------------
    #  Training
    # ----------------------
        
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        
        for i, (_, source_img, captions, lengths, latent_code, _) in enumerate(train_loader):                       
            model.train()
                        
            source_img = source_img.to(device)
            captions = captions.to(device)
            latent_code = latent_code.to(device)
            
            #Inputs
            source_ft = img_encoder.embed(source_img)
            
            caption_ft = caption_encoder(captions, lengths)
#             caption_ft = torch.from_numpy(bert_client.encode(list(captions))).cuda()
            if args.gan_latent_dim == 512:
                latent_code = torch.mean(latent_code, 2).squeeze(1)
            if args.mlp_input == 3072:
                concatenated = torch.cat((source_ft, caption_ft), 1)
            elif args.mlp_input == 4096:
                concatenated = torch.cat((source_ft, caption_ft, caption_ft), 1)
            else:
                raise ValueError('Wrong args.mlp_input !')
            #Train 
            
            optimizer.zero_grad()
            model_output = model(concatenated)
#             print('Model output variance:', torch.var(model_output))
            experiment.log_metric('Model predicted variance', torch.var(model_output).item())
            experiment.log_metric('Batch ground truth variance', torch.var(latent_code).item())
            if args.gan_latent_dim == 8192:
                model_output = model_output.view((-1, 16, 512))
            
#             model_output = torch.clamp(model_output, min=-1.0, max=1.0)
            
            curr_loss = loss(model_output, latent_code)
#             if curr_loss > 200000:
#                 pdb.set_trace()
                
            curr_loss.backward()
            optimizer.step()
            
            experiment.log_metric('loss', curr_loss.item())
            print(i, curr_loss)
            
        #Evaluate on valid
        losses = []
        for i, (_, source_img, captions, lengths, latent_code, _) in enumerate(train_loader):
            with torch.no_grad():
                
                source_img = source_img.to(device)
                captions = captions.to(device)
                latent_code = latent_code.to(device)
            
                #Inputs
                source_ft = img_encoder.embed(source_img)
                caption_ft = caption_encoder(captions, lengths)
#                 caption_ft = torch.from_numpy(bert_client.encode(list(captions))).cuda()

                if args.gan_latent_dim == 512:
                    latent_code = torch.mean(latent_code, 2).squeeze(1)
                #Concatenate image and text
                if args.mlp_input == 3072:
                    concatenated = torch.cat((source_ft, caption_ft), 1)
                elif args.mlp_input == 4096:
                    concatenated = torch.cat((source_ft, caption_ft, caption_ft), 1)
                else:
                    raise ValueError('Wrong args.mlp_input !')
                model_output = model(concatenated)
                if args.gan_latent_dim == 8192:
                    model_output = model_output.view((-1, 16, 512))
                curr_loss = loss(model_output, latent_code)
                losses.append(curr_loss.cpu().numpy())
        
        epoch_val_loss = np.mean(losses)
        experiment.log_metric('val_loss', epoch_val_loss)
        print('-----------------------Validation loss:', epoch_val_loss)
        if epoch_val_loss < min_val_loss:
            torch.save(model, os.path.join(save_folder, 'latent_encoder_best.pt'))
            torch.save({'state_dict': model.state_dict()}, os.path.join(save_folder, 'checkpoint_best.pth.tar'))
            
            

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument('--save', type=str, default='/home/datasets/fashion-iq/models/stylegan_embeds',
                        help='path for saving trained models')
    parser.add_argument('--img_root', type=str, default='../data/toptee_512/',
                        help='root directory that contains images')
    parser.add_argument('--caption_root', type=str, default='../data/captions_vec',
                        help='root directory that contains captions')
    parser.add_argument('--embeddings_root', type=str, default='../data/stylegan_latents/',
                        help='root directory that extracted GAN latent codes from images')
#     parser.add_argument('--text_encoder_path', type=str, default='/home/datasets/fashion-iq/models/start-kit/dress-20190925-135258/cap-1024.th')
    parser.add_argument('--text_encoder_path', type=str, default='/home/datasets/fashion-iq/models/start-kit/toptee-20191115-110235/cap-1024.th')
    parser.add_argument('--trainclasses_file', type=str, default='toptee.txt',
                        help='text file that contains training classes')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='size for randomly cropping images')
    parser.add_argument('--data_set', type=str, default='toptee')
    parser.add_argument('--log_step', type=int, default=3,
                        help='step size for printing log info')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=2048,
                        help='dimension of image and text embedding vectors')
    parser.add_argument('--mlp_input', type=int, default=4096,
                        help='dimension of concatenated image and text embedding vectors - 3072 or 4096')
    parser.add_argument('--gan_latent_dim', type=int, default=512,
                        help='output of MLP model of latent vector used as an input to StyleGAN model')
    parser.add_argument('--embed_size_1', type=int, default=8192,
                        help='dimension of MLP embedding layer #1')
    parser.add_argument('--embed_size_2', type=int, default=8192,
                        help='dimension of MLP embedding layer #2')
    # Learning parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dropout_prob', type=float, default=0.0)
#     parser.add_argument('--a_t', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
#     parser.add_argument('--ranker', type=str, default='resnet')
    parser.add_argument('--optimizer', type=str,
                        default='adam', choices=['rmsprop', 'adam'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss', type=str, default='dist', choices=['mse', 'me', 'dist'])
    parser.add_argument('--epochs', type=int, default=30)
#     parser.add_argument('--lr_decay', type=float, default=0.5,
#                         help='learning rate decay (dafault: 0.5)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    


    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    train(args)

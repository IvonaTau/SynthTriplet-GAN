# Description

## Makefile

In general, the project is managed through `Makefile`. 

You can use it to build (with command `make docker-build`) and run Docker images and do other things. See the comments of individual Makefile goals in the file for details. 

Also, at the top of the `Makefile`, you can find which environment variables are required by the `Makefile`.


## Requirements
torch == 1.3.1
torchvision == 0.4.2


# Data preparation

For new dataset prepare annotation files in `data/captions` and `data/image_splits`. 

Then build vocabulary by running:

```
python3 build_vocab.py --data_set shoes
```

# Training

Pre-train image and text encoders:

```
CUDA_VISIBLE_DEVICES=2 python3.7 train.py --data_set=shoes --img_root=/home/datasets/style_search-dialog/images
```

Train SynthTriplet GAN with triplet loss ($L_C$):

```
CUDA_VISIBLE_DEVICES=2 python3.7 train-c-w.py --loss_type=cgan --img_root=../datasets/images --data_set=shoes
```


Train SynthTriplet GAN with triplet loss and gradient penalty ($L_W$):

```
CUDA_VISIBLE_DEVICES=2 python3.7 train-c-w.py --loss_type=wgan --img_root=../datasets/images --data_set=shoes
```



Train SynthTriplet GAN with triplet loss, text-adaptive discriminator and reconstruction loss ($L_T$):
```
python3.7 train-t.py --img_root=../datasets/images --data_set=shoes --trainclasses_file=shoes.txt --a_t=10 --ranker=encoder --freeze=gan

```


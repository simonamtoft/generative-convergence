# Investigating Convergence Metrics for Deep Generative Models

Modern Generative Models have achieved impressive results in Computer Vision, Natural Language Processing, and Density Estimation. However such results rely on brittle models that achieve convergence through optimization tricks, heavy hyperparameter optimization, and domain knowledge. In this project, we aim to study and compare the robustness of Deep Latent Variable and Flow Models with a focus on the role of random initialization (seed) on the training dynamics.

The project is carried out by [Simon Amtoft Pedersen](https://github.com/simonamtoft), and supervised by [Giorgio Giannone](https://georgosgeorgos.github.io/) and [Ole Winther](https://orbit.dtu.dk/en/persons/ole-winther).


## Adapting Autoencoders to Two-dimensional Toy Data

In order to train autoencoders such as the standard variational autoencoder (VAE) or a hierarchical autoencoder such as the ladder variational autoencoder (LVAE), the training approach has to be adapted. These models have a generative part given by <img src="https://latex.codecogs.com/svg.image?%5Cbg_black%20p(z)%20=%20p(x%7Cz)%5Ccdot%20p(z)"/>, where we try to model the original distribution of the data with `p`, such that we can draw samples with

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_black&space;z&space;\sim&space;p(z)" title="\bg_black z \sim p(z)" />

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_black&space;x&space;\sim&space;p(x|z)" title="\bg_black x \sim p(x|z)" />


. This is done by encoding the original data into the latent space, using the distribution `q`, such that reconstructed images can be created by drawing from 
<img src="https://bit.ly/3mJ0sUq"/>
.

The training of either of these two models optimize the evidence lower bound (ELBO) <img src="https://bit.ly/3Jtd8Zm" align="center" border="0" alt="\mathcal{L}_{ELBO}(x) = \mathcal{L}_{KL} + \mathcal{L}_{recon}" width="196" height="18" />, consisting of a reconstruction and a Kullback–Leibler (KL) divergence term. The difference in training on two-dimensional data instead of standard image data like MNIST is the way in which we use the reconstruction term, while the KL term is calculated by the encoder and decoder distributions <img src="https://bit.ly/3pzQcQn" align="center" border="0" alt="\mathcal{L}_{KL} = KL(q||p) = \mathbb{E}_q \left[log \frac{p(z)}{q(z|x)} \right]" width="312" height="47" />. For binarized MNIST data, the reconstruction term is simply the binary cross-entropy loss between the original images and the reconstructed images achieved from a pass through of the model. 

A way to adapt the autoencoder models is to model each dimension of the data with a mean and variance, such that for two-dimensional data the size of the decoder output is 4. Then a likelihood distribution is created from this decoder output, <img src="https://bit.ly/3pCjtKk" align="center" border="0" alt="ax + b = c" width="89" height="17" /> from which we compute the log probability of the input two-dimensional data point to originate from such a distribution, which is then our reconstruction term.




## Convergence Metrics
In order to compare convergence of different models, each model is trained a number of times (e.g. 10), from which metrics on the test losses for each epoch during training can be calculated. For each of these runs, a different random seed is used for all used procedures (see [seed_everything](https://github.com/simonamtoft/generative-convergence/blob/main/lib/random_seed.py)).


### FFJORD Toy Data



### Binarized MNIST 


## Model Training

Training a model on toy data can be run by simply

```cmd
python train_toy.py -m <model_name> -d <dataset_name> -e <n_epochs> -mute -n <n_seeds>
```

Training a model on MNIST data can be run by simply

```cmd
python train.py -m <model_name> -e <n_epochs> -mute -n <n_seeds>
```


Where

- `model_name` picks the model architecture to train. Can be either of the implemented models: `vae`, `lvae`, `flow` or `draw` (Note: the DRAW model is not implemented for training on the toy data).
- `dataset_name` picks the toy data to train on. Can be either of the datasets from FFJORD.
- `n_epochs` decides the number of epochs to train over.
- `-mute` mutes the output from tqdm, don't use flag if you want the output.
- `n_seeds` decides the number of training runs to do on different seeds. If you just want to train a single model, don't set this flag (default is 1).

Example for no mute train of Ladder VAE on the 8gaussians dataset.

```cmd
python train_toy.py -m lvae -d 8gaussians -e 500
```

The used torch install

```pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html```


## References
This project uses model implementations from the following repositories:

- [VAE, LadderVAE, DRAW models](https://github.com/simonamtoft/recurrence-and-attention-latent-variable-models)
- [Flow model](https://github.com/didriknielsen/survae_flows)

The toy data is from the [FFJORD paper](https://arxiv.org/abs/1810.01367) (arXiv:1810.01367)

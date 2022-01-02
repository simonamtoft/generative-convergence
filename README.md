# Investigating Convergence Metrics for Deep Generative Models

Modern Generative Models have achieved impressive results in Computer Vision, Natural Language Processing, and Density Estimation. However such results rely on brittle models that achieve convergence through optimization tricks, heavy hyperparameter optimization, and domain knowledge. In this project, we aim to study and compare the robustness of Deep Latent Variable and Flow Models with a focus on the role of random initialization (seed) on the training dynamics.

The project is carried out by [Simon Amtoft Pedersen](https://github.com/simonamtoft), and supervised by [Giorgio Giannone](https://georgosgeorgos.github.io/) and [Ole Winther](https://orbit.dtu.dk/en/persons/ole-winther).


## Adapting Autoencoders to Two-dimensional Toy Data

In order to train autoencoders such as the standard variational autoencoder (VAE) or a hierarchical autoencoder such as the ladder variational autoencoder (LVAE), the training approach has to be adapted. These models have a generative part given by <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_black&space;p(z)&space;=&space;p(x|z)\cdot&space;p(z)" title="\bg_black p(z) = p(x|z)\cdot p(z)" />, where we try to model the original distribution of the data with `p`, such that we can draw samples with <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_black&space;z&space;\sim&space;p(z)" title="\bg_black z \sim p(z)" /> and <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_black&space;x&space;\sim&space;p(x|z)" title="\bg_black x \sim p(x|z)" />. This is done by encoding the original data into the latent space, using the distribution `q`, such that reconstructed images can be created by drawing from <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_black&space;q(z|x)" title="\bg_black q(z|x)" />.

The training of either of these two models optimize the evidence lower bound (ELBO) <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_black&space;\mathcal{L}_{ELBO}&space;=&space;\mathcal{L}_{KL}&space;&plus;&space;\mathcal{L}_{recon}" title="\bg_black \mathcal{L}_{ELBO} = \mathcal{L}_{KL} + \mathcal{L}_{recon}" />, consisting of a reconstruction and a Kullback–Leibler (KL) divergence term. The difference in training on two-dimensional data instead of standard image data like MNIST is the way in which we use the reconstruction term, while the KL term is calculated by learned distributions with: 

<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_black&space;\mathcal{L}_{KL}&space;=&space;KL(q||p)&space;=&space;\mathbb{E}_q&space;\left[log&space;\frac{p(z)}{q(z|x)}&space;\right]" title="\bg_black \mathcal{L}_{KL} = KL(q||p) = \mathbb{E}_q \left[log \frac{p(z)}{q(z|x)} \right]" /> 

For binarized MNIST data, the reconstruction term is simply the binary cross-entropy loss between the original images and the reconstructed images achieved from a pass through of the model. 

A way to adapt the autoencoder models is to model each dimension of the data with a mean and variance, such that the decoder outputs these <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_black&space;[\mu_x,&space;\sigma_x]&space;\sim&space;x&space;" title="\bg_black [\mu_x, \sigma_x] \sim x " />. Then a likelihood distribution is created from this decoder output, <img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_black&space;\mathcal{N}(\mu_x,&space;\sigma_x)" title="\bg_black \mathcal{N}(\mu_x, \sigma_x)" /> from which we compute the log probability of the input two-dimensional data point to originate from such a distribution, which is then our reconstruction term (see [implementation](https://github.com/simonamtoft/generative-convergence/blob/main/trainers_toy/train_vae.py#L69)).



## Convergence Metrics
In order to compare convergence of different models, each model is trained a number of times (e.g. 10), from which metrics on the test losses for each epoch during training can be calculated. For each of these runs, a different random seed is used for all used procedures (see [seed_everything](https://github.com/simonamtoft/generative-convergence/blob/main/lib/random_seed.py)). Then, the min, max and final test loss is taken from each run, along with computing the mean over the first five epochs. This is done for each of the 10 runs of each model on the different datasets, where the mean is taken over the 10 runs which has different random initializations.


### FFJORD Toy Data
For the FFJORD toy data, we've chosen to look at the `8gaussians` and `checkerboard` toy data (see [gen_data](https://github.com/simonamtoft/generative-convergence/blob/main/lib/gen_data.py#L18)). For each of the two datasets, a set of Flow, VAE and LVAE models are trained and computed metrics for, as shown on the plots and table below.


![metrics 8gaussians](./losses/metrics_8gaussians.png)
![metrics checkerboard](./losses/metrics_checkerboard.png)


|               | model   |   min |   max |   last |   first |   exceeds |
|---------------|:--------|------:|------:|-------:|--------:|----------:|
|  8gaussians   | flow    |  2.91 |  3.36 |   2.95 |    3.06 |       9.9 |
|               | lvae    |  5.82 | 11.04 |   5.82 |    9.29 |       0   |
|               | vae     |  5.82 | 16.11 |   5.82 |   10.3  |       0   |
|  checkerboard | flow    |  3.58 |  4.23 |   3.63 |    3.79 |       9.5 |
|               | lvae    |  6.92 | 12.42 |   6.92 |   10.48 |       0   |
|               | vae     |  6.91 | 16.86 |   6.91 |   11.52 |       0   |

### Binarized MNIST 

Additionally, metrics are computed using the binarized MNIST data, where a set fo DRAW, Flow, VAE and LVAE models are trained and computed metrics for, as shown on the plots and table below.

![metrics mnist](./losses/metrics_mnist.png)

| model   |    min |           max |   last |   first |   exceeds |
|:--------|-------:|--------------:|-------:|--------:|----------:|
| draw    |  86.33 | 373.11        |  86.52 |  221.09 |       0   |
| flow    | 162.19 | 4.64727e+18   | 176.54 |  267.04 |       2.2 |
| lvae    | 125.87 | 306.51        | 126.11 |  252.48 |       0   |
| vae     | 124.65 | 283.04        | 125.07 |  243.93 |       0.1 |


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

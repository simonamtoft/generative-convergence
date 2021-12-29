# Investigation of Convergence of Generative Models

Modern Generative Models have achieved impressive results in Computer Vision, Natural Language Processing, and Density Estimation. However such results rely on brittle models that achieve convergence through optimization tricks, heavy hyperparameter optimization, and domain knowledge. In this project, we aim to study and compare the robustness of Deep Latent Variable and Flow Models with a focus on the role of random initialization (seed) on the training dynamics.



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

- `model_name` picks the model architecture to train. Can be either of the implemented models: `vae`, `lvae`, `flow` or `draw`.
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

# bivariate_causal_inference_gan

This repository contains generator (gen.py) and discriminator (disc.py) modules for bivariate causal inference modeling in GAN framework, according to the additive noise model [1].

## Generator

Generator from Gaussian noise generates a set of samples $y = f(x) + \epsilon$, where noise $\epsilon$ is independent of $x$, and $f(.)$ is a piecewise linear, monotonous function. 
Example output can be generated by executing 
```
python gen.py
```

## Discriminator

The discriminator is a DeepSet neural network (average pooling). It accepts two inputs tensors $X$ and $Y$ of size $(n_b, dim_x, n)$, and $(n_b, dim_y, n)$, where $(n_b)$ is batch size, $dim_x$ and $dim_y$ are corresponding dimensions, and $n$ is sample size.

## Bibliography

[1] https://proceedings.neurips.cc/paper_files/paper/2008/file/f7664060cc52bc6f3d620bcedc94a4b6-Paper.pdf

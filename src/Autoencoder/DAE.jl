# DENOISING AUTOENCODER
using Flux
using Images
using Flux.Data, MLDatasets
using Flux.Data:DataLoader
using Noise
device = cpu 
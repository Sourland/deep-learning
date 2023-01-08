# DENOISING AUTOENCODER
using Flux
using Images
using Flux.Data, MLDatasets
using Flux.Data:DataLoader
using Noise

include("train.jl")
device = cpu 

device = cpu # where will the calculations be performed?

x_train, y_train = MNIST.traindata(Float32);
x_test, y_test = MNIST.testdata(Float32);

x_train = 0.299 * x_train[:,:,1,:] + 0.587* x_train[:,:,2,:] + 0.114 * x_train[:,:,3,:]
x_test = 0.299 * x_test[:,:,1,:] + 0.587* x_test[:,:,2,:] + 0.114 * x_test[:,:,3,:]

m, n = size(x_test[:,:,1])

x_train = reshape(x_train, (m*n,size(x_train)[3]))
x_test = reshape(x_test, (m*n,size(x_test)[3]))

# add some salt pepper noise to the image
x_train_noise = add_gauss(x_train, 0.15) 
x_test_noise = add_gauss(x_test, 0.15) 
loader = DataLoader((data = x_train_noise, label = x_train), batchsize=512, shuffle=true)

input_size = m*n

DAE1 = Chain(
    #ENCODER
    Dense(input_size, input_size),
    BatchNorm(input_size, relu),

    Dense(input_size, input_size),

    Dense(input_size, input_size),
    BatchNorm(input_size, relu),

    Dense(input_size, input_size, sigmoid)
    
   ) |> device


DAE2 = Chain(
#ENCODER
Dense(input_size, 128),
BatchNorm(128, relu),

Dense(128, 16),

Dense(16,128),

# Dense(input_size, input_size),
BatchNorm(128, relu),

Dense(128, input_size, sigmoid)

) |> device

loss(x, y) = Flux.Losses.mse(DAE2(x), y)

dae_params = Flux.params(DAE2)
optim = ADAM(0.05)

train!(loss, dae_params, optim, loader, 35)
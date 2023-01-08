using Flux
using Images
using Flux.Data, MLDatasets
using Flux.Data:DataLoader
using Noise

include("train.jl")

device = cpu # where will the calculations be performed?

x_train, y_train = MNIST.traindata(Float32);
x_test, y_test = MNIST.testdata(Float32);

m, n = size(x_test[:,:,1])

x_train = reshape(x_train, (m*n,size(x_train)[3]))
x_test = reshape(x_test, (m*n,size(x_test)[3]))

loader = DataLoader((data = x_train, label = x_train), batchsize=512, shuffle=true)


autoencoder1 = Chain(
    #ENCODER
    Dense(m*n, 512, relu),
    Dense(512, 256, relu),
    Dense(256, 64, relu),
    Dense(64, 8, relu),
    
    #DECODER
    Dense(8, 64, relu),
    Dense(64, 256, relu),
    Dense(256, 512, relu),
    Dense(512, m*n, sigmoid)
   ) |> device


autoencoder2 = Chain(
    #ENCODER
    Dense(m*n, 256, relu),
    Dense(256, 64, relu),
    Dense(64, 8, relu),

    #DECODER
    Dense(8, 64, relu),
    Dense(64, 256, relu),
    Dense(256, m*n,sigmoid)
    ) |> device

autoencoder3 = Chain(
    #ENCODER
    Dense(m*n, 64, relu),
    Dense(64, 8, relu),

    #DECODER
    Dense(8, 64, relu),
    Dense(64, m*n, sigmoid),
    ) |> device

loss1(x, y) = Flux.Losses.mse(autoencoder1(x), y)
loss2(x, y) = Flux.Losses.mse(autoencoder2(x), y)
loss3(x, y) = Flux.Losses.mse(autoencoder3(x), y)

opt = ADAM(0.05)
params1 = Flux.params(autoencoder1)
params2 = Flux.params(autoencoder2)
params3 = Flux.params(autoencoder3) # parameters

# train!(loss,1 params1, opt, loader, 10)
# train!(loss2, params2, opt, loader, 35)
train!(loss3, params3, opt, loader, 35)
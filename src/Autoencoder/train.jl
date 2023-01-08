using Flux

function train!(model_loss, model_params, opt, loader, epochs = 10)
    train_steps = 0
    "Start training for total $(epochs) epochs" |> println
    for epoch = 1:epochs
        print("Epoch $(epoch): ")
        ℒ = 0
        for x in loader
            loss, back = Flux.pullback(model_params) do
                model_loss(x[1], x[2])
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, model_params, grad)
            train_steps += 1
            ℒ += loss
        end
        println("ℒ = $ℒ")
    end
    "Total train steps: $train_steps" |> println
end

function fit(model, data, optimizer, loss_func)
Loss(x,y) = loss_func(x,y)
model_params = Flux.params(model)

end
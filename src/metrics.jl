using Plots

function plot_metrics(metrics)
    title = "Model Cross Entropy Loss"
    label = ["Training" "Evaluation"]
    fig_loss = plot([metrics["loss"], metrics["val_loss"]], title = title, label = label, linewidth = 1.5)
    xlabel!("Epoch")
    ylabel!("L(ŷ, y)")
    png("Loss");

    title = "Model Accuracy"
    label = ["Training" "Evaluation"]
    fig_acc = plot([metrics["accuracy"], metrics["val_accuracy"]], title = title, label = label, linewidth = 1.5)
    xlabel!("Epoch")
    ylabel!("Accuracy")
    png("Accuracy");
end
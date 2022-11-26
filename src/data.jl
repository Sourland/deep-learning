using MLDatasets:CIFAR10
x_train, y_train = CIFAR10.traindata();
x_test, y_test = CIFAR10.testdata();
x_train = 0.299 * x_train[:,:,1,:] + 0.587* x_train[:,:,2,:] + 0.114 * x_train[:,:,3,:]
x_test = 0.299 * x_test[:,:,1,:] + 0.587* x_test[:,:,2,:] + 0.114 * x_test[:,:,3,:]
x_train = reshape(x_train, (32*32,size(x_train)[3]))
x_test = reshape(x_test, (32*32,size(x_test)[3]))



function one_hot_encoding!( number_of_classes, labels)
    encoded_labels = zeros(number_of_classes, size(labels)[1])
    for idx in 1:size(labels)[1]
        encoded_labels[labels[idx]+1,idx] = 1
    end
    return encoded_labels
end

y_train = one_hot_encoding!(maximum(y_train)+1, y_train)
y_test = one_hot_encoding!(maximum(y_test)+1, y_test)
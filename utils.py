"""
    Problem Set 5: Recognition Using Deep Networks
    Joseph Nelson Farrell
    5330 Computer Vision and Pattern Recognition
    Northeastern University
    Bruce Maxwell, PhD.
    4.1.2024
    
    This file contains a library of utility function related to project 5.

    Functions List (in order):

        1. get_load_digits_data
        2. train_loop
        3. test_loop
        4. get_saved_model
        5. execute_model_n_times
        6. apply_layer_1_filters
        7. GreekTransform (class)
        8. get_transform_greek_letters
"""
# import libraries
import torch
import torchvision
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as opt
from model_class import NeuralNetwork
import cv2

###################################################################################################################################################
def get_load_digits_data(batch_size_train: int, batch_size_test: int) -> tuple:
    """
        Function: get_load_digits_data
            This function will get the digits data, transform it, and return two dataloaders,
            train and test.

        Parameters:
            batch_size_train: (int) - training batch size.
            batch_size_test: (int) - test batch size.

        Returns: 
            train_loader: digits data read to be used in model training.
            test_loader:  digits data read to be used in model testing.
    """

    train_loader = torch.utils.data.DataLoader(
                        datasets.MNIST(
                            root = "digits_data",
                            train = True,
                            download = True,
                            transform = torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize( (0.1307, ), (0.3081, ))
                            ])
                        ),
                batch_size = batch_size_train
            )

    test_loader = torch.utils.data.DataLoader(
                            datasets.MNIST(
                                root = "digits_data",
                                train = False,
                                download = True,
                                transform = torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize( (0.1307, ), (0.3081, ))
                                ])
                        ),
                    batch_size = batch_size_test
                )
    return train_loader, test_loader

###################################################################################################################################################
def train_loop(dataloader, model, optimizer, train_losses_list, train_accuracy, train_counter_list, epoch, log_interval, batch_size_train):
    """
        Function: train_loop
            This function will execute a training loop for a neural network.

        Parameters:
            dataloader: data to be used to train a model.optimizer, train_losses_list, train_accuracy, train_counter_list, epoch, log_interval, batch_size_train
            model: (NeuralNetwork) - the model the function will train.
            optimizer: optimizer for the model.
            train_losses_list: list to store losses to track to training performance.
            train_accuracy: list to store accuracy score to track the training performance.
            train_counter_list: list of samples seen.
            epoch: the number of epochs to train.
            log_interval: used to determine when to pull out trackers.
            batch_size_train: the data batch size.

        Returns: 
            train_losses_list: list of losses to track to training performance.
            train_accuracy_list of accuracy score to track the training performance.
            train_counter_list: list of samples seen.
    """
    # track accuracy
    correct = 0
    total = 0

    # get the size of the data
    size = len(dataloader.dataset)

    # set to training mode
    model.train()

    # iterate over batches
    for batch, (X, y) in enumerate(dataloader):

        # set gradient to zero
        optimizer.zero_grad()

        # compute prediction and loss
        out = model(X)
        loss = F.nll_loss(out, y)

        # get the predicted label
        _, predicted = torch.max(out.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        # back propagation
        loss.backward()
        optimizer.step()

        if batch % log_interval == 0:
            print(f'Loss: {loss.item()} [{batch * batch_size_train}/{size}] ({100.0 * batch / len(dataloader)})')

            # update trackers
            train_losses_list.append(loss.item())
            train_counter_list.append(((batch * batch_size_train) + (epoch - 1) * size))
            
    # update accuracy
    accuracy = (correct / total) * 100
    train_accuracy.append(accuracy)
    print(f'Train Accuracy: {accuracy}')

    return train_losses_list, train_accuracy, train_counter_list

###################################################################################################################################################
def test_loop(dataloader, model, test_accuracy, test_losses):
    """
        Function: test_loop
            This function will test model performance on test data.

        Parameters:
            dataloader: data to be used to train a model.optimizer, train_losses_list, train_accuracy, train_counter_list, epoch, log_interval, batch_size_train
            model: (NeuralNetwork) - the model the function will train.
            optimizer: optimizer for the model.
            test_losses: list to store losses to track to training performance.
            test_accuracy: list to store accuracy score to track the training performance.

        Returns: 
            test_losses: list of losses to track to training performance.
            test_accuracy: list of accuracy score to track the training performance.
    """
    # get the size
    size = len(dataloader.dataset)

    # set to evaluate mode
    model.eval()

    # trackers
    test_loss = 0
    correct = 0

    # set to no gradient
    with torch.no_grad():
        for X, y in dataloader:
            output = model(X)
            test_loss += F.nll_loss(output, y, size_average = False).item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()
    test_loss /= size
    test_losses.append(test_loss)
    accuracy = (100 * correct / size)
    test_accuracy.append(accuracy)

    print(f"Test Error: \n \
            Accuracy : {accuracy:>0.1f}% \n \
            Average Loss: {test_loss:>8f}\n")
    
    return test_accuracy, test_losses
    

###################################################################################################################################################
def get_saved_model(model_path, optimizer_path, learning_rate, momentum):
    """
        Function: get_saved_model
            This function will get a saved model and optimizer state.

        Parameters:
            dataloader: data to be used to train a model.optimizer, train_losses_list, train_accuracy, train_counter_list, epoch, log_interval, batch_size_train
            model: (NeuralNetwork) - the model.
            optimizer: optimizer for the model.
            test_losses: list to store losses to track to training performance.
            test_accuracy: list to store accuracy score to track the training performance.

        Returns: 
            test_losses: list of losses to track to training performance.
            test_accuracy: list of accuracy score to track the training performance.
    """

    # insantiate model and optimizer
    model_conti = NeuralNetwork()
    continued_optimizer = opt.SGD(model_conti.parameters(), lr = learning_rate, momentum = momentum)

    # load saved model and optimizer 
    model_conti.load_state_dict(torch.load(model_path))
    continued_optimizer.load_state_dict(torch.load(optimizer_path))

    return model_conti, continued_optimizer

###################################################################################################################################################
def execute_model_n_times(model, test_loader, num_images):
    """
        Function: execute_model_n_times
            This function will execute a model on a specified number of data.

        Parameters:
            dataloader: data to be used to train a model.optimizer, train_losses_list, train_accuracy, train_counter_list, epoch, log_interval, batch_size_train
            model: (NeuralNetwork) - the model to use.
            num_images: the number of imaged to test.

        Returns: 
            true_labels: the true labels of the test data.
            predicted_labels: the predicted labels returned by the model.
            probas: the probabilites for each of the classes for each image.
    """
    # set model to evaluation mode
    model.eval()

    # containers for plots
    true_labels = []
    predicted_labels = []
    probas = []

    # tracker
    t = 0

    # iterate over first 10 images in test_loader
    for images, labels in test_loader:
        for i in range(len(images)):

            # set to no gradient
            with torch.no_grad():
                output = model(images[i])
                print('*' * 50)
                print(f'Actual Class: {labels[i]}')
                probabilities = torch.softmax(output, dim = 1)
                predicted_class = torch.argmax(probabilities, dim = 1)
                print(f"Predicted class (index): {predicted_class.item()}")
                print(f'Class Probabilities: {output[0]}')

                # update containers
                true_labels.append(labels[i])
                predicted_labels.append(predicted_class.item())
                probas.append(output[0])
            
            # update tracker
            t += 1

            # end inner loop
            if t == num_images:
                break
        # outer loop   
        if t == num_images:
            break

    return true_labels, predicted_labels, probas

###################################################################################################################################################
def apply_layer_1_filters(first_image, filters):
    """
        Function: apply_layer_1_filter
            This function will apply the filters of the first layer of the model

        Parameters:
            first_image: image data
            filters: tensor of filters

        Returns: 
            filter_images_list: a list of the image after the filters have been applied
    """

    filter_images_list = []

    for i in range(len(filters)):
        filter = filters[i, 0]
        filtered_image = cv2.filter2D(first_image, -1, filter)
        filter_images_list.append(filtered_image)

    return filter_images_list

###################################################################################################################################################
# greek data set transform
class GreekTransform:
    """
        class: GreekTransfrom

        This class will transform a directory of Greek letter images into the appropriate size and shape.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

###################################################################################################################################################
def get_transform_greek_letters(training_set_path, batch_size, shuffle_status):
    """ 
        Functions: get_transform_greek_letters
            This function will get the greek letters from a directory, it transfrom using GreekTransform and
            normalize, and return a dataloader.

        Parameters:
            training_set_path: (str) - the path to the directory containing the images.
            batch_size: (int) - the size the training batches.
            shuffle_status: (bool) - whether or not the shuffle the digits data.

        Returns:
            greek_letters: (data_loader) - the greek letters ready to be put in a NeuralNetwork
    """

    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( training_set_path,
                                            transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                        GreekTransform(),
                                                                                        torchvision.transforms.Normalize(
                                                                                            (0.1307,), (0.3081,) ) ] ) ),
        batch_size = batch_size,
        shuffle = shuffle_status )
    
    return greek_train


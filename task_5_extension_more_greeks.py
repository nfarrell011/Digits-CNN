"""
    Problem Set 5: Recognition Using Deep Networks: Extension
    Joseph Nelson Farrell & Harshil Bhojwani 
    5330 Computer Vision and Pattern Recognition
    Northeastern University
    Bruce Maxwell, PhD.
    4.1.2024
    
    This file will read in the trained NeuralNetwork. It change the last layer to accomdate the number classes,
    here that will be 5.

    It will then read in and transform an example set of hand drawn Greek letters. It will train the NeuralNetwork on 
    these letters and plot the results.
"""
# functions are in seperate files, utils and utils_plots
from utils import get_saved_model
from utils import train_loop
from utils import GreekTransform
from utils import get_transform_greek_letters
from utils_plots import plot_greek_accuracy_and_loss
from utils_plots import plot_example_images_with_labels

# read in packages
import torch.optim as opt
import os
import torch
from torch import nn
import torchvision
import numpy as np
import sys

###################################################################################################################################################
# main
def main(argv):
    """
        This function will read in the two sets of Greek letters and train the model on both.
        It will produce figures of the results.

        Parameters: 
            None

        Returns:
            None
    """
    # set parameters
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 42
    batch_size = 5
    shuffle_status = True

    # containers
    train_losses_list = []
    train_accuracy_list = []
    train_counter_list = []

    # sets paths
    model_file = 'results/model.pth'
    optimizer_file = 'results/optimizer.pth'
    homebrew_greek_letters_folder = 'greek_train_homebrew_2'

    # set torch settings
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # set the current path
    path = os.getcwd()

    # set paths to weights and optimizer
    model_path = os.path.join(path, model_file)
    optimizer_path = os.path.join(path, optimizer_file)

    # check if the figs folder exists
    figs_folder = 'figs'
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder, exist_ok=True)
        print(f"The folder '{figs_folder}' was created.")
    else:
        print(f"The folder '{figs_folder}' already exists.")

    # check the files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model File Not Found!")
    if not os.path.exists(optimizer_path):
        raise FileNotFoundError("Optimizer File Not Found!")

    # get the model
    model, optimizer = get_saved_model(model_path, optimizer_path, learning_rate, momentum)

    # turn off gradient
    for param in model.parameters():
        param.requires_grad = False

    # turn on last layer gradient
    model.fc2 = nn.Linear(50, 5) 
    for param in model.fc2.parameters():
        param.requires_grad = True

    # update the optimizer      
    optimizer = opt.SGD(model.parameters(),
                            lr = learning_rate, momentum=momentum)

    # set the path to greek letters folder
    greek_letters_folder = os.path.join(path, homebrew_greek_letters_folder)
    if not os.path.exists(greek_letters_folder):
        raise FileNotFoundError("Greek Letters Folder Not Found!")

    # get the greek letters
    greek_train = get_transform_greek_letters(greek_letters_folder, batch_size, shuffle_status)

    # extract examples to plot
    examples = enumerate(greek_train)
    batch_idx, (example_data, example_target) = next(examples)

    # plot some examples
    greek_letter_example_fig = 'greek_example_images_2.png'
    plot_example_images_with_labels(example_data, example_target, 5, path, greek_letter_example_fig)

    # train model for 12 epochs
    n_epochs = 11
    for i in range(n_epochs):
        train_losses_list, train_accuracy_list, train_counter_list = train_loop(greek_train, model, optimizer, train_losses_list, train_accuracy_list, 
                                                                            train_counter_list, i, log_interval, batch_size)
    # plot results
    epochs_for_x_axis = np.arange(1, 12, 1)
    figure_name = "greek_letters_losses_and_accuracy_2.png"
    title_suffix = "Greek Letters 2"
    plot_greek_accuracy_and_loss(epochs_for_x_axis, train_accuracy_list, train_losses_list, title_suffix, path, figure_name)


if __name__ == "__main__":
    main(sys.argv)
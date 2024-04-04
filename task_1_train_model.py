"""
    Problem Set 5: Recognition Using Deep Networks: Task 1
    Joseph Nelson Farrell
    5330 Computer Vision and Pattern Recognition
    Northeastern University
    Bruce Maxwell, PhD.
    4.1.2024
    
    This file will read in the MNIST digits and train the neural network (NeuralNetwork) defined in model_class.py.
    It will create figures of the traing and testing results and save the train model to a file.
"""

# functions and classes are in seperate files
from model_class import NeuralNetwork
from utils import get_load_digits_data
from utils_plots import plot_example_images_with_labels
from utils import train_loop
from utils import test_loop
from utils_plots import plot_loss_with_respect_to_samples
from utils_plots import plot_accuracy_with_respect_to_samples

# packages
from torchsummary import summary
from torchviz import make_dot
import torch
import torch.optim as opt
import os
import sys

##################################################### Main ####################################################################
# main
def main(argv):
    """
        This function will train and test NeuralNetwork

        Pamameters:
            * n_epochs: (int) - optional - number of the training epochs

        Returns:
            * None
    """
    # set parameters
    if len(argv) > 1:
        n_epochs = argv[1]
    else:
        n_epochs = 10
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 42
    num_to_plot = 6

    # set torch settings
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # set the current path
    path = os.getcwd()

    # check if the figs folder exists
    figs_folder = 'figs'
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder, exist_ok=True)
        print(f"The folder '{figs_folder}' was created.")
    else:
        print(f"The folder '{figs_folder}' already exists.")

    # check if the results folder exists
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder, exist_ok = True)
        print(f"The folder '{results_folder}' was created.")
    else:
        print(f"The folder '{results_folder}' already exists.")

    # load network
    model = NeuralNetwork()

    # get the data loaders for digits
    train_loader, test_loader = get_load_digits_data(batch_size_train, batch_size_test)

    # extract examples to plot
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_target) = next(examples)
    example_images_file_name = "example_images.png"
    plot_example_images_with_labels(example_data, example_target, num_to_plot, path, example_images_file_name)

    # plot model structure
    figs_folder_path = path + "/figs"
    print("Path:", figs_folder_path)
    y = model(example_data[1])
    make_dot(y, params = dict(list(model.named_parameters()))).render(os.path.join(figs_folder_path, "model_viz"), format = "png")

    # define optimizer
    optimizer = opt.SGD(model.parameters(), lr = learning_rate, momentum = momentum)

    # tracking containers
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [n * len(train_loader.dataset) for n in range(n_epochs + 1)]
    train_accuracy = []
    test_accuracy = []

    # train model
    test_loop(test_loader, model, test_accuracy, test_losses)
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch} \n ---------------------------------------')
        train_losses, train_accuracy, train_counter = train_loop(train_loader, model, optimizer, train_losses, train_accuracy, train_counter, epoch, log_interval, batch_size_train)
        test_accuracy, test_losses = test_loop(test_loader, model, test_accuracy, test_losses)

    # save network and optimizer
    torch.save(model.state_dict(), "results/model.pth")
    torch.save(optimizer.state_dict(), "results/optimizer.pth")

    # plot results
    losses_with_respect_samples_file_name = "losses_with_respect_to_samples.png"
    plot_loss_with_respect_to_samples(test_losses, train_losses, test_counter, train_counter, path, losses_with_respect_samples_file_name)

    accuracy_with_respect_to_samples = "accuracy_with_respect_to_sample.png"
    plot_accuracy_with_respect_to_samples(test_counter, test_accuracy, train_accuracy, path, accuracy_with_respect_to_samples)

if __name__ == "__main__":
    main(sys.argv)
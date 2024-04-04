"""
    Problem Set 5: Recognition Using Deep Networks: Task 1
    Joseph Nelson Farrell & Harshil Bhojwani 
    5330 Computer Vision and Pattern Recognition
    Northeastern University
    Bruce Maxwell, PhD.
    4.1.2024
    
    This file will perform a gridsearch of 3 NeuralNetworks and two hyperparameters and plot the results.
"""
# functions are classes are defined in seperate files
from model_class import NeuralNetwork, NeuralNetwork_2, NeuralNetwork_3
from utils_plots import plot_grid_search_results
from utils import get_load_digits_data
from utils import train_loop
from utils import test_loop

# import packages
from torchviz import make_dot
import torch
import torch.optim as opt
import pandas as pd
import os
import sys

############################################### Main ########################################################################
# main 
def main(argv):
    """
        Function: main
            This function will test the trained model on the first 10 digits in the digits dataset

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
    batch_sizes_train = [32, 64, 128]
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 42

    # set torch settings
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # set the current path
    path = os.getcwd()

    # check if the figs folder exists
    figs_folder = 'figs'
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder, exist_ok = True)
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

    # load networks 
    model_1 = NeuralNetwork()
    model_2 = NeuralNetwork_2()
    model_3 = NeuralNetwork_3()

    # this will used during the grid search to restart the network
    instantiator_list = [NeuralNetwork, NeuralNetwork_2, NeuralNetwork_3]

    # get the data loaders for digits
    train_loader, test_loader = get_load_digits_data(batch_sizes_train[0], batch_size_test)

    # extract examples to plot
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_target) = next(examples)

    # plot model structure for model_2 and model_2
    figs_folder_path = path + "/figs"
    y = model_2(example_data[1])
    make_dot(y, params = dict(list(model_2.named_parameters()))).render(os.path.join(figs_folder_path, "model_2_viz"), format = "png")
    y = model_3(example_data[1])
    make_dot(y, params = dict(list(model_3.named_parameters()))).render(os.path.join(figs_folder_path, "model_3_viz"), format = "png")

    # define optimizers
    optimizer = opt.SGD(model_1.parameters(), lr = learning_rate, momentum = momentum)
    optimizer_2 = opt.SGD(model_2.parameters(), lr = learning_rate, momentum = momentum)
    optimizer_3 = opt.SGD(model_3.parameters(), lr = learning_rate, momentum = momentum)

    # container for overall gridsearch results
    results_dict = {}

    # iterators for the first run through
    models_list = [model_1, model_2, model_3]
    optimizer_list = [optimizer, optimizer_2, optimizer_3]

    # tracker to restart model
    i = 0

    # iterate over models
    for idx, (model, optimizer) in enumerate(zip(models_list, optimizer_list)):

        # individual model results container
        model_results = {}

        # iterate over batches
        for batch_size in batch_sizes_train:
            
            # load data with new batch size
            train_loader, test_loader = get_load_digits_data(batch_size, batch_size_test)

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
                train_losses, train_accuracy, train_counter = train_loop(train_loader, model, optimizer, train_losses, train_accuracy, train_counter, epoch, log_interval, batch_size)
                test_accuracy, test_losses = test_loop(test_loader, model, test_accuracy, test_losses)

                # update model container
                model_results[batch_size] = {
                    "train_losses": train_losses,
                    "train_accuracy": train_accuracy,
                    "train_counter": train_counter,
                    "test_losses": test_losses,
                    "test_accuracy": test_accuracy,
                    "test_counter": test_counter
                }

            # restart model and optimizer for next iter
            model = instantiator_list[i]()
            optimizer = opt.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
        
        # update tracker
        i += 1

        # update results dict
        results_dict[f"model_{idx + 1}"] = model_results

    # convert dict to frame
    results_frame = pd.DataFrame(results_dict)

    # plot and save results
    save_name = "gridsearch_results.png"
    plot_grid_search_results(results_frame, batch_sizes_train, path, save_name)

if __name__ == "__main__":
    main(sys.argv)


  


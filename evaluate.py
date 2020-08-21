"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import models.net as net
import models.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    for val_batch, loclabels_batch, memlabels_batch in dataloader:
            # move to GPU if available
            # if params.cuda:
            #     val_batch, q8labels_batch = val_batch.cuda(non_blocking=True), q8labels_batch.cuda(non_blocking=True)
            # if params.cuda:
            #     q3labels_batch = q3labels_batch.cuda(non_blocking=True)

            # compute model output and loss
            # N x 3 x 1632, N x 6 x 1632
            locoutput_batch, memoutput_batch = model(val_batch)


            locloss = loss_fn(locoutput_batch.cpu(), loclabels_batch)
            memloss = loss_fn(memoutput_batch.cpu(), memlabels_batch)

            loss = locloss + memloss
            
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            locoutput_batch = locoutput_batch.data.cpu().numpy()
            memoutput_batch = memoutput_batch.data.cpu().numpy()
            memlabels_batch = memlabels_batch.data.numpy()
            loclabels_batch = loclabels_batch.data.numpy()

            # mask shape = N x 1632
            # compute all metrics on this batch
            summary_batch = {'val_locaccuracy': metrics['Loc_accuracy'](locoutput_batch, loclabels_batch), 'val_memaccuracy': metrics['Mem_accuracy'](memoutput_batch, memlabels_batch)}
            summary_batch['val_loss'] = loss.item()
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Validation metrics: " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation..")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

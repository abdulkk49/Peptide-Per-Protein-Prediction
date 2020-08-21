"""Train the model"""

import argparse
import logging
import os, sys

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import utils
from evaluate import evaluate
from os.path import join, exists, dirname, abspath, realpath

sys.path.append(dirname(abspath("__file__")))

from models.data_loader import *
from models.net import *
from transformers import BertModel, BertTokenizer
import re
import os
import requests, h5py
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./Embeddings',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='./trial',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--batch', default=128,
                    help="Batch Size for Training")
parser.add_argument('--num_workers', default=1,
                    help="Num workers for Training")

def download_file(url, filename):
    response = requests.get(url, stream=True)
    with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                        total=int(response.headers.get('content-length', 0)),
                        desc=filename) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)


def embedModel():
    pwd = dirname(realpath("__file__"))
    print("Present Working Directory: ", pwd)
    #Pretrained Model files
    modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
    configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
    vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'

    #Setting folder paths
    downloadFolderPath = 'models/ProtBert/'
    modelFolderPath = downloadFolderPath

    #Setting file paths
    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
    configFilePath = os.path.join(modelFolderPath, 'config.json')
    vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')

    #Creading model directory
    if not os.path.exists(modelFolderPath):
        os.makedirs(modelFolderPath)

    #Downloading pretrained model
    if not os.path.exists(modelFilePath):
        download_file(modelUrl, modelFilePath)
    if not os.path.exists(configFilePath):
        download_file(configUrl, configFilePath)
    if not os.path.exists(vocabFilePath):
        download_file(vocabUrl, vocabFilePath)

    #Initializing Tokenizer, Model
    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )
    model = BertModel.from_pretrained(modelFolderPath)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, tokenizer



bertModel, tokenizer = embedModel()
bertModel = bertModel.eval()

def collate_fn(batch):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    # print(input_ids.shape, attention_mask.shape)
    embedding = torch.zeros((input_ids.shape[0],1024), dtype = torch.float32).to(device)
    i = 0
    bs = 32
    x = 0
    with torch.no_grad():
        while i + bs <= len(input_ids):
            e = bertModel(input_ids=input_ids[i:i+bs],attention_mask=attention_mask[i:i+bs])[0]
            # print(e.shape)
            for seq_num in range(len(e)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = e[seq_num, 1:seq_len-1, :]
                seq_emd = torch.mean(seq_emd, -2)
                # print(seq_emd.shape)
                embedding[x,:] = seq_emd
                x += 1
        
            i += bs
            print(e.shape, i)

    with torch.no_grad(): 
    	if i != len(input_ids):
            #Final batch < 32
            e = bertModel(input_ids=input_ids[i:len(input_ids)],attention_mask=attention_mask[i:len(input_ids)])[0]
            # print(e.shape)
            for seq_num in range(len(e)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = e[seq_num, 1:seq_len-1, :]
                seq_emd = torch.mean(seq_emd, -2)
                # print(seq_emd.shape)
                embedding[x,:] = seq_emd
                x += 1
            print(e.shape, i)

    loclabel = torch.stack([item[2] for item in batch])
    memlabel = torch.stack([item[3] for item in batch])
    print(loclabel.shape, memlabel.shape)
    return embedding, loclabel, memlabel


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, loclabels_batch, memlabels_batch) in enumerate(dataloader):
            # move to GPU if available
            # if params.cuda:
            #     train_batch, q8labels_batch, mask = train_batch.cuda(non_blocking=True),\
            #     q8labels_batch.cuda(non_blocking=True), mask.cuda(non_blocking = True)
            # compute model output and loss
            # N x 10, N x 3
            locoutput_batch, memoutput_batch = model(train_batch)

            locloss = loss_fn(locoutput_batch.cpu(), loclabels_batch)
            memloss = loss_fn(memoutput_batch.cpu(), memlabels_batch)

            loss = locloss + memloss
            # clear previous gradients, compute gradients of all variables wrt loss
            
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                locoutput_batch = locoutput_batch.data.cpu().numpy()
                memoutput_batch = memoutput_batch.data.cpu().numpy()
                memlabels_batch = memlabels_batch.data.numpy()
                loclabels_batch = loclabels_batch.data.numpy()
                # compute all metrics on this batch
                summary_batch = {'loc_accuracy': metrics['Loc_accuracy'](locoutput_batch, loclabels_batch), 'mem_accuracy': metrics['Mem_accuracy'](memoutput_batch, memlabels_batch)}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_locacc = 0.0
    best_val_memacc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_locacc = val_metrics['val_locaccuracy']
        val_memacc = val_metrics['val_memaccuracy']
        is_locbest = val_locacc >= best_val_locacc
        is_membest = val_memacc >= best_val_memacc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                                is_locbest = is_locbest,
                                is_membest = is_membest,
                                checkpoint = model_dir)

        # If best_eval, best_save_path
        if is_locbest:
            logging.info("- Found new best loc accuracy")
            best_val_locacc = val_locacc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_locbest_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        if is_membest:
            logging.info("- Found new best mem accuracy")
            best_val_memacc = val_memacc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_membest_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)


        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

if __name__ == '__main__':

    # Load the parameters from json file
    pwd = dirname(realpath("__file__"))
    print("Present Working Directory: ", pwd)
    args = parser.parse_args()
    json_path = os.path.join(pwd, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    params.batch_size = int(args.batch)
    params.num_workers = int(args.num_workers)
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
    print("Params: " ,params.__dict__)

    #Load sequences
    sequences_Example =[]
    count = 0
    with open("./train/sequences.txt", "r") as f:
        for seq in f.readlines():
            desc = str(seq).rstrip('\n')
            sequences_Example.append(desc)
            count += 1
    print("Total training data points(Clean): ", str(count))
    
    #Replace "UZOB" with "X"
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Tokenizing input sequences
    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)
    train_input_ids = torch.tensor(ids['input_ids']).to(device)
    train_attention_mask = torch.tensor(ids['attention_mask']).to(device)
    # Set the logger

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    train_dl = fetch_dataloader('train', 'loclabels.txt', 'memlabels.txt', train_input_ids, train_attention_mask, params, collate_fn)
    
    del train_input_ids
    del train_attention_mask

    sequences_Example =[]
    count = 0
    with open("./val/sequences.txt", "r") as f:
        for seq in f.readlines():
            desc = str(seq).rstrip('\n')
            sequences_Example.append(desc)
            count += 1
    print("Total validation data points(Clean): ", str(count))
    
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)
    val_input_ids = torch.tensor(ids['input_ids']).to(device)
    val_attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    val_dl = fetch_dataloader('val', 'loclabels.txt', 'memlabels.txt', val_input_ids, val_attention_mask, params, collate_fn)

    del val_input_ids
    del val_attention_mask
    del sequences_Example

    logging.info("- done.")

    # Define the model and optimizer
    model = ProteinNet(params)
    if params.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = loss_fn
    metrics = metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, join(pwd, "trial"),
                        None)

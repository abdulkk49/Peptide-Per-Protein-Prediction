import csv
import os
import sys
from itertools import groupby
from Bio import SeqIO
import random
from sklearn import preprocessing


test_loc = []
test_mem = []
test_seq = []
trainval_loc = []
trainval_mem = []
trainval_seq = []
def load_dataset(path_fasta):
    """
    Loads dataset into memory from fasta file
    """
    fasta_sequences = SeqIO.parse(open(path_fasta),'fasta')
    
    for fasta in fasta_sequences:
        desc = fasta.description.split(" ")
        labels = desc[1].split("-")
        if len(labels) > 2:
          continue
        loclabel, memlabel, sequence = labels[0], labels[1], str(fasta.seq)
        if len(desc) > 2:
          test_loc.append(loclabel)
          test_mem.append(memlabel)
          test_seq.append(sequence)
        else:
          trainval_loc.append(loclabel)
          trainval_mem.append(memlabel)
          trainval_seq.append(sequence)

def save_dataset(save_dir, seq, loc, mem):
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sequences.txt'), 'w') as f1, open(os.path.join(save_dir, 'memlabels.txt'), 'w') as f2,\
          open(os.path.join(save_dir, 'loclabels.txt'), 'w') as f3:
            for sequence, loclabel, memlabel in zip(seq, loc, mem):
                f1.write("{}\n".format(" ".join(sequence)))
                f2.write("{}\n".format(memlabel))
                f3.write("{}\n".format(loclabel))
    print("- done.")

def fasta_iter(path_fasta):
    """
    given a fasta file. yield tuples of header, sequence
    """
    with open(path_fasta) as fa:
        # ditch the boolean (x[0]) and just keep the header or sequence since
        # we know they alternate.
        faiter = (x[1] for x in groupby(fa, lambda line: line[0] == ">"))
        for header in faiter:
            # drop the ">"
            header = header.next()[1:].strip()
            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.next())
            yield header, seq

def encodeLabels(labels):
    locEncoder = preprocessing.LabelEncoder()
    labels = locEncoder.fit_transform(labels)
    locEncoder_name_mapping = dict(zip(locEncoder.classes_, locEncoder.transform(locEncoder.classes_)))
    print(locEncoder_name_mapping)
    return labels, locEncoder


load_dataset('deeploc_data.fasta')
trainval_locmem = list(zip(trainval_loc, trainval_mem))
from sklearn.model_selection import train_test_split
train_seq, val_seq, train_locmem, val_locmem = train_test_split(trainval_seq, trainval_locmem, test_size = 0.1, random_state = 32)
train_loc, train_mem = map(list, zip(*train_locmem))
val_loc, val_mem =  map(list, zip(*val_locmem))
train_loc, locEncoder = encodeLabels(train_loc)
train_mem, memEncoder = encodeLabels(train_mem)
save_dataset("./train", train_seq, train_loc, train_mem)
val_loc, locEncoder = encodeLabels(val_loc)
val_mem, memEncoder = encodeLabels(val_mem)
save_dataset("./val", val_seq, val_loc, val_mem)
test_loc, locEncoder = encodeLabels(test_loc)
test_mem, memEncoder = encodeLabels(test_mem)
save_dataset("./test", test_seq, test_loc, test_mem)
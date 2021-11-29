'''
Train a transformer-based model for emotion classification - dataset augmented with adversarial examples
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from transformers import AdamW
from data_prep import get_test
from data_prep_sentences import get_train
from adv_data_prep import get_adv
import sys
import os
import argparse
from tools import AverageMeter, accuracy_topk, get_default_device
from models import ElectraSequenceClassifier, BertSequenceClassifier, RobertaSequenceClassifier
from transformers import ElectraTokenizer

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=25):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (id, mask, target) in enumerate(train_loader):

        id = id.to(device)
        mask = mask.to(device)
        target = target.to(device)

        # Forward pass
        logits = model(id, mask)
        loss = criterion(logits, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs.update(acc.item(), id.size(0))
        losses.update(loss.item(), id.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {prec.val:.3f} ({prec.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      loss=losses, prec=accs))

def eval(val_loader, model, criterion, device):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (id, mask, target) in enumerate(val_loader):

            id = id.to(device)
            mask = mask.to(device)
            target = target.to(device)

            # Forward pass
            logits = model(id, mask)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, target)
            accs.update(acc.item(), id.size(0))
            losses.update(loss.item(), id.size(0))

    print('Test\t Loss ({loss.avg:.4f})\t'
            'Accuracy ({prec.avg:.3f})\n'.format(
              loss=losses, prec=accs))


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Specify output th file')
    commandLineParser.add_argument('ARCH', type=str, help='electra, bert, roberta')
    commandLineParser.add_argument('DATA_PATH_TRAIN', type=str, help='data filepath')
    commandLineParser.add_argument('DATA_PATH_TEST', type=str, help='data filepath')
    commandLineParser.add_argument('ADV_DIR', type=str, help='Base Directory with adversarial examples')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=2, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.000001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=int, default=10, help="Specify scheduler rate")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    args = commandLineParser.parse_args()

    torch.manual_seed(args.seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/adv_train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()


    # Get all train data
    original_train_tweets, original_train_labels = get_train(args.ARCH, args.DATA_PATH_TRAIN)
    adv_train_tweets, adv_train_labels = get_adv(args.ADV_DIR)
    tweets_train = original_train_tweets + adv_train_tweets
    labels_train = original_train_labels + adv_train_labels
    
    # Create Train dataloader
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    encoded_inputs = tokenizer(tweets_train, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    labels_train = torch.LongTensor(labels_train)
    train_ds = TensorDataset(ids, mask, labels_train)
    train_dl = DataLoader(train_ds, batch_size=args.B, shuffle=True)

    # Create Test dataloader
    input_ids_test, mask_test, labels_test = get_test(args.ARCH, args.DATA_PATH_TEST)
    test_ds = TensorDataset(input_ids_test, mask_test, labels_test)
    test_dl = DataLoader(test_ds, batch_size=args.B)

    # Initialise classifier
    if args.ARCH == 'electra':
        model = ElectraSequenceClassifier()
    elif args.ARCH == 'bert':
        model = BertSequenceClassifier()
    elif args.ARCH == 'roberta':
        model = RobertaSequenceClassifier()
    else:
        raise Exception("Something has gone wrong with architecture definition.")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.sch])

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    for epoch in range(args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_dl, model, criterion, optimizer, epoch, device)
        scheduler.step()

        # evaluate on validation set
        eval(val_dl, model, criterion, device)
    
    # evaluate on test set
    print("Test set\n")
    eval(test_dl, model, criterion, device)

    # Save the trained model
    state = model.state_dict()
    torch.save(state, args.OUT)
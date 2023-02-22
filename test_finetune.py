import os
import argparse
from datetime import datetime

import torch
from torch.nn import functional as F

import timm
import numpy as np
import pandas as pd
from tqdm import tqdm
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from scipy import stats

from src.data.data_loading import load_datasets
from src.model.conv4_epnet import Conv4 as Conv4Vanilla
from src.model.conv4 import Conv4
from src.model.resnet12 import Resnet12
from src.model.wrn import WideResNet
from src.utils import NullLayer, pairwise_distances_logits, accuracy


parser = argparse.ArgumentParser()
# Config
parser.add_argument('--root_data_path', required=True, type=str,
                    help="""Path to the root data folder. Must be the
                            parent folder containing dataset folders.""")
parser.add_argument('--datasets', required=False, type=str, nargs='+',
                    choices=["min", "bccd", "hep",
                             "chestx", "isic", "eurosat", "plant", "ikea"],
                    help="Choose a subset of datasets to test")
parser.add_argument('--test_iters', required=True, type=int,
                    help="""Number of testing iterations per dataset.""")
parser.add_argument('--ft_iters', default=800, type=int,
                    help="""Number iterations to finetune for each for each
                    finetuning epoch.""")
# Few-shot params
parser.add_argument('--shots', required=True, type=int,
                    help="""Number of labelled examples in an episode""")
parser.add_argument('--ways', required=False, default=5, type=int,
                    help="""Number of classes used in an episode.""")
parser.add_argument('--query_size', required=False, default=15, type=int,
                    help="""Number of unlabelled examples in an episode.""")
# Hyperparams
parser.add_argument('--emmet', required=False, default=True, type=bool,
                    help="""Use embedding mixup.""")
parser.add_argument('--img_size', required=True, type=int,
                    help="""Image size used.""")
parser.add_argument('--ft_epochs', required=False, type=int, nargs='+',
                    help="Choose a subset of finetuning epochs to test.")
# Model
parser.add_argument('--model_type', default=False, required=True, choices=[
                        'CONV4', "WRN", "RESNET12", "RESNET18",
                        "RESNET50", "DINO_SMALL", "CONV4_BASE",
                        "CONV4_VANILLA"
                    ], help="""Model type. Either CONV4, 
                    RESNET18, or DINO_SMALL""")
parser.add_argument('--model_path', default=False, required=True, type=str,
                    help="Path to model weights")
parser.add_argument('--conv4_prot_size', default=64, type=int,
                    help="Size of the CONV4 prototype")
parser.add_argument('--device', default='cuda', type=str,
                    help="Device to use for testing (ex: cuda:0).")
parser.add_argument('--num_classes_init', default=64, type=int,
                    help="""Number of classes to initialize the embedding
                    network with, if required.""")
args = parser.parse_args()


def get_model(model_type, num_classes, model_path, device, conv4_prot_size=64):
    if model_type == "CONV4":
        model = Conv4(num_classes, use_fc=False, prototype_size=conv4_prot_size)
        raise NotImplemented
    elif model_type == "CONV4_VANILLA":
        model = Conv4Vanilla(avgpool=True)
        model.add_classifier(64)
    elif model_type == "RESNET12":
        model = Resnet12(1, 0.1, num_classes=num_classes, use_fc=False)
        raise NotImplemented
    elif model_type == "RESNET18":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.fc = NullLayer()
        raise NotImplemented
    elif model_type == "RESNET50":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        model.fc = NullLayer()
        raise NotImplemented
    elif model_type == "WRN":
        model = WideResNet(28, 10, use_fc=False, num_classes=num_classes)
        model.fc = NullLayer()
        raise NotImplemented
    elif model_type == "DINO_SMALL":
        model = timm.create_model('vit_small_patch8_224_dino', pretrained=True)

    print("Using model", model_type)

    # model.load_state_dict(torch.load(model_path))

    # Freeze model except for last layer
    for param in model.parameters():
        param.requires_grad = False
    
    if model_type == "CONV4_VANILLA":
        model.classifier.requires_grad_(True) # CONV4
        model.conv3.requires_grad_(True) # CONV4
    if model_type == "DINO_SMALL":
        model.blocks[-1].mlp.fc2.requires_grad_(True) # DINO-S

    model.to(device)

    return model


if __name__ == "__main__":
    ROOT_DATA_PATH = args.root_data_path
    DATASET_SUBSET = args.datasets
    TEST_ITERS = args.test_iters
    FT_ITERS = args.ft_iters
    SHOTS = args.shots
    WAYS = args.ways
    QUERY = args.query_size
    IMG_SIZE = args.img_size
    MODEL_TYPE = args.model_type
    MODEL_PATH = args.model_path
    CONV4_PROT_SIZE = args.conv4_prot_size
    DEVICE = args.device
    NUM_CLASSES = args.num_classes_init
    EMMET = args.emmet

    datasets = load_datasets(ROOT_DATA_PATH, IMG_SIZE, SHOTS, WAYS, QUERY,
                             dataset_subset=DATASET_SUBSET)

    # Finetuning and Testing
    if args.ft_epochs is None:
        FT_EPOCHS = [0,1,2,5,10,20]
    else:
        FT_EPOCHS = args.ft_epochs
    WARMUP_EPOCHS = 5
    LR = 1e-3

    # Support/query indices for extraction when testing
    support_indices = np.zeros(WAYS * (SHOTS + QUERY), dtype=bool)
    selection = np.arange(WAYS) * (SHOTS + QUERY)
    for offset in range(SHOTS):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)

    results = [["Dataset", "Epoch", "Acc.", "Uncertainty"]]
    for loader, dataset_name in datasets:
        print("==================================")
        print("==== Testing", dataset_name, "====")

        comp_data = []
        comp_labels = []
        print("Compiling dataset episodes")
        for i in tqdm(range(TEST_ITERS)):
            data, labels = loader.get_episode()
            comp_data.append(data)
            comp_labels.append(labels)

        dataset_results = []

        # Extract a single batch to finetune on
        ft_data, ft_labels = loader.get_episode()
        ft_data, ft_labels = ft_data.to(DEVICE).squeeze(0), ft_labels.to(DEVICE).squeeze(0)

        for EPOCHS in FT_EPOCHS:
            print(f"--- Finetuning {MODEL_TYPE} for {EPOCHS} on {dataset_name}")
            print("Loading model...")
            model = get_model(MODEL_TYPE, NUM_CLASSES, MODEL_PATH,
                              DEVICE, conv4_prot_size=CONV4_PROT_SIZE)
            
            # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.05)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=WARMUP_EPOCHS, max_epochs=EPOCHS)
            scaler = torch.cuda.amp.GradScaler()
            
            if EPOCHS > 0:
                print("Finetuning...")
                data, labels = ft_data, ft_labels
                sort = torch.sort(labels)
                data = data.squeeze(0)[sort.indices].squeeze(0)
                labels = labels.squeeze(0)[sort.indices].squeeze(0)
                
                for epoch in range(EPOCHS):
                    losses = []
                    accuracies = []
                    for i in tqdm(range(FT_ITERS)):
                        # Regularize data through shuffling class sections
                        d, c, h, w = data.shape

                        # Shuffle shots
                        shuffled_data = data.reshape(WAYS, (SHOTS + QUERY), c, h, w)
                        shuffled_data = shuffled_data[:, :SHOTS]
                        shuffled_prots = torch.randperm(SHOTS)

                        shuffled_data = shuffled_data[:, shuffled_prots]
                        shuffled_ways = torch.randperm(WAYS)

                        if data.shape[1] < (SHOTS + QUERY):
                            # Repeat shot dimension to fill
                            repeat_axis = (SHOTS + QUERY) // SHOTS
                            shuffled_data = shuffled_data.repeat(1, repeat_axis, 1, 1, 1)

                        # Shuffle ways
                        shuffled_data = shuffled_data[shuffled_ways].reshape(d,c,h,w)

                        # Randaug
                        # for i in range(0, shuffled_data.shape[0]):
                        #     img = T.ToPILImage()(shuffled_data[i])
                        #     shuffled_data[i] = T.ToTensor()(tfm(img))
                        # shuffled_data = shuffled_data * torch.rand_like(shuffled_data)

                        with torch.cuda.amp.autocast():
                            prototypes = model(shuffled_data)

                            if EMMET:
                                a = prototypes.reshape(WAYS, (SHOTS + QUERY), -1)
                                b = a[torch.randperm(WAYS), :, :]
                                b = b[:, torch.randperm(SHOTS + QUERY), :]
                                prototypes = ((a + b)/2).reshape(WAYS * (SHOTS + QUERY), -1)

                            support_indices = np.zeros(data.size(0), dtype=bool)
                            selection = np.arange(WAYS) * (SHOTS + QUERY)
                            for offset in range(SHOTS):
                                support_indices[selection + offset] = True
                            query_indices = torch.from_numpy(~support_indices)
                            support_indices = torch.from_numpy(support_indices)

                            support = prototypes.squeeze(0)[support_indices]
                            support = support.reshape(WAYS, SHOTS, -1)
                            support = support.mean(dim=1).squeeze(0)

                            query = prototypes.squeeze(0)
                            query = query[query_indices]

                            logits = pairwise_distances_logits(query, support)
                            query_labels = labels[query_indices].long()
                            acc = accuracy(logits.squeeze(0), query_labels.squeeze(0))
                            accuracies.append(acc.item())

                            # Calculate loss
                            loss = F.cross_entropy(logits.squeeze(0), query_labels.squeeze(0))
                            losses.append(loss.item())

                            scaler.scale(loss).backward()

                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                    print(f'Epoch {epoch}: Loss {np.mean(losses)} Accuracy {np.mean(accuracies)}')
                    scheduler.step()
            
            print("Testing...")
            prot_accuracies = []
            losses = []
            for i in tqdm(range(TEST_ITERS)):
                data = comp_data[i]
                labels = comp_labels[i]
                # data, labels = loader.get_episode()
                data, labels = data.to(DEVICE).squeeze(0), labels.to(DEVICE).squeeze(0)

                sort = torch.sort(labels)
                data = data.squeeze(0)[sort.indices].squeeze(0)
                labels = labels.squeeze(0)[sort.indices].squeeze(0)

                support_labels = labels[support_indices].long()
                labels = labels[query_indices].long()

                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        prototypes = model(data)

                # Use average support from prototypes
                support = prototypes.squeeze(0)[support_indices]
                support = support.reshape(WAYS, SHOTS, -1)
                support = support.mean(dim=1).squeeze(0)

                query = prototypes.squeeze(0)
                query = query[query_indices]

                # Calculate accuracy
                logits = pairwise_distances_logits(query, support)
                acc = accuracy(logits.squeeze(0), labels.squeeze(0))
                prot_accuracies.append(acc.item())

            print(f'Accuracy {np.mean(prot_accuracies)*100}%')

            confidence_interval = stats.norm.interval(0.95, loc=np.mean(prot_accuracies), scale=stats.sem(prot_accuracies))
            confidence_interval = ((confidence_interval[1] - confidence_interval[0])/2)*100
            print(f'95% confidence interval {confidence_interval}%')

            results.append([dataset_name, EPOCHS, round(np.mean(prot_accuracies)*100, 2), 
                            round(confidence_interval, 2)])
    
    results = pd.DataFrame(results)
    csv_filename = f"{MODEL_TYPE}_Results_{SHOTS}shots_emmet{EMMET}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv"
    results.to_csv(csv_filename, header=False)

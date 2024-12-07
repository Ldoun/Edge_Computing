import os
import sys
import logging
from functools import partial
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from data import Cifar100, Subset
from trainer import Trainer
from config import get_args
from lr_scheduler import get_sch
from utils import seed_everything, handle_unhandled_exception, save_to_json
from pruning_utils import pruning_model, pruning_model_structured, prune_model_custom, check_sparsity, extract_mask, remove_prune

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed) #fix seed
    device = torch.device('cuda:0') #use cuda:0

    result_path = os.path.join(args.result_path, args.prune_type + '_' +  str(args.pruning_ratio) + '_' + str(len(os.listdir(args.result_path))))
    os.makedirs(result_path)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    
    logger.info(args)
    save_to_json(vars(args), os.path.join(result_path, 'config.json'))
    sys.excepthook = partial(handle_unhandled_exception,logger=logger)

    # skf = StratifiedKFold(n_splits=args.cv_k, random_state=args.seed, shuffle=True).split(train_data['path'], train_data['label']) #Using StratifiedKFold for cross-validation    
    # for fold, (train_index, valid_index) in enumerate(skf): #by skf every fold will have similar label distribution

    dataset = Cifar100(args.path, train=True, augment=True, download=True)
    dataset_no_augment = Cifar100(args.path, train=True, augment=False, download=True)
    train_index, valid_index = train_test_split(range(len(dataset)), test_size=0.3, random_state=1)

    train_dataset = Subset(dataset, train_index)
    valid_dataset = Subset(dataset_no_augment, valid_index)
    test_dataset = Cifar100(args.path, train=False, augment=False, download=False)

    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, 
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    model = resnet18(num_classes=100).to(device) #make model based on the model name and args
    torch.save(model.state_dict(), os.path.join(result_path, 'init.pt'))

    logger.info('Dense model Training')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_sch(args.scheduler, optimizer, epochs=args.epochs)

    trainer = Trainer(
        train_loader, valid_loader, model, loss_fn, optimizer, scheduler, device, args.patience, args.epochs, result_path, logger)
    trainer.train()
    trainer.test(test_loader) 

    logger.info('Sparse model Training')
    if args.pruning_ratio != 0.0:
        check_sparsity(model, logger)
        if args.prune_type == 'structured':
            pruning_model_structured(model, args.pruning_ratio, logger)
        else:
            pruning_model(model, args.pruning_ratio, logger)

        check_sparsity(model, logger)
        current_mask = extract_mask(model.state_dict())
        remove_prune(model)

        initialization = torch.load(os.path.join(result_path, 'init.pt'))
        model.load_state_dict(initialization)
        prune_model_custom(model, current_mask, logger)
        check_sparsity(model, logger)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_sch(args.scheduler, optimizer, epochs=args.epochs)

    trainer = Trainer(
        train_loader, valid_loader, model, loss_fn, optimizer, scheduler, device, args.patience, args.epochs, result_path, logger)
    trainer.train()
    trainer.test(test_loader) 
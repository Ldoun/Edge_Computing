import os
import sys
import logging
from functools import partial
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch import optim, nn
import torch.ao.quantization
from torch.utils.data import DataLoader
# from torchvision.models import resnet18
from torchvision.models.quantization import resnet18

from data import Cifar10, Subset
from trainer import Trainer
from config import get_args
from lr_scheduler import get_sch
from utils import seed_everything, handle_unhandled_exception, save_to_json, print_size_of_model

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed) #fix seed
    device = torch.device('cuda:0') #use cuda:0

    if args.train_dense:
        result_path = os.path.join(args.result_path, 'Dense'+ '_' + str(len(os.listdir(args.result_path))))
    else:   
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

    dataset = Cifar10(args.path, train=True, augment=True, download=True)
    dataset_no_augment = Cifar10(args.path, train=True, augment=False, download=True)
    train_index, valid_index = train_test_split(range(len(dataset)), test_size=0.3, random_state=1)

    train_dataset = Subset(dataset, train_index)
    valid_dataset = Subset(dataset_no_augment, valid_index)
    test_dataset = Cifar10(args.path, train=False, augment=False, download=False)

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
    
    model = resnet18(num_classes=10).to(device)

    if args.train_dense:
        logger.info('Dense model Training')
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = get_sch(args.scheduler, optimizer, epochs=args.epochs)

        trainer = Trainer(
            train_loader, valid_loader, model, False, loss_fn, optimizer, scheduler, device, args.patience, args.epochs, result_path, logger)
        trainer.train()
        trainer.test(test_loader) 

    else:
        model = model.load_state_dict(torch.load(os.path.join(args.dense_model, 'best_model.pt')))
        model.eval()
        model.fuse_model()

        
        if args.is_qat:
            logger.info('Quantization aware training')

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = get_sch(args.scheduler, optimizer, epochs=args.epochs)

            # model.qconfig = torch.ao.quantization.default_qconfig
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86') # per-channel quantization
            torch.ao.quantization.prepare_qat(model, inplace=True)

            trainer = Trainer(
                train_loader, valid_loader, model, True, loss_fn, optimizer, scheduler, device, args.patience, 10, result_path, logger)
            trainer.train()

            model = torch.ao.quantization.convert(model.eval(), inplace=False)
            model.eval()
            trainer.test(test_loader)

        else:
            logger.info('Post-training quantization')
            # model.qconfig = torch.ao.quantization.default_qconfig
            model.qconfig = torch.ao.quantization.get_default_qconfig('x86') # per-channel quantization

            logger.info(model.qconfig)
            torch.ao.quantization.prepare(model, inplace=True)
            
            trainer = Trainer(
                train_loader, valid_loader, model, False, loss_fn, None, None, device, args.patience, args.epochs, result_path, logger)
            trainer.test(train_loader) # calibrate with the training set

            mocdl = torch.ao.quantization.convert(model, inplace=False)

            trainer = Trainer(
                train_loader, valid_loader, model, False, loss_fn, None, None, device, args.patience, args.epochs, result_path, logger)
            trainer.test(test_loader)
        
        print_size_of_model(model, logger)


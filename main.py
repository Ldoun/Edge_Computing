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
from trainer import Trainer, test
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
        result_path = os.path.join(args.result_path, 'is_qat=' + str(args.is_qat)  + '_' + str(len(os.listdir(args.result_path))))
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
        model.load_state_dict(torch.load(os.path.join(args.dense_model, 'best_model.pt')))
        model.eval()
        model.cuda()
        logger.info('Before Quantization')
        test(model, test_loader, 'cuda', logger)

        if args.is_qat:
            from torch.ao.quantization import QConfig, default_per_channel_weight_fake_quant, default_weight_fake_quant, FakeQuantize, MovingAverageMinMaxObserver

            model.train()
            model.fuse_model(is_qat=True)
            logger.info('Quantization aware training')

            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            scheduler = get_sch(args.scheduler, optimizer, epochs=10)

            # model.qconfig = torch.ao.quantization.default_qconfig
            qconfig = QConfig(
                activation=FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=True,
                ),
                weight=default_per_channel_weight_fake_quant if args.ch_quantize else default_weight_fake_quant,
            )
            model.qconfig = qconfig
            torch.ao.quantization.prepare_qat(model, inplace=True)

            trainer = Trainer(
                train_loader, valid_loader, model, True, loss_fn, optimizer, scheduler, device, args.patience, 10, result_path, logger)
            trainer.train()

            model.cpu()
            model = torch.ao.quantization.convert(model.eval(), inplace=False)
            test(model, test_loader, 'cpu', logger)

        else:
            from torch.ao.quantization import QConfig, default_per_channel_weight_observer, default_weight_observer, HistogramObserver

            model.cpu()
            model.fuse_model()
            logger.info('Post-training quantization')
            # model.qconfig = torch.ao.quantization.default_qconfig
            
            qconfig = qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=True),
                weight=default_per_channel_weight_observer if args.ch_quantize else default_weight_observer,
                # weight=default_weight_observer,
            )
            model.qconfig = qconfig

            logger.info(model.qconfig)
            torch.ao.quantization.prepare(model, inplace=True)
            test(model, train_loader, 'cpu', logger, early_stop=32) # calibrate with the training set

            model = torch.ao.quantization.convert(model, inplace=False)
            test(model, test_loader, 'cpu', logger)
        
        print_size_of_model(model, logger)


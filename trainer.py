import os
import sys
import torch
import numpy as np
import torch.ao.quantization
import torch.nn.intrinsic

class Trainer():
    def __init__(self, train_loader, valid_loader, model, is_qat, loss_fn, optimizer, scheduler, device, patience, epochs, result_path, fold_logger):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.is_qat = is_qat
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.epochs = epochs
        self.logger = fold_logger
        self.best_model_path = os.path.join(result_path, 'best_model.pt')
    
    def train(self):
        best = 0
        best_epoch = -1
        for epoch in range(1,self.epochs+1):
            self.logger.info(f'lr: {self.scheduler.get_last_lr()}')
            loss_train, score_train = self.train_step()
            loss_val, score_val = self.valid_step()
            self.scheduler.step()

            if self.is_qat:
                if epoch > 5:
                    self.model.apply(torch.ao.quantization.disable_observer)
                if epoch > 3:
                    self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

            self.logger.info(f'Epoch {str(epoch).zfill(5)}: t_loss:{loss_train:.3f} t_score:{score_train:.3f} v_loss:{loss_val:.3f} v_score:{score_val:.3f}')

            if score_val > best:
                best = score_val
                torch.save(self.model.state_dict(), self.best_model_path)
                bad_counter = 0
                best_epoch = epoch

            else:
                bad_counter += 1

            if bad_counter == self.patience:
                break
        print(f'saved {best_epoch}-model')

    def train_step(self):
        self.model.train()

        total_loss = 0
        correct = 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)            
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.shape[0]
            correct += sum(output.argmax(dim=1) == y).item() # classification task
        
        return total_loss/len(self.train_loader.dataset), correct/len(self.train_loader.dataset)
    
    def valid_step(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            for x, y in self.valid_loader:
                x, y = x.to(self.device), y.to(self.device)

                output = self.model(x)
                loss = self.loss_fn(output, y)

                total_loss += loss.item() * x.shape[0]
                correct += sum(output.argmax(dim=1) == y).item() # classification task
                
        return total_loss/len(self.valid_loader.dataset), correct/len(self.valid_loader.dataset)
    
    def test(self, test_loader):
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.loss_fn(output, y)

                total_loss += loss.item() * x.shape[0]
                correct += sum(output.argmax(dim=1) == y).item() # classification task

        self.logger.info(f'Loss: {total_loss/len(test_loader.dataset)}')
        self.logger.info(f'Acc: {correct/len(test_loader.dataset)}')


def test(model, test_loader, device, logger, early_stop=-1):
    model.eval()
    with torch.no_grad():
        correct = 0
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            correct += sum(output.argmax(dim=1) == y).item() # classification task

            if early_stop == i:
                logger.info('calibration data. Early stopped')
                return

    logger.info(f'Acc: {correct/len(test_loader.dataset)}')
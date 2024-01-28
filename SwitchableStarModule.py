import matplotlib
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import Accuracy

from modules.SwitchableStarModel import SwitchablestarModel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in',
                                 nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in',
                                 nonlinearity='leaky_relu')


class SwitchablestarModule(LightningModule):
    def __init__(self, hparams, cbits, do_log):
        super().__init__()

        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.do_log = do_log

        self.challenges = []
        self.preds = []

        self.cbits = cbits
        self.train_accs = []
        self.val_accs = []
        self.test_accs = 0

        self.train_counts_real = torch.zeros(4, device='cuda')
        self.val_counts_real = torch.zeros(4, device='cuda')
        self.test_counts_real = torch.zeros(4, device='cuda')

        self.train_counts_pred = torch.zeros(4, device='cuda')
        self.val_counts_pred = torch.zeros(4, device='cuda')
        self.test_counts_pred = torch.zeros(4, device='cuda')

        self.model = SwitchablestarModel(
            hparams["ns"], hparams['n_layer'], cbits)
        self.loss = nn.CrossEntropyLoss()
        self.model.apply(weights_init)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        challenge, real_response = batch
        real_response = real_response.squeeze()
        gen_response = self.model(challenge).squeeze()
        loss = self.loss(gen_response, real_response)

        gen_response = gen_response.softmax(dim=1)
        preds = gen_response.argmax(dim=1)
        self.train_acc(preds, real_response)

        if self.do_log and self.current_epoch == 0:
            self._add_class_count(real_response, self.train_counts_real)

        if self.do_log and self.current_epoch % 5 == 0:
            self._add_class_count(preds, self.train_counts_pred)
            self.logger.experiment.add_scalars(
                "Pred Distr.", {"Train": preds.float().mean()},
                self.current_epoch)

        return loss

    def training_epoch_end(self, outputs):
        loss = torch.stack(
            [output["loss"] for output in outputs]).mean()
        self.logger.experiment.add_scalars("loss", {"train_loss": loss},
                                           self.current_epoch)

        self.train_accs.append(self.train_acc.compute().item())
        self.log('Train Accuracy', self.train_acc)

        if self.do_log and self.current_epoch % 5 == 0:
            self.logger.experiment.add_figure(
                'Train Class Preds',
                self._get_class_barplot(self.train_counts_pred),
                self.current_epoch
            )
            plt.close()
            self.logger.experiment.add_figure(
                'Train Class Real',
                self._get_class_barplot(self.train_counts_real),
                self.current_epoch
            )
            plt.close()

            self.train_counts_pred = torch.zeros(4, device='cuda')

    def validation_step(self, batch, batch_idx):
        challenge, real_response = batch
        real_response = real_response.squeeze()
        gen_response = self.model(challenge)
        loss = self.loss(gen_response, real_response)

        gen_response = gen_response.softmax(dim=1)
        preds = gen_response.argmax(dim=1)
        self.val_acc(preds, real_response)

        if self.do_log and self.current_epoch == 0:
            self._add_class_count(real_response, self.val_counts_real)

        if self.do_log and self.current_epoch % 5 == 0:
            self._add_class_count(preds, self.val_counts_pred)
            self.logger.experiment.add_scalars(
                "Pred Distr.", {"Val": preds.float().mean()},
                self.current_epoch)

        return loss

    def validation_epoch_end(self, outputs):
        loss = torch.stack(outputs).mean()
        self.logger.experiment.add_scalars("loss", {"val_loss": loss},
                                           self.current_epoch)
        val_acc = self.val_acc.compute().item()
        self.val_accs.append(val_acc)
        self.log('Val Accuracy', self.val_acc, on_epoch=True)

        if self.do_log and self.current_epoch % 5 == 0:
            self.logger.experiment.add_figure(
                'Val Class Preds',
                self._get_class_barplot(self.val_counts_pred),
                self.current_epoch
            )
            plt.close()
            self.logger.experiment.add_figure(
                'Val Class Real',
                self._get_class_barplot(self.val_counts_real),
                self.current_epoch
            )
            plt.close()
            self.val_counts_pred = torch.zeros(4, device='cuda')

    def test_step(self, batch, batch_idx):
        challenge, real_response = batch
        gen_response = self.model(challenge)
        real_response = real_response.squeeze()
        loss = self.loss(gen_response, real_response)

        gen_response = gen_response.softmax(dim=1)
        preds = gen_response.argmax(dim=1)
        self.test_acc(preds, real_response)

        if self.do_log:
            self._add_class_count(real_response, self.test_counts_real)
            self._add_class_count(preds, self.test_counts_pred)
            self.logger.experiment.add_scalars(
                "Pred Distr.", {"Test": preds.float().mean()},
                self.current_epoch)

        return loss

    def test_epoch_end(self, outputs):
        loss = torch.stack(outputs).mean()
        self.logger.experiment.add_scalars("loss", {"test_loss": loss},
                                           self.current_epoch)
        self.test_accs = self.test_acc.compute().item()
        self.log('Test Accuracy', self.test_acc, on_epoch=True)

        if self.do_log:
            self.logger.experiment.add_figure(
                'Test Class Preds',
                self._get_class_barplot(self.test_counts_pred),
                self.current_epoch
            )
            plt.close()
            self.logger.experiment.add_figure(
                'Test Class Real',
                self._get_class_barplot(self.test_counts_real),
                self.current_epoch
            )
            plt.close()

    '''def backward(self, trainer, loss, optimizer_idx):
        super().backward(trainer, loss, optimizer_idx)
        if self.do_log and self.current_epoch % 5 == 0:
            board.plot_grad_flow(
                self.model.named_parameters(), self.gen_grad
            )'''

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.hparams.lr,
            (0.9, 0.999)
        )
        return optimizer

    def _add_class_count(self, data, counts):
        vals, cnts = torch.unique(data, return_counts=True)
        for idx, cnts in zip(vals, cnts):
            counts[idx] += cnts

    def _get_class_barplot(self, counts):
        counts = counts.tolist()
        df = pd.DataFrame({
            '00': counts[0],
            '01': counts[1],
            '10': counts[2],
            '11': counts[3]
        }, index=[0])
        g = sns.barplot(df)
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        return g.figure

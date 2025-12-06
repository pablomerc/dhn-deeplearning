"""Train the model.
"""

import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import wandb

from input_pipeline import create_dataloader
from models_hamiltonian import get_model_hamiltoinian


class Trainer(object):

  def __init__(self, config, dtype=torch.float32):
    self.config = config
    self.dtype = dtype
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # general
    self.workdir = self.config.workdir
    # optim
    self.num_epochs = self.config.optim.num_epochs
    self.lr = self.config.optim.lr
    # logging
    self.per_save_epochs = self.config.logging.per_save_epochs
    self.per_save_tmp_epochs = self.config.logging.per_save_tmp_epochs
    self.per_eval_epochs = self.config.logging.per_eval_epochs
    self.num_eval_batches = self.config.logging.num_eval_batches
    self.num_vis = self.config.logging.num_vis

    # dataloaders
    self.train_loader = create_dataloader(config.data, split='train')
    self.test_loader = create_dataloader(config.data, split='test')

    # networks
    self.hamiltonian_net = get_model_hamiltoinian(config.model, dtype=dtype)
    self.hamiltonian_net = self.hamiltonian_net.to(self.dtype).to(self.device)

    # optimizer
    self.optimizer = optim.Adam(self.hamiltonian_net.parameters(), lr=self.lr, weight_decay=1e-4)
    self.scheduler = CosineAnnealingLR(self.optimizer, self.num_epochs, eta_min=self.lr)

    # Initialize wandb
    # Extract experiment name from workdir (e.g., results/ar/two_body_baseline_hnn_tf -> two_body_baseline_hnn_tf)
    exp_name = os.path.basename(self.workdir) if os.path.basename(self.workdir) else 'experiment'
    wandb.init(
      project='dhn-deeplearning',
      name=exp_name,
      config={
        'num_epochs': self.num_epochs,
        'lr': self.lr,
        'batch_size': config.data.batch_size,
        'workdir': self.workdir,
        'model': config.model.hamiltonian,
        'q_dim': config.model.q_dim,
        'embedding_dim': config.model.embedding_dim,
      },
      dir=self.workdir,
    )

  def preprocess_data(self, data):
    for k in data:
      if isinstance(data[k], torch.Tensor) and k != 'idx':
        data[k] = data[k].to(self.dtype).to(self.device)
    data['idx'] = data['idx'].to(torch.long).to(self.device)
    return data

  def train_step(self):
    dict_losses_all = []
    for data in self.train_loader:
      self.optimizer.zero_grad()
      data = self.preprocess_data(data)
      loss_train, dict_losses = self.hamiltonian_net.get_losses(data, self.config.loss)
      loss_train.backward()
      self.optimizer.step()
      dict_losses_all.append(dict_losses)
    self.scheduler.step()
    # Average losses across all batches in the epoch
    dict_losses_mean = {}
    for k in dict_losses_all[0]:
      # Detach tensors and convert to Python floats before averaging
      values = [dict_losses[k].detach().cpu().item() if isinstance(dict_losses[k], torch.Tensor) else dict_losses[k]
                for dict_losses in dict_losses_all]
      dict_losses_mean[k] = np.mean(values)
    return dict_losses_mean

  def eval_step(self):
    self.hamiltonian_net.eval()
    with torch.no_grad():
      # Evaluate on training set
      dict_losses_train_all = []
      for i, data in enumerate(self.train_loader):
        if i >= self.num_eval_batches:
          break
        data = self.preprocess_data(data)
        dict_losses, dict_vals = self.hamiltonian_net.inference(data)
        dict_losses_train_all.append(dict_losses)
        if i == 0:
          dict_vis_train = self.hamiltonian_net.get_vis_dict(dict_vals, num_vis=self.num_vis)
      dict_losses_train_mean = {}
      for k in dict_losses_train_all[0]:
        dict_losses_train_mean[k] = np.mean([dict_losses[k] for dict_losses in dict_losses_train_all])

      # Evaluate on test set
      dict_losses_test_all = []
      for i, data in enumerate(self.test_loader):
        if i >= self.num_eval_batches:
          break
        data = self.preprocess_data(data)
        dict_losses, dict_vals = self.hamiltonian_net.inference(data)
        dict_losses_test_all.append(dict_losses)
        if i == 0:
          dict_vis_test = self.hamiltonian_net.get_vis_dict(dict_vals, num_vis=self.num_vis)
      dict_losses_test_mean = {}
      for k in dict_losses_test_all[0]:
        dict_losses_test_mean[k] = np.mean([dict_losses[k] for dict_losses in dict_losses_test_all])

      return dict_losses_train_mean, dict_losses_test_mean, dict_vis_train, dict_vis_test

  def train_and_eval(self):
    writer = SummaryWriter(self.workdir)

    for epoch in range(self.num_epochs + 1):
      self.hamiltonian_net.train()
      dict_losses = self.train_step()

      # Log to tensorboard
      for k in dict_losses:
        writer.add_scalar(k, dict_losses[k], epoch)

      # Log to wandb (prefix train/ for training metrics)
      wandb_log = {f'train/{k}': dict_losses[k] for k in dict_losses}
      wandb_log['epoch'] = epoch
      wandb_log['lr'] = self.scheduler.get_last_lr()[0]
      wandb.log(wandb_log, step=epoch)

      # Print training losses to console
      loss_str = ', '.join([f'{k}: {dict_losses[k]:.6f}' for k in dict_losses])
      print(f'Epoch {epoch}: {loss_str}, lr: {self.scheduler.get_last_lr()[0]:.6f}')

      if epoch % self.per_eval_epochs == 0:
        dict_losses_eval_train, dict_losses_eval_test, dict_vis_train, dict_vis_test = self.eval_step()

        # Log eval metrics on training set to tensorboard
        for k in dict_losses_eval_train:
          writer.add_scalar(f'eval_train/{k}', dict_losses_eval_train[k], epoch)

        # Log eval metrics on test set to tensorboard
        for k in dict_losses_eval_test:
          writer.add_scalar(f'eval_test/{k}', dict_losses_eval_test[k], epoch)

        # Log eval metrics to wandb (prefix eval_train/ and eval_test/)
        wandb_log_eval = {}
        for k in dict_losses_eval_train:
          wandb_log_eval[f'eval_train/{k}'] = dict_losses_eval_train[k]
        for k in dict_losses_eval_test:
          wandb_log_eval[f'eval_test/{k}'] = dict_losses_eval_test[k]
        wandb_log_eval['epoch'] = epoch
        wandb.log(wandb_log_eval, step=epoch)

        # Log images to wandb (from test set)
        for k in dict_vis_test:
          image_tensor = dict_vis_test[k]
          for i in range(min(self.num_vis, image_tensor.shape[0])):
            # Tensorboard
            writer.add_image(k + f'/sample_{i}', image_tensor[i], epoch)
            # Wandb
            wandb.log({f'images/{k}/sample_{i}': wandb.Image(image_tensor[i])}, step=epoch)

      if epoch % self.per_save_epochs == 0:
        self.save_checkpoint(epoch, is_tmp=False)

      if epoch % self.per_save_tmp_epochs == 0:
        self.save_checkpoint(epoch, is_tmp=True)

    # Finish wandb run
    wandb.finish()

  def save_checkpoint(self, epoch, is_tmp=False):
    checkpoint = {
      'epoch': epoch,
      'model_state_dict': self.hamiltonian_net.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
    }
    ckpt_name = 'checkpoint.pth' if is_tmp else f'checkpoint_ep{epoch}.pth'
    ckpt_path = os.path.join(self.workdir, ckpt_name)
    torch.save(checkpoint, ckpt_path)

  def load_checkpoint(self, ckpt_name='checkpoint.pth'):
    ckpt_path = os.path.join(self.workdir, ckpt_name)
    checkpoint = torch.load(ckpt_path)
    self.hamiltonian_net.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

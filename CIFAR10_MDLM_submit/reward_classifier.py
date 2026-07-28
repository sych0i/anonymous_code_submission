"""Train a plain (noise-free) CIFAR-10 classifier used as the VISTA-SMC
reward model r(x0) = log p(c|x0).

Unlike `classifier.py` (which conditions on noisy xt/sigma for CBG/FUDGE
guidance), this classifier only ever sees fully-decoded clean images: VISTA-SMC
estimates V_t(xt) via full multi-step rollout to x0, so the reward model never
needs to look at masked/noisy intermediate states.

Preprocessing matches how generated samples are decoded at sampling time
(see `smc.py:_save_samples`): raw uint8 pixels in [0, 255], reshaped to
(3, 32, 32) in (C, H, W) order, scaled to [0, 1] with no channel mean/std
normalization.
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


CIFAR10_CLASSES = [
  'airplane', 'automobile', 'bird', 'cat', 'deer',
  'dog', 'frog', 'horse', 'ship', 'truck',
]


def load_cifar10_arrays(data_dir):
  """Loads raw CIFAR-10 batches into (images, labels) uint8/int64 arrays.

  Images are returned as (N, 3, 32, 32) uint8 in (C, H, W) order, matching
  the layout produced by the diffusion model's raw-pixel tokenizer.
  """
  def _load_batch(path):
    with open(path, 'rb') as f:
      batch = pickle.load(f, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.uint8)
    labels = np.array(batch[b'labels'], dtype=np.int64)
    return images, labels

  batches_dir = os.path.join(data_dir, 'cifar-10-batches-py')
  train_images, train_labels = [], []
  for i in range(1, 6):
    images, labels = _load_batch(
      os.path.join(batches_dir, f'data_batch_{i}'))
    train_images.append(images)
    train_labels.append(labels)
  train_images = np.concatenate(train_images, axis=0)
  train_labels = np.concatenate(train_labels, axis=0)

  test_images, test_labels = _load_batch(
    os.path.join(batches_dir, 'test_batch'))
  return (train_images, train_labels), (test_images, test_labels)


class CIFAR10RewardDataset(torch.utils.data.Dataset):
  """images: (N, 3, 32, 32) uint8. Applies pad-crop + hflip iff train=True."""

  def __init__(self, images, labels, train):
    self.images = images
    self.labels = labels
    self.train = train

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    img = self.images[index]
    if self.train:
      img = np.pad(
        img, ((0, 0), (4, 4), (4, 4)), mode='reflect')
      top = np.random.randint(0, 9)
      left = np.random.randint(0, 9)
      img = img[:, top:top + 32, left:left + 32]
      if np.random.rand() < 0.5:
        img = img[:, :, ::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    label = int(self.labels[index])
    return img, label


class SmallCNNClassifier(nn.Module):
  """Lightweight CNN: reward evals happen O(N * T * J) times per generation,
  so this favors speed over capacity."""

  def __init__(self, num_classes=10):
    super().__init__()
    self.block1 = nn.Sequential(
      nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
      nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
      nn.MaxPool2d(2))  # 32 -> 16
    self.block2 = nn.Sequential(
      nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
      nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
      nn.MaxPool2d(2))  # 16 -> 8
    self.block3 = nn.Sequential(
      nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
      nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
      nn.AdaptiveAvgPool2d(1))  # 8 -> 1
    self.fc = nn.Linear(128, num_classes)

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = x.flatten(1)
    return self.fc(x)

  def log_prob(self, x):
    """log p(c|x0) for every class c, i.e. r(x0) per candidate target class."""
    return F.log_softmax(self.forward(x), dim=-1)


class CIFARResNet18(nn.Module):
  """Higher-capacity CIFAR-10 evaluator/reward model with input normalization."""

  def __init__(self):
    super().__init__()
    self.register_buffer(
      'mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
    self.register_buffer(
      'std', torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))
    self.backbone = torchvision.models.resnet18(
      weights=None, num_classes=len(CIFAR10_CLASSES))
    self.backbone.conv1 = nn.Conv2d(
      3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.backbone.maxpool = nn.Identity()

  def forward(self, images):
    return self.backbone((images - self.mean) / self.std)

  def log_prob(self, images):
    return F.log_softmax(self.forward(images), dim=-1)


@torch.no_grad()
def _evaluate(model, loader, device):
  model.eval()
  correct, total, loss_sum = 0, 0, 0.0
  for images, labels in loader:
    images, labels = images.to(device), labels.to(device)
    logits = model(images)
    loss_sum += F.cross_entropy(
      logits, labels, reduction='sum').item()
    correct += (logits.argmax(dim=-1) == labels).sum().item()
    total += labels.numel()
  return correct / total, loss_sum / total


def train(args):
  device = torch.device(
    args.device if torch.cuda.is_available() else 'cpu')
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  (train_images, train_labels), (val_images, val_labels) = (
    load_cifar10_arrays(args.data_dir))
  train_loader = torch.utils.data.DataLoader(
    CIFAR10RewardDataset(train_images, train_labels, train=True),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, drop_last=True)
  val_loader = torch.utils.data.DataLoader(
    CIFAR10RewardDataset(val_images, val_labels, train=False),
    batch_size=256, shuffle=False, num_workers=args.num_workers)

  model = SmallCNNClassifier(num_classes=10).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs)

  os.makedirs(args.output_dir, exist_ok=True)
  best_val_acc = 0.0
  best_path = os.path.join(args.output_dir, 'best.ckpt')

  for epoch in range(args.epochs):
    model.train()
    running_loss, running_correct, running_total = 0.0, 0, 0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      logits = model(images)
      loss = F.cross_entropy(logits, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * labels.numel()
      running_correct += (logits.argmax(dim=-1) == labels).sum().item()
      running_total += labels.numel()
    scheduler.step()

    train_acc = running_correct / running_total
    train_loss = running_loss / running_total
    val_acc, val_loss = _evaluate(model, val_loader, device)
    print(
      f'epoch {epoch + 1:03d}/{args.epochs} | '
      f'train_loss {train_loss:.4f} train_acc {train_acc:.4f} | '
      f'val_loss {val_loss:.4f} val_acc {val_acc:.4f}',
      flush=True)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
        'epoch': epoch,
        'classes': CIFAR10_CLASSES,
      }, best_path)

  torch.save({
    'model_state_dict': model.state_dict(),
    'val_acc': val_acc,
    'epoch': args.epochs - 1,
    'classes': CIFAR10_CLASSES,
  }, os.path.join(args.output_dir, 'last.ckpt'))
  print(f'Best val_acc: {best_val_acc:.4f} (saved to {best_path})')


def _parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-dir', default='data/cifar10')
  parser.add_argument('--output-dir', default='outputs/cifar10/reward_classifier')
  parser.add_argument('--epochs', type=int, default=30)
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--num-workers', type=int, default=4)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cuda:0')
  return parser.parse_args()


if __name__ == '__main__':
  train(_parse_args())

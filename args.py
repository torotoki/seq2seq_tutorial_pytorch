import os
from argparse import ArgumentParser

def get_args():
  parser = ArgumentParser(description='PyTorch Stack-augmented Decoder Neural Network')
  parser.add_argument('--gpu', type=int, default=1)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--sentence_length', type=int, default=20)
  return parser.parse_args()

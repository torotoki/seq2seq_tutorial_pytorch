# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from utils import asMinutes, timeSince
from dataset import Lang, prepareData
from dataset import indexesFromSentence, variableFromSentence
from dataset import MAX_LENGTH, SOS_token, EOS_token
from model import EncoderRNN, DecoderRNN, AttnDecoderRNN


use_cuda = torch.cuda.is_available()
teacher_forcing_ratio = 0.5

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

def variablesFromPair(pair):
  input_variable = variableFromSentence(input_lang, pair[0])
  target_variable = variableFromSentence(output_lang, pair[1])
  return (input_variable, target_variable)


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
  encoder_hidden = encoder.initHidden()

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  input_length = input_variable.size()[0]
  target_length = target_variable.size()[0]

  encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
  encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

  loss = 0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(
      input_variable[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0][0]

  decoder_input = Variable(torch.LongTensor([[SOS_token]]))
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input

  decoder_hidden = encoder_hidden

  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_outputs)
      loss += criterion(decoder_output, target_variable[di])
      decoder_input = target_variable[di]  # Teacher forcing

  else:
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_outputs)
      topv, topi = decoder_output.data.topk(1)
      ni = topi[0][0]

      decoder_input = Variable(torch.LongTensor([[ni]]))
      decoder_input = decoder_input.cuda() if use_cuda else decoder_input

      loss += criterion(decoder_output, target_variable[di])
      if ni == EOS_token:
        break

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.data[0] / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
  start = time.time()
  plot_losses = []
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every

  encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
  training_pairs = [variablesFromPair(random.choice(pairs))
            for i in range(n_iters)]
  criterion = nn.NLLLoss()

  for iter in range(1, n_iters + 1):
    training_pair = training_pairs[iter - 1]
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    loss = train(input_variable, target_variable, encoder,
           decoder, encoder_optimizer, decoder_optimizer, criterion)
    print_loss_total += loss
    plot_loss_total += loss

    if iter % print_every == 0:
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0
      print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                     iter, iter / n_iters * 100, print_loss_avg))

    if iter % plot_every == 0:
      plot_loss_avg = plot_loss_total / plot_every
      plot_losses.append(plot_loss_avg)
      plot_loss_total = 0

  showPlot(plot_losses)

def showPlot(points):
  plt.figure()
  fig, ax = plt.subplots()
  # this locator puts ticks at regular intervals
  loc = ticker.MultipleLocator(base=0.2)
  ax.yaxis.set_major_locator(loc)
  plt.plot(points)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
  input_variable = variableFromSentence(input_lang, sentence)
  input_length = input_variable.size()[0]
  encoder_hidden = encoder.initHidden()

  encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
  encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_variable[ei],
                         encoder_hidden)
    encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

  decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input

  decoder_hidden = encoder_hidden

  decoded_words = []
  decoder_attentions = torch.zeros(max_length, max_length)

  for di in range(max_length):
    decoder_output, decoder_hidden, decoder_attention = decoder(
      decoder_input, decoder_hidden, encoder_outputs)
    decoder_attentions[di] = decoder_attention.data
    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    if ni == EOS_token:
      decoded_words.append('<EOS>')
      break
    else:
      decoded_words.append(output_lang.index2word[ni])

    decoder_input = Variable(torch.LongTensor([[ni]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

  return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
  for i in range(n):
    pair = random.choice(pairs)
    print('>', pair[0])
    print('=', pair[1])
    output_words, attentions = evaluate(encoder, decoder, pair[0])
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)


if use_cuda:
  encoder1 = encoder1.cuda()
  attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

######################################################################
#

evaluateRandomly(encoder1, attn_decoder1)

output_words, attentions = evaluate(
  encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())


def showAttention(input_sentence, output_words, attentions):
  # Set up figure with colorbar
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(attentions.numpy(), cmap='bone')
  fig.colorbar(cax)

  # Set up axes
  ax.set_xticklabels([''] + input_sentence.split(' ') +
             ['<EOS>'], rotation=90)
  ax.set_yticklabels([''] + output_words)

  # Show label at every tick
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()


def evaluateAndShowAttention(input_sentence):
  output_words, attentions = evaluate(
    encoder1, attn_decoder1, input_sentence)
  print('input =', input_sentence)
  print('output =', ' '.join(output_words))
  showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")

import os
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IWSLT2017
from typing import Iterable, List
from for_forest import dependency_forest
import for_encoder as dep
import datetime
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
from nltk.translate.nist_score import corpus_nist
from nltk.translate.ribes_score import corpus_ribes
import numpy as np
import random
import spacy
mylog = open('iwslt_forest_en_de_recode.log', mode = 'a',encoding='utf-8')

from supar import Parser
parser = Parser.load('biaffine-dep-en')

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'de'
Tokenizer = spacy.load("de_core_news_sm")
# Place-holders
token_transform = {}
vocab_transform = {}

# Create source and target language tokenizer. Make sure to install the dependencies.
# pip install -U spacy
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = IWSLT2017(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=2,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 4096):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size).to(DEVICE)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.10):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)
        self.dependency_forest_layer = dep.DependencyForestLayer(d_model=emb_size, nhead=16)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor,
                dependency_forest):
        src_emb = self.positional_encoding(self.src_tok_emb(src))

        dependency_forest_outs = self.dependency_forest_layer(src = src_emb,dependency_forest = dependency_forest)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        outs = self.transformer(dependency_forest_outs, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def save_model(score,best_score,model):
    if score > best_score:
        best_score = score
        checkpoint = {
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "scheduler_state_dict":scheduler.state_dict()
        }
        torch.save(checkpoint, 'iwslt_forest_en_de_ckpt_best.pt')
    return best_score

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

def get_metric(Tokenizer,TEXT,SRC_LANGUAGE,TGT_LANGUAGE):
    candidate_corpus_list = []
    candidate_corpus_token_list = []
    references_corpus_list = []
    references_corpus_token_list = []
    final = []
    date_iter = IWSLT2017(split=TEXT, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    for src, tgt in date_iter:
        references_corpus_token = []
        references_corpus_list.append(tgt.strip())
        doc = Tokenizer(tgt.strip())
        for token in doc:
            references_corpus_token.append(token.text)
        references_corpus_token_list.append([references_corpus_token])
        translation = (translate(transformer, src))
        candidate_corpus_token = []
        candidate_corpus_list.append(translation.strip())
        doc = Tokenizer(translation.strip())
        for token in doc:
            candidate_corpus_token.append(token.text)
        candidate_corpus_token_list.append(candidate_corpus_token)
    final.append(references_corpus_list)
    bleu1 = corpus_bleu(references_corpus_token_list, candidate_corpus_token_list, weights=(1, 0, 0, 0))*100
    bleu4 = corpus_bleu(references_corpus_token_list, candidate_corpus_token_list, weights=(0.25, 0.25, 0.25, 0.25))*100
    print("BLEU1 score on the " + TEXT + " set is " + str(bleu1),file=mylog)
    print("BLEU4 score on the " + TEXT + " set is " + str(bleu4),file=mylog)
    return bleu4

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1024)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 2048
BATCH_SIZE = 1024
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

from transformers import AdamW, get_linear_schedule_with_warmup
optimizer = AdamW(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 3000, num_training_steps = 300000)

from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch_list, tgt_batch_list, src_sample_list = [], [], []
    for src_sample, tgt_sample in batch:
        src_batch_list.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch_list.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
        src_sample_list.append(src_sample.rstrip("\n"))
    dependency_forest_set = dependency_forest(parser,src_batch_list,src_sample_list)
    src_batch_list = pad_sequence(src_batch_list, padding_value=PAD_IDX)
    tgt_batch_list = pad_sequence(tgt_batch_list, padding_value=PAD_IDX)
    return src_batch_list, tgt_batch_list, dependency_forest_set


from torch.utils.data import DataLoader
def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = IWSLT2017(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt, dependency_forest_set in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        for i in dependency_forest_set[1:]:
            dependency_forest_set[0] = torch.cat((dependency_forest_set[0], i), 0)
        dependency_forest_set = dependency_forest_set[0]
        dependency_forest_set = dependency_forest_set.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask,dependency_forest = dependency_forest_set)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()
        scheduler.step()
        losses += loss.item()
        torch.cuda.empty_cache()
    return losses / len(train_dataloader)


def evaluate(model):
    with torch.no_grad():
        model.eval()
        losses = 0

        val_iter = IWSLT2017(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        for src, tgt,dependency_forest_set in val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            for i in dependency_forest_set[1:]:
                dependency_forest_set[0] = torch.cat((dependency_forest_set[0], i), 0)
            dependency_forest_set = dependency_forest_set[0]
            dependency_forest_set = dependency_forest_set.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask,dependency_forest = dependency_forest_set)

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
        print("Metrics Calculation",file=mylog)
        bleu4 = get_metric(Tokenizer,"valid",SRC_LANGUAGE,TGT_LANGUAGE)
    return losses / len(val_dataloader),bleu4

from timeit import default_timer as timer
NUM_EPOCHS = 100
best_score = 15
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss,bleu4 = evaluate(transformer)
    best_score = save_model(bleu4,best_score,transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"),file=mylog)
    print(file=mylog)
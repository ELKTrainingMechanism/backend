import os
os.system("pip install fancy_einsum einops datasets transformers git+https://github.com/neelnanda-io/Easy-Transformer.git@clean-transformer-demo matplotlib plotly")
os.system("mkdir gptfiles")

#importing necessary modules
import einops
from fancy_einsum import einsum
from dataclasses import dataclass
from easy_transformer import EasyTransformer
import torch
import torch.nn as nn
import numpy as np
import math
from easy_transformer.utils import get_corner, gelu_new, tokenize_and_concatenate
import tqdm.auto as tqdm
import datasets
import transformers
import plotly.express as px

torch.manual_seed(42)

import sys
import json

def get_value(string_arg):
    # Find the index of the colon
    colon_index = string_arg.index(':')
    # Extract the substring after the colon
    substring = string_arg[colon_index + 1:]
    # Convert the substring to an integer
    integer_value = int(substring)
    return integer_value 

# # Read input value from command-line arguments
input_d_model = get_value(sys.argv[1])
input_n_heads = get_value(sys.argv[6])
input_d_head = get_value(sys.argv[4])
input_d_mlp = get_value(sys.argv[5])
input_n_layers = get_value(sys.argv[7])

input_n_ctx = get_value(sys.argv[1])

input_d_model2 = sys.argv[8]
input_n_heads2 = sys.argv[13]
input_d_head2 = sys.argv[11]
input_d_mlp2 = sys.argv[12]
input_n_layers2 = sys.argv[14]
# print(input_value)
# input_value = eval(input_value)
# print("The input type is: ")
# output_value = input_value['d_model']

# # Print the output value
# print(output_value)
# print("excuse me")
# print("\n Exiting SSH. ")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

output = f"The device is: {device}"

# Save the output to a file
with open("result.txt", "w") as file:
    file.write(output)

# defining all necessary classes for the DemoTransformerClass
@dataclass
class Config:
    name: str = 'None'
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()
print(cfg)

output = f"The config is: {cfg}"

# Save the output to a file
with open("result.txt", "w") as file:
    file.write(output)

class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        embed = self.W_E[tokens, :] # [batch, position, d_model]
        if self.cfg.debug: print("Embeddings:", embed.shape)
        return embed

class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, residual):
        # residual: [batch, position, d_model]
        if self.cfg.debug: print("Residual:", residual.shape)
        residual = residual - einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1", "mean") + cfg.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        if self.cfg.debug: print("Normalized:", residual.shape)
        return normalized

class PosEmbed(nn.Module):
  def __init__(self, cfg):
      super().__init__()
      self.cfg = cfg
      self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
      nn.init.normal_(self.W_pos, std=self.cfg.init_range)

  def forward(self, tokens):
      # tokens: [batch, position]
      if self.cfg.debug: print("Tokens:", tokens.shape)
      pos_embed = self.W_pos[:tokens.size(1), :] # [position, d_model]
      pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
      if self.cfg.debug: print("pos_embed:", pos_embed.shape)
      return pos_embed

class Attention(nn.Module):
  def __init__(self, cfg):
      super().__init__()
      self.cfg = cfg
      self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
      nn.init.normal_(self.W_Q, std=self.cfg.init_range)
      self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
      self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
      nn.init.normal_(self.W_K, std=self.cfg.init_range)
      self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
      self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
      nn.init.normal_(self.W_V, std=self.cfg.init_range)
      self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))

      self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
      nn.init.normal_(self.W_O, std=self.cfg.init_range)
      self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))

      self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device=device))

  def forward(self, normalized_resid_pre):
      # normalized_resid_pre: [batch, position, d_model]
      if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)

      q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q
      k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K

      attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
      attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
      attn_scores = self.apply_causal_mask(attn_scores)

      pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]

      v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V

      z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)

      attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", z, self.W_O) + self.b_O
      return attn_out

  def apply_causal_mask(self, attn_scores):
      # attn_scores: [batch, n_heads, query_pos, key_pos]
      mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
      attn_scores.masked_fill_(mask, self.IGNORE)
      return attn_scores

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))

    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_mid:", normalized_resid_mid.shape)
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid, self.W_in) + self.b_in
        post = gelu_new(pre)
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
        return mlp_out

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        # resid_pre [batch, position, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre)
        resid_mid = resid_pre + attn_out

        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        resid_post = resid_mid + mlp_out
        return resid_post

class Unembed(nn.Module):
  def __init__(self, cfg):
      super().__init__()
      self.cfg = cfg
      self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
      nn.init.normal_(self.W_U, std=self.cfg.init_range)
      self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))

  def forward(self, normalized_resid_final):
      # normalized_resid_final [batch, position, d_model]
      if self.cfg.debug: print("Normalized_resid_final:", normalized_resid_final.shape)
      logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_final, self.W_U) + self.b_U
      return logits

# defining subclass to initialize transformers
class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        # tokens [batch, position]
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed
        for block in self.blocks:
            residual = block(residual)
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        # logits have shape [batch, position, logits]
        return logits

# loading reference gpt small and medium models for their tokenizers and vocabulary
reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

def lm_cross_entropy_loss(logits, tokens):
    # Measure next token loss
    # Logits have shape [batch, position, d_vocab]
    # Tokens have shape [batch, position]
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()

def save_checkpoint(model, optimizer, epoch, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)

def calculate_metrics(model):
    total_loss = 0
    total_words = 0

    with torch.no_grad():
        for batch in (tester_data_loader):
            tokens = batch['tokens'].cuda()
            logits = model(tokens)

            # Calculate the cross-entropy loss
            loss = lm_cross_entropy_loss(logits, tokens)

            total_loss += loss.item() * tokens.numel()
            total_words += tokens.numel()

    # Calculate average loss and perplexity
    average_loss = total_loss / total_words
    perplexity = math.exp(average_loss)
    return perplexity, average_loss

# initializing the training hyperparameters
batch_size = 1
num_epochs = 1
max_steps = 1
log_every = 10
lr = 1e-3
weight_decay = 1e-2

# loading the training dataset for the model
dataset = datasets.load_dataset('wikitext','wikitext-103-raw-v1', split="train")
testerdataset = datasets.load_dataset('wikitext','wikitext-103-raw-v1', split="test")

# tokenization function
def tokenize_dataset(dataset,model_cfg,batch_size):
  tokens_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model_cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
  data_loader = torch.utils.data.DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
  return data_loader

# train function
def train_model(data_loader, model, num_epochs, max_steps):
  # defining the optimizer to be used
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
  training_losses = []
  losses = []
  perplexities = []
  total_loss = 0
  total_words = 0
  print("Number of steps:", max_steps)
  print("Number of batches:", len(data_loader))
  for epoch in range(num_epochs):
      for c, batch in tqdm.tqdm(enumerate(data_loader)):
          tokens = batch['tokens'].cuda()
          logits = model(tokens)
          loss = lm_cross_entropy_loss(logits, tokens)
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          training_losses.append(loss.item())
          filepath = '/gptfiles/' + model.cfg.name

          # Calculate average loss and perplexity
          if c % 100 == 0:
              iteration_perplexity, iteration_loss = calculate_metrics(model)
              losses.append(iteration_loss)
              perplexities.append(iteration_perplexity)
              print(f"Step: {c}, Loss: {iteration_loss:.4f}, Perplexity: {iteration_perplexity:.4f}")
          if c % 1000 == 0:
              save_checkpoint(model, optimizer, epoch, filepath)
          if c > max_steps:
              save_checkpoint(model, optimizer, epoch, filepath)
              break

  return training_losses, losses, perplexities

# initializing configuration of small model
model_cfg_small = Config(name='small', debug=False, d_model=input_d_model, n_heads=input_n_heads, d_head=input_d_head, d_mlp=input_d_mlp, n_layers=input_n_layers, n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)
# initializing the small model

output = f"The small model config is: {model_cfg_small}"

# Save the output to a file
with open("result.txt", "w") as file:
    file.write(output)

model_small = DemoTransformer(model_cfg_small)
data_loader = tokenize_dataset(dataset,model_small.cfg,batch_size)
tester_data_loader = tokenize_dataset(testerdataset,model_small.cfg,batch_size)
model_small.cuda()

small_training_losses, small_losses, small_perplexities = train_model(data_loader, model_small, num_epochs, max_steps)

output = f"Training Losses: {small_training_losses[0]:.4f}, Loss: {small_losses[0]:.4f}, Perplexity: {small_perplexities[0]:.4f}"

# Save the output to a file
with open("result.txt", "w") as file:
    file.write(output)

# small_model = DemoTransformer(model_small.cfg)
# small_model.load_state_dict(torch.load('/py/gptfiles/small')['model_state_dict'])
# small_model.cuda()
# checkpoint_small_training_losses, checkpoint_small_losses, checkpoint_small_perplexities = train_model(data_loader, small_model, num_epochs, 5000)

# small_losses.extend(checkpoint_small_losses)
# small_training_losses.extend(checkpoint_small_training_losses)
# small_perplexities.extend(checkpoint_small_perplexities)

# # initializing configuration of medium (& scaled) model
# model_cfg_medium = Config(name='medium', debug=False, d_model=input_value['d_model2'], n_heads=input_value['n_heads2'], d_head=input_value['d_head2'], d_mlp=input_value['d_mlp2'], n_layers=input_value['n_layers2'], n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)

# # initializing the medium model
# model_medium = DemoTransformer(model_cfg_medium)
# model_medium.cuda()

# medium_training_losses, medium_losses, medium_perplexities = train_model(data_loader, model_medium, num_epochs,  max_steps)

# medium_model = DemoTransformer(model_medium.cfg)
# medium_model.load_state_dict(torch.load('/py/gptfiles/medium')['model_state_dict'])
# medium_model.cuda()
# checkpoint_medium_training_losses, checkpoint_medium_losses, checkpoint_medium_perplexities = train_model(data_loader, medium_model, num_epochs, 5000)

# medium_losses.extend(checkpoint_medium_losses)
# medium_training_losses.extend(checkpoint_medium_training_losses)
# medium_perplexities.extend(checkpoint_medium_perplexities)


# class EmbedScale(nn.Module):
#     def __init__(self, cfg, model):
#         super().__init__()
#         self.cfg = cfg
#         self.W_E_old = model.embed.W_E.data
#         self.W_E_new = torch.empty((cfg.d_vocab, cfg.d_model - model.cfg.d_model))
#         nn.init.normal_(self.W_E_new, std=self.cfg.init_range)
#         self.W_E_new.data = self.W_E_new.data.to(self.W_E_old.device)  # Move W_E_new to the same device as W_E_old
#         self.W_E = nn.Parameter(torch.cat((self.W_E_old,self.W_E_new),dim=1))

#     def forward(self, tokens):
#         # tokens: [batch, position]
#         if self.cfg.debug: print("Tokens:", tokens.shape)
#         embed = self.W_E[tokens, :] # [batch, position, d_model]
#         if self.cfg.debug: print("Embeddings:", embed.shape)
#         return embed

# class LayerNormScale(nn.Module):
#     def __init__(self, cfg, model, block_num, layernormclass):
#         super().__init__()
#         self.cfg = cfg

#         if layernormclass == "layer1":
#             self.w_old = model.blocks[block_num].ln1.w.data
#             self.b_old = model.blocks[block_num].ln1.b.data

#         if layernormclass == "layer2":
#             self.w_old = model.blocks[block_num].ln2.w.data
#             self.b_old = model.blocks[block_num].ln2.b.data

#         if layernormclass == "final":
#             self.w_old = model.ln_final.w.data
#             self.b_old = model.ln_final.b.data

#         self.w_new = torch.ones(cfg.d_model - model.cfg.d_model)
#         self.b_new = torch.zeros(cfg.d_model - model.cfg.d_model)
#         self.w_new.data = self.w_new.data.to(self.w_old.device)  # Move W_E_new to the same device as W_E_old
#         self.b_new.data = self.b_new.data.to(self.w_old.device)  # Move W_E_new to the same device as W_E_old
#         self.w = nn.Parameter(torch.cat((self.w_old,self.w_new),dim=0))
#         self.b = nn.Parameter(torch.cat((self.b_old,self.b_new),dim=0))

#     def forward(self, residual):
#         # residual: [batch, position, d_model]
#         if self.cfg.debug: print("Residual:", residual.shape)
#         residual = residual - einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
#         # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
#         scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1", "mean") + cfg.layer_norm_eps).sqrt()
#         normalized = residual / scale
#         normalized = normalized * self.w + self.b
#         if self.cfg.debug: print("Normalized:", residual.shape)
#         return normalized

# class PosEmbedScale(nn.Module):
#     def __init__(self, cfg, model):
#         super().__init__()
#         self.cfg = cfg
#         self.W_pos_old = model.pos_embed.W_pos.data
#         self.W_pos_new = torch.empty((cfg.n_ctx, cfg.d_model - model.cfg.d_model))
#         nn.init.normal_(self.W_pos_new, std=self.cfg.init_range)
#         self.W_pos_new.data = self.W_pos_new.data.to(self.W_pos_old.device)  # Move W_E_new to the same device as W_E_old
#         self.W_pos = nn.Parameter(torch.cat((self.W_pos_old,self.W_pos_new),dim=1))

#     def forward(self, tokens):
#         # tokens: [batch, position]
#         if self.cfg.debug: print("Tokens:", tokens.shape)
#         pos_embed = self.W_pos[:tokens.size(1), :] # [position, d_model]
#         pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
#         if self.cfg.debug: print("pos_embed:", pos_embed.shape)
#         return pos_embed

# class AttentionScale(nn.Module):
#     def __init__(self, cfg, model, block_num):
#         super().__init__()
#         self.cfg = cfg

#         model_d_model = model.cfg.d_model
#         model_n_heads = model.cfg.n_heads
#         model_d_heads = model.cfg.d_head

#         # W_Q
#         self.W_Q_old = model.blocks[block_num].attn.W_Q.data
#         self.W_Q_cat1 = torch.empty((model_n_heads, model_d_model, cfg.d_head - model_d_heads))
#         nn.init.normal_(self.W_Q_cat1, std=self.cfg.init_range)
#         self.W_Q_cat2 = torch.empty((model_n_heads, cfg.d_model - model_d_model, cfg.d_head))
#         nn.init.normal_(self.W_Q_cat2, std=self.cfg.init_range)
#         self.W_Q_cat1.data = self.W_Q_cat1.data.to(self.W_Q_old.device)
#         self.W_Q_cat2.data = self.W_Q_cat2.data.to(self.W_Q_old.device)
#         self.W_Q_oldhead = torch.cat((self.W_Q_old, self.W_Q_cat1), dim=2)
#         self.W_Q_oldhead = torch.cat((self.W_Q_oldhead, self.W_Q_cat2), dim=1)

#         # W_K
#         self.W_K_old = model.blocks[block_num].attn.W_K.data
#         self.W_K_cat1 = torch.empty((model_n_heads, model_d_model, cfg.d_head - model_d_heads))
#         nn.init.normal_(self.W_K_cat1, std=self.cfg.init_range)
#         self.W_K_cat2 = torch.empty((model_n_heads, cfg.d_model - model_d_model, cfg.d_head))
#         nn.init.normal_(self.W_K_cat2, std=self.cfg.init_range)
#         self.W_K_cat1.data = self.W_K_cat1.data.to(self.W_K_old.device)
#         self.W_K_cat2.data = self.W_K_cat2.data.to(self.W_K_old.device)
#         self.W_K_oldhead = torch.cat((self.W_K_old, self.W_K_cat1), dim=2)
#         self.W_K_oldhead = torch.cat((self.W_K_oldhead, self.W_K_cat2), dim=1)

#         # W_V
#         self.W_V_old = model.blocks[block_num].attn.W_V.data
#         self.W_V_cat1 = torch.empty((model_n_heads, model_d_model, cfg.d_head - model_d_heads))
#         nn.init.normal_(self.W_V_cat1, std=self.cfg.init_range)
#         self.W_V_cat2 = torch.empty((model_n_heads, cfg.d_model - model_d_model, cfg.d_head))
#         nn.init.normal_(self.W_V_cat2, std=self.cfg.init_range)
#         self.W_V_cat1.data = self.W_V_cat1.data.to(self.W_V_old.device)
#         self.W_V_cat2.data = self.W_V_cat2.data.to(self.W_V_old.device)
#         self.W_V_oldhead = torch.cat((self.W_V_old, self.W_V_cat1), dim=2)
#         self.W_V_oldhead = torch.cat((self.W_V_oldhead, self.W_V_cat2), dim=1)

#       # W_O
#         self.W_O_old = model.blocks[block_num].attn.W_O.data
#         self.W_O_cat1 = torch.empty((model_n_heads, model_d_heads, cfg.d_model - model_d_model))
#         nn.init.normal_(self.W_O_cat1, std=self.cfg.init_range)
#         self.W_O_cat2 = torch.empty((model_n_heads, cfg.d_head - model_d_heads, cfg.d_model))
#         nn.init.normal_(self.W_O_cat2, std=self.cfg.init_range)
#         self.W_O_cat1.data = self.W_O_cat1.data.to(self.W_O_old.device)
#         self.W_O_cat2.data = self.W_O_cat2.data.to(self.W_O_old.device)
#         self.W_O_oldhead = torch.cat((self.W_O_old, self.W_O_cat1), dim=2)
#         self.W_O_oldhead = torch.cat((self.W_O_oldhead, self.W_O_cat2), dim=1)

#         # b_Q
#         self.b_Q_old = model.blocks[block_num].attn.b_Q.data
#         self.b_Q_new = torch.zeros((model_n_heads, cfg.d_head - model_d_heads))
#         self.b_Q_new.data = self.b_Q_new.data.to(self.b_Q_old.device)
#         self.b_Q_oldhead = torch.cat((self.b_Q_old, self.b_Q_new), dim=1)

#         # b_K
#         self.b_K_old = model.blocks[block_num].attn.b_K.data
#         self.b_K_new = torch.zeros((model_n_heads, cfg.d_head - model_d_heads))
#         self.b_K_new.data = self.b_K_new.data.to(self.b_K_old.device)
#         self.b_K_oldhead = torch.cat((self.b_K_old, self.b_K_new), dim=1)

#         # b_V
#         self.b_V_old = model.blocks[block_num].attn.b_V.data
#         self.b_V_new = torch.zeros((model_n_heads, cfg.d_head - model_d_heads))
#         self.b_V_new.data = self.b_V_new.data.to(self.b_V_old.device)
#         self.b_V_oldhead = torch.cat((self.b_V_old, self.b_V_new), dim=1)

#         # b_O
#         self.b_O_old = model.blocks[block_num].attn.b_O.data
#         self.b_O_new = torch.zeros((cfg.d_model - model_d_model))
#         self.b_O_new.data = self.b_O_new.data.to(self.b_O_old.device)
#         self.b_O = nn.Parameter(torch.cat((self.b_O_old, self.b_O_new), dim=0))

#         # new injected heads
#         self.W_Q_newhead = torch.empty((cfg.n_heads - model_n_heads, cfg.d_model, cfg.d_head))
#         nn.init.normal_(self.W_Q_newhead, std=self.cfg.init_range)
#         self.b_Q_newhead = torch.zeros((cfg.n_heads - model_n_heads, cfg.d_head))

#         self.W_K_newhead = torch.empty((cfg.n_heads - model_n_heads, cfg.d_model, cfg.d_head))
#         nn.init.normal_(self.W_K_newhead, std=self.cfg.init_range)
#         self.b_K_newhead = torch.zeros((cfg.n_heads - model_n_heads, cfg.d_head))

#         self.W_V_newhead = torch.empty((cfg.n_heads - model_n_heads, cfg.d_model, cfg.d_head))
#         nn.init.normal_(self.W_V_newhead, std=self.cfg.init_range)
#         self.b_V_newhead = torch.zeros((cfg.n_heads - model_n_heads, cfg.d_head))

#         self.W_O_newhead = torch.empty((cfg.n_heads - model_n_heads, cfg.d_head, cfg.d_model))
#         nn.init.normal_(self.W_O_newhead, std=self.cfg.init_range)

#         self.W_Q_newhead.data = self.W_Q_newhead.to(self.W_Q_old.device)
#         self.b_Q_newhead.data = self.b_Q_newhead.to(self.b_V_old.device)
#         self.W_K_newhead.data = self.W_K_newhead.to(self.W_Q_old.device)
#         self.b_K_newhead.data = self.b_K_newhead.to(self.b_V_old.device)
#         self.W_V_newhead.data = self.W_V_newhead.to(self.W_Q_old.device)
#         self.b_V_newhead.data = self.b_V_newhead.to(self.b_V_old.device)
#         self.W_O_newhead.data = self.W_O_newhead.to(self.b_V_old.device)

#         self.W_Q = nn.Parameter(torch.cat((self.W_Q_oldhead,self.W_Q_newhead),dim=0))
#         self.b_Q = nn.Parameter(torch.cat((self.b_Q_oldhead,self.b_Q_newhead),dim=0))
#         self.W_K = nn.Parameter(torch.cat((self.W_K_oldhead,self.W_K_newhead),dim=0))
#         self.b_K = nn.Parameter(torch.cat((self.b_K_oldhead,self.b_K_newhead),dim=0))
#         self.W_V = nn.Parameter(torch.cat((self.W_V_oldhead,self.W_V_newhead),dim=0))
#         self.b_V = nn.Parameter(torch.cat((self.b_V_oldhead,self.b_V_newhead),dim=0))
#         self.W_O = nn.Parameter(torch.cat((self.W_O_oldhead,self.W_O_newhead),dim=0))

#         self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device=device))

#     def forward(self, normalized_resid_pre):
#         # normalized_resid_pre: [batch, position, d_model]
#         if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)

#         q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q
#         k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K

#         attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
#         attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
#         attn_scores = self.apply_causal_mask(attn_scores)

#         pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]

#         v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V

#         z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)

#         attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", z, self.W_O) + self.b_O
#         return attn_out

#     def apply_causal_mask(self, attn_scores):
#         # attn_scores: [batch, n_heads, query_pos, key_pos]
#         mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
#         attn_scores.masked_fill_(mask, self.IGNORE)
#         return attn_scores

# class TransformerBlockScale(nn.Module):
#     def __init__(self, cfg, model, block_num):
#         super().__init__()
#         self.cfg = cfg

#         self.ln1 = LayerNormScale(cfg, model, block_num, "layer1")
#         self.attn = AttentionScale(cfg, model, block_num)
#         self.ln2 = LayerNormScale(cfg, model, block_num, "layer2")
#         self.mlp = MLPScale(cfg, model, block_num)

#     def forward(self, resid_pre):
#         # resid_pre [batch, position, d_model]
#         normalized_resid_pre = self.ln1(resid_pre)
#         attn_out = self.attn(normalized_resid_pre)
#         resid_mid = resid_pre + attn_out

#         normalized_resid_mid = self.ln2(resid_mid)
#         mlp_out = self.mlp(normalized_resid_mid)
#         resid_post = resid_mid + mlp_out
#         return resid_post

# class MLPScale(nn.Module):
#     def __init__(self, cfg, model, block_num):
#         super().__init__()
#         self.cfg = cfg

#         self.W_in_old = model.blocks[block_num].mlp.W_in.data
#         self.W_in_cat1 = torch.empty((model.cfg.d_model, cfg.d_mlp - model.cfg.d_mlp))
#         nn.init.normal_(self.W_in_cat1, std=self.cfg.init_range)
#         self.W_in_cat2 = torch.empty((cfg.d_model - model.cfg.d_model, cfg.d_mlp))
#         nn.init.normal_(self.W_in_cat2, std=self.cfg.init_range)
#         self.W_in_cat1.data = self.W_in_cat1.data.to(self.W_in_old.device)
#         self.W_in_cat2.data = self.W_in_cat2.data.to(self.W_in_old.device)
#         self.W_in = torch.cat((self.W_in_old,self.W_in_cat1),dim=1)
#         self.W_in = nn.Parameter(torch.cat((self.W_in,self.W_in_cat2),dim=0))

#         self.b_in_old = model.blocks[block_num].mlp.b_in.data
#         self.b_in_new = torch.zeros((cfg.d_mlp - model.cfg.d_mlp))
#         self.b_in_new.data = self.b_in_new.data.to(self.b_in_old.device)
#         self.b_in = nn.Parameter(torch.cat((self.b_in_old,self.b_in_new),dim=0))

#         self.W_out_old = model.blocks[block_num].mlp.W_out.data
#         self.W_out_cat1 = torch.empty((model.cfg.d_mlp, cfg.d_model - model.cfg.d_model))
#         nn.init.normal_(self.W_out_cat1, std=self.cfg.init_range)
#         self.W_out_cat2 = torch.empty((cfg.d_mlp - model.cfg.d_mlp, cfg.d_model))
#         nn.init.normal_(self.W_out_cat2, std=self.cfg.init_range)
#         self.W_out_cat1.data = self.W_out_cat1.data.to(self.W_out_old.device)
#         self.W_out_cat2.data = self.W_out_cat2.data.to(self.W_out_old.device)
#         self.W_out = torch.cat((self.W_out_old, self.W_out_cat1), dim=1)
#         self.W_out = nn.Parameter(torch.cat((self.W_out, self.W_out_cat2), dim=0))

#         self.b_out_old = model.blocks[block_num].mlp.b_out.data
#         self.b_out_new = torch.zeros((cfg.d_model - model.cfg.d_model))
#         self.b_out_new.data = self.b_out_new.data.to(self.b_out_old.device)
#         self.b_out = nn.Parameter(torch.cat((self.b_out_old, self.b_out_new), dim=0))

#     def forward(self, normalized_resid_mid):
#         # normalized_resid_mid: [batch, position, d_model]
#         if self.cfg.debug: print("Normalized_resid_mid:", normalized_resid_mid.shape)
#         pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid, self.W_in) + self.b_in
#         post = gelu_new(pre)
#         mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
#         return mlp_out

# class UnembedScale(nn.Module):
#   def __init__(self, cfg, model):
#       super().__init__()
#       self.cfg = cfg
#       self.W_U_old = model.unembed.W_U.data
#       self.W_U_new = torch.zeros((cfg.d_model - model.cfg.d_model, cfg.d_vocab))
#       self.W_U_new.data = self.W_U_new.data.to(self.W_U_old.device)
#       self.W_U = nn.Parameter(torch.cat((self.W_U_old,self.W_U_new),dim=0))
#       self.b_U = nn.Parameter(model.unembed.b_U)

#   def forward(self, normalized_resid_final):
#       # normalized_resid_final [batch, position, d_model]
#       if self.cfg.debug: print("Normalized_resid_final:", normalized_resid_final.shape)
#       logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_final, self.W_U) + self.b_U
#       return logits

# # defining subclass to initialize scaled up transformers
# class ScaledUpTransformer(nn.Module):
#     def __init__(self, cfg, model):
#         super().__init__()
#         self.cfg = cfg
#         self.cfg.name = 'scaled'
#         self.embed = EmbedScale(cfg, model)
#         self.pos_embed = PosEmbedScale(cfg, model)
#         self.scaledblocks = nn.ModuleList([TransformerBlockScale(cfg, model, block_num) for block_num in range(model.cfg.n_layers)])
#         self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers - model.cfg.n_layers)])
#         self.ln_final = LayerNormScale(cfg, model, 0, "final")
#         self.unembed = UnembedScale(cfg, model)

#     def forward(self, tokens):
#         # tokens [batch, position]
#         embed = self.embed(tokens)
#         pos_embed = self.pos_embed(tokens)
#         residual = embed + pos_embed
#         for scaledblock in self.scaledblocks:
#             residual = scaledblock(residual)
#         for block in self.blocks:
#             residual = block(residual)
#         normalized_resid_final = self.ln_final(residual)
#         logits = self.unembed(normalized_resid_final)
#         # logits have shape [batch, position, logits]
#         return logits


# # initializing the scaled model
# model_scaled = ScaledUpTransformer(model_cfg_medium, model_small)
# model_scaled.cuda()

# scaled_training_losses, scaled_losses, scaled_perplexities = train_model(data_loader, model_scaled, num_epochs, max_steps)

# new_model = DemoTransformer(model_small.cfg)
# scaled_model = ScaledUpTransformer(model_medium.cfg, new_model)
# scaled_model.load_state_dict(torch.load('/py/gptfiles/scaled')['model_state_dict'])
# scaled_model.cuda()

# checkpoint_scaled_training_losses, checkpoint_scaled_losses, checkpoint_scaled_perplexities = train_model(data_loader, scaled_model, num_epochs, 5000)

# scaled_losses.extend(checkpoint_scaled_losses)
# scaled_training_losses.extend(checkpoint_scaled_training_losses)
# scaled_perplexities.extend(checkpoint_scaled_perplexities)

# import plotly.graph_objects as go
# import numpy as np

# # Define the losses for each model
# model_training_losses = {
#     "Small": small_training_losses,
#     "Medium": medium_training_losses,
#     "Scaled": scaled_training_losses,
# }

# # Define the losses for each model
# model_losses = {
#     "Small": small_losses,
#     "Medium": medium_losses,
#     "Scaled": scaled_losses,
# }

# # Define the perplexities for each model
# model_perplexities = {
#     "Small": small_perplexities,
#     "Medium": medium_perplexities,
#     "Scaled": scaled_perplexities,
# }

# # Create a list of x-values
# x_values = np.arange(len(medium_losses)) * (model_cfg_small.n_ctx * batch_size)

# # Create the figure
# loss_graph = go.Figure()
# perplexity_graph = go.Figure()
# training_loss_graph = go.Figure()

# # Add line traces for each model
# for model_name, losses in model_losses.items():
#     loss_graph.add_trace(go.Scatter(x=x_values, y=losses, name=model_name))

# # Add line traces for each model
# for model_name, perplexities in model_perplexities.items():
#     perplexity_graph.add_trace(go.Scatter(x=x_values, y=perplexities, name=model_name))

# # Add line traces for each model
# for model_name, training_losses in model_training_losses.items():
#     training_loss_graph.add_trace(go.Scatter(x=x_values, y=training_losses, name=model_name))

# # Update the layout for the graph
# training_loss_graph.update_layout(
#     title="Cross-entropy loss on the training distribution for 35K iterations",
#     xaxis=dict(title="Tokens"),
#     yaxis=dict(title="Training Loss")
# )

# # Show the plot
# training_loss_graph.show()

# # Update the layout for the graph
# loss_graph.update_layout(
#     title="Cross-entropy loss for training for 35K iterations",
#     xaxis=dict(title="Tokens"),
#     yaxis=dict(title="Loss")
# )

# # Show the plot
# loss_graph.show()

# perplexity_graph.update_layout(
#     title="Perplexity for training for 35K iterations",
#     xaxis=dict(title="Tokens"),
#     yaxis=dict(title="Perplexity")
# )

# # Show the plot
# perplexity_graph.show()

# output_value = small_losses[-1]
# print(output_value)
from x_transformers import TransformerWrapper, Decoder, Encoder, XTransformer
import torch

L = 10
D = 512
B = 2

model = XTransformer(
    dim = 512,
    enc_num_tokens = 3,
    enc_depth = 6,
    enc_heads = 8,
    enc_max_seq_len = 10,
    dec_num_tokens = 3,
    dec_depth = 6,
    dec_heads = 8,
    dec_max_seq_len = 10,
    tie_token_emb = True,      # tie embeddings of encoder and decoder
    enc_rotary_pos_emb = True,
    dec_rotary_pos_emb = True
).to('cuda')


src = torch.randint(0, 3, (B,L),device='cuda',dtype=torch.long)
src_mask = torch.ones_like(src).bool()
src_mask[:,-1] = False
output = model(src, src, src_mask=src_mask, tgt_mask=~src_mask)

print(output.shape)
print(output)
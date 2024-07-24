
import torch
import math


def sequence_mask(x, valid_lens, value=0):
    max_len = x.size(1)
    mask = torch.arange(max_len, dtype=torch.float32,device=x.device)[None,:]<valid_lens[:,None]
    x[~mask] = value
    return x

def masked_softmax(x, valid_lens):
    if valid_lens is None:
        return torch.softmax(x,dim=-1)
    else:
        shape = x.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        x = sequence_mask(x.reshape(-1, shape[-1]), valid_lens, -1e6)
        return torch.softmax(x.reshape(shape),dim=-1)
    
class DotProductAttention(torch.nn.Module):
    def __init__(self, dropout, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens):
        d= queries.size(-1)
        scores = torch.bmm(queries, keys.transpose(1,2))/math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)



def transpose_qkv(x, num_heads):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = x.permute(0,2,1,3)
    x = x.reshape(-1,x.shape[2],x.shape[3])
    return x

def transpose_output(x, num_heads):
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0,2,1,3)
    x = x.reshape(x.shape[0],x.shape[1],-1)
    return x

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, bias=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.w_q = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.w_k = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.w_v = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.w_o = torch.nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.w_q(queries), self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,repeats = self.num_heads, dim = 0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.w_o(output_concat)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_size, dropout, max_len=1000, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.p = torch.zeros((1, max_len, hidden_size))
        x = torch.arange(max_len,dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, hidden_size, 2, dtype=torch.float32)/hidden_size)
        self.p[:,:,0::2] = torch.sin(x)
        self.p[:,:,1::2] = torch.cos(x)

    def forward(self, x):
        x = x + self.p[:,:x.shape[1], :].to(x.device)
        return self.dropout(x)

from .conv.seq2seq import ConvS2S
from .rnn.seq2seq import Seq2Seq
from .transformer.transformer import Transformer
from .dynamic_conv.seq2seq import DynamicConvS2S


def get_model(model_type, in_dim, out_dim, max_len, mcfg):
    """ Model dispatcher
    params:
        model_type: str
        mcfg: model config
        in_dim: input vocab size
        out_dim: output vocab size
    """
    if model_type == 'rnn':
        h_dim = mcfg['h_dim']
        emb_dim = mcfg['emb_dim']
        bidirect = mcfg['bidirect']
        attention_type = mcfg['attention_type']
        dropout = mcfg['dropout']
        enc_layers = mcfg['enc_layers']
        dec_layers = mcfg['dec_layers']
        seq2seq = Seq2Seq(in_dim, emb_dim, h_dim, out_dim, enc_layers, dec_layers,
                          enc_bidirect=bidirect, dropout=dropout,
                          attention=attention_type, max_len=max_len)
    elif model_type == 'conv':
        h_dim = mcfg['h_dim']
        emb_dim = mcfg['emb_dim']
        enc_layers = mcfg['enc_layers']
        dec_layers = mcfg['dec_layers']
        kernel_size = mcfg['kernel_size']
        dropout = mcfg['dropout']
        cache_mode = mcfg['cache_mode']
        seq2seq = ConvS2S(in_dim, emb_dim, h_dim, out_dim, enc_layers, dec_layers,
                          kernel_size=kernel_size, dropout=dropout, max_len=max_len,
                          cache_mode=cache_mode)
    elif model_type == 'transformer':
        d_model = mcfg['d_model']
        d_ff = mcfg['d_ff']
        n_layers = mcfg['n_layers']
        n_heads = mcfg['n_heads']
        dropout = mcfg['dropout']
        norm_pos = mcfg['norm_pos']
        seq2seq = Transformer(in_dim, out_dim, max_len, d_model, d_ff, n_layers, n_heads, dropout,
                              norm_pos)
    elif model_type == 'dynamic_conv':
        conv_type = mcfg['conv_type']
        kernel_sizes = mcfg['kernel_sizes']
        d_model = mcfg['d_model']
        d_ff = mcfg['d_ff']
        n_heads = mcfg['n_heads']
        dropout = mcfg['dropout']
        norm_pos = mcfg['norm_pos']
        seq2seq = DynamicConvS2S(in_dim, out_dim, max_len, conv_type, kernel_sizes, d_model, d_ff,
                                 n_heads, dropout, norm_pos)
    else:
        raise ValueError(model_type)

    return seq2seq

"""Portable PGOE primitives. Thickness/wavelength units are micrometres."""
import numpy as np

W = np.linspace(.45, .85, 41)
MAX = 24
ARMS = ('full_aug', 'generic_aug', 'no_phase_interface', 'no_global',
        'generic_mlp', 'full_mlp', 'prism_nm_rope', 'prism_scaled_rope',
        'prism_nm_rope_b1000', 'prism_scaled_rope_b1000')


def sample(rng, L, N, domain='normal'):
    if not 1 < L <= MAX:
        raise ValueError('Layer count must be between 2 and 24')
    if domain not in ('normal', 'high', 'high_index', 'dispersion', 'strong'):
        raise ValueError(domain)
    a = rng.uniform(2.1, 2.6, (N, L)) if domain in ('high', 'high_index') else rng.uniform(1.35, 2.05, (N, L))
    b = rng.uniform(.08, .12, (N, L)) if domain in ('dispersion', 'strong') else rng.uniform(0, .03, (N, L))
    return a, b, rng.uniform(.065, .18, (N, L))


def indices(a, b):
    return a[..., None] + b[..., None] * ((.65 / W)**2 - 1)


def tmm(a, b, d):
    ns = indices(a, b)
    if not (np.all(ns > 0) and np.all(d > 0)):
        raise ValueError('Positive refractive indices and thicknesses required')
    A = np.ones((len(a), len(W)), complex)
    B = np.zeros_like(A); C = B.copy(); D = A.copy()
    for j in range(a.shape[1]):
        n = ns[:, j]; phase = 2*np.pi*n*d[:, j, None]/W
        co = np.cos(phase); si = 1j*np.sin(phase)
        A, B, C, D = A*co+B*si*n, A*si/n+B*co, C*co+D*si*n, C*si/n+D*co
    den = A+B+C+D
    return abs((A+B-C-D)/den)**2, abs(2/den)**2


def recursive_reflectance(a, b, d):
    n = indices(a, b); r = (n[:, -1]-1)/(n[:, -1]+1)
    for j in range(a.shape[1]-2, -1, -1):
        q = r*np.exp(4j*np.pi*n[:, j+1]*d[:, j+1, None]/W)
        s = (n[:, j]-n[:, j+1])/(n[:, j]+n[:, j+1])
        r = (s+q)/(1+s*q)
    q = r*np.exp(4j*np.pi*n[:, 0]*d[:, 0, None]/W)
    s = (1-n[:, 0])/(1+n[:, 0])
    return abs((s+q)/(1+s*q))**2


def features(a, b, d, kind='full'):
    N, L = a.shape
    if not 1 < L <= MAX or b.shape != a.shape or d.shape != a.shape:
        raise ValueError('Expected equally shaped N by L arrays, 2 <= L <= 24')
    basic = np.stack(((a-1.7)/.35, (d-.1225)/.0575, b/.05,
                     np.broadcast_to(np.arange(L)/(L-1), (N,L)),
                     np.broadcast_to(np.arange(L)/24, (N,L)), np.full((N,L), L/24)), -1)
    # Historical generic expansion operates on an already-float32 basic tensor.
    gb = basic.astype('float32')
    generic = np.zeros((N, MAX, 24))
    generic[:, :L] = np.concatenate((gb, gb**2, gb**3, gb*np.roll(gb, -1, axis=-1)), -1)
    if kind == 'generic':
        return generic.astype('float32')
    if kind not in ('full','no_phase','no_interface','no_phase_interface','no_global','no_dispersion'):
        raise ValueError(kind)
    bb = np.zeros_like(b) if kind == 'no_dispersion' else b
    wave = np.array([.45,.65,.85])
    nv = a[...,None]+bb[...,None]*((.65/wave)**2-1)
    ph = 2*np.pi*nv*d[...,None]/wave
    left = np.c_[np.ones(N), a[:,:-1]]; right = np.c_[a[:,1:], np.ones(N)]
    glob = [a.mean(1), a.std(1), d.mean(1), d.std(1), (a*d).sum(1)/3, bb.mean(1)]
    f = np.zeros((N, MAX, 24))
    f[:,:L] = np.concatenate((basic, nv/2, np.sin(ph), np.cos(ph),
        ((left-a)/(left+a))[...,None], ((a-right)/(a+right))[...,None],
        (a*d/.65)[...,None], np.stack(glob,-1)[:,None,:].repeat(L,axis=1)), -1)
    if kind == 'no_dispersion':
        f[:,:L,23] = b.mean(1)[:,None]
    ids = []
    if kind in ('no_phase','no_phase_interface'): ids += [9,10,11,12,13,14,17]
    if kind in ('no_interface','no_phase_interface'): ids += [15,16]
    if kind == 'no_global': ids += list(range(18,24))
    f[:,:,ids] = generic[:,:,ids]
    return f.astype('float32')


def feature_kind(arm):
    if arm in ('full_aug','full_mlp'): return 'full'
    if arm in ('no_phase_interface','no_global'): return arm
    if arm not in ARMS: raise ValueError(arm)
    return 'generic'


def query(a, b, delta):
    N,L = a.shape
    u = np.broadcast_to(delta[:,None,:]/.01, (N,L,delta.shape[1]))
    angle = u[...,None]*np.array([1,2,3])
    out = np.zeros((N,MAX,delta.shape[1],7))
    out[:,:L] = np.concatenate((u[...,None], np.sin(angle), np.cos(angle)-1), -1)
    return out.astype('float32')


def reverse_array(values, lengths):
    out = np.zeros_like(values)
    for i,L in enumerate(lengths): out[i,:L] = values[i,:L][::-1]
    return out


def torch_reverse(torch, values, lengths):
    j = torch.arange(values.shape[1], device=values.device)[None]
    ids = (lengths[:,None]-1-j).clamp(min=0)
    tail = [1]*(values.ndim-2)
    out = values.gather(1, ids.reshape(*ids.shape,*tail).expand_as(values))
    return out.masked_fill((j>=lengths[:,None]).reshape(*j.expand(len(values),-1).shape,*tail), 0.)


def positions(a, b, d, arm, unit):
    out = np.zeros((len(a),MAX), np.float32)
    if arm.startswith('prism_nm_rope'): out[:,:d.shape[1]] = np.cumsum(d,1)*1000
    elif arm.startswith('prism_scaled_rope'):
        if unit <= 0: raise ValueError('Position unit must be positive')
        out[:,:d.shape[1]] = np.cumsum(d,1)/unit
    return out


def make_model(arm, seed, device='cpu'):
    import torch
    if arm not in ARMS: raise ValueError(arm)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Linear(24,64)
            layer = torch.nn.TransformerEncoderLayer(64,4,128,dropout=0.,batch_first=True,norm_first=True)
            self.encoder = torch.nn.TransformerEncoder(layer,2,norm=torch.nn.LayerNorm(64),enable_nested_tensor=False)
            self.head = torch.nn.Sequential(torch.nn.Linear(71,64),torch.nn.GELU(),torch.nn.Linear(64,2))
        def rotate(self, x, pos):
            base = 1000. if arm.endswith('_b1000') else 10000.
            freq = base**(-torch.arange(0,x.shape[-1],2,device=x.device,dtype=x.dtype)/x.shape[-1])
            theta = pos[:,None,:,None]*freq
            a,b = x[...,0::2],x[...,1::2]
            return torch.stack((a*theta.cos()-b*theta.sin(),a*theta.sin()+b*theta.cos()),-1).flatten(-2)
        def forward(self, x, q, qzero, mask, pos=None):
            z = self.input(x)
            if 'rope' in arm:
                if pos is None: raise ValueError('RoPE requires positions')
                for layer in self.encoder.layers:
                    u = layer.norm1(z); B,L,_ = u.shape
                    qkv = torch.nn.functional.linear(u,layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias)
                    qq,kk,vv = qkv.reshape(B,L,3,4,16).permute(2,0,3,1,4).unbind(0)
                    att = torch.nn.functional.scaled_dot_product_attention(self.rotate(qq,pos),self.rotate(kk,pos),vv,attn_mask=mask[:,None,None,:],dropout_p=0.)
                    out = layer.self_attn.out_proj(att.transpose(1,2).reshape(B,L,64))
                    z = z+layer.dropout1(out)
                    z = z+layer._ff_block(layer.norm2(z))
                z = self.encoder.norm(z)
            else: z = self.encoder(z,src_key_padding_mask=~mask)
            B,L,Q,_ = q.shape
            full = self.head(torch.cat((z[:,:,None,:].expand(B,L,Q,64),q),-1))
            zero = self.head(torch.cat((z,qzero.expand(B,L,7)),-1))[:,:,None,:]
            return (full-zero).masked_fill(~mask[:,:,None,None],0.)
    torch.manual_seed(20261017701+seed)
    net = Net()
    if arm.endswith('_mlp'):
        original = net.input
        net.input = torch.nn.Sequential(torch.nn.Linear(24,64),torch.nn.GELU(),torch.nn.Linear(64,64))
        net.input[0].load_state_dict(original.state_dict())
    return net.to(device)


def metrics(y, pred, lengths):
    n,l,q,t = y.shape
    valid = np.arange(l)[None] < lengths[:,None]
    truth = np.concatenate((np.zeros((n,1,t)),y.reshape(n,l*q,t)),1)
    scores = np.concatenate((np.zeros((n,1,t)),pred.reshape(n,l*q,t)),1)
    av = np.concatenate((np.ones((n,1),bool),np.repeat(valid,q,1)),1)
    choice = np.where(av[:,:,None],scores,-np.inf).argmax(1)
    best = np.where(av[:,:,None],truth,-np.inf).max(1)
    selected = np.take_along_axis(truth,choice[:,None,:],1)[:,0]
    layer = np.maximum(choice-1,0)//q
    layer_best = np.take_along_axis(y.max(2),layer[:,None,:],1)[:,0]
    layer_best = np.where(choice==0,0.,layer_best)
    return dict(regret=best-selected,layer_regret=best-layer_best,within_layer_regret=layer_best-selected,
                choice=choice,selected_layer=np.where(choice==0,-1,layer),selected_query=np.where(choice==0,-1,(choice-1)%q),
                noop=choice==0,gain_rmse=np.array(np.sqrt((((pred-y)**2)*valid[:,:,None,None]).sum()/(valid.sum()*q*t))))

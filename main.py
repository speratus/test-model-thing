import torch
import torch.nn as nn
import torch.optim as opt

class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def forward(self, x: torch.Tensor):
        return self.embed(x)

class Decoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        return self.decode(x), torch.sigmoid(self.stop(x))

class Layer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        
        self.decay = nn.Parameter(torch.zeros((dim, )))
        self.register_buffer("states", torch.zeros((dim, )))

        self.register_buffer("decaytrace", torch.zeros((dim, )))
        self.register_buffer("embedtrace", torch.zeros((256, dim)))
        
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias = False)
        self.silu = nn.SiLU()

    def forward(self, enc: torch.Tensor, x: torch.Tensor, dummy: torch.Tensor):
        decay = torch.sigmoid(self.decay)
        state = (decay * self.states) + enc + dummy

        return x + self.silu(self.weights(self.norm(state))), state, decay

class Model(nn.Module):
    def __init__(self, dim: int, layers: int, temp: float, lr: float, device: torch.device | str | None = None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dim = dim
        self.layercount = layers
        self.temp = temp
        self.lr = lr

        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)

        self.layers = nn.ModuleList([Layer(dim) for _ in range(layers)])
        self.to(self.device)
        self.optimizer = opt.AdamW(self.parameters(), lr = lr)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.train()

    def sample(self, output: torch.Tensor) -> int:
        probs = torch.softmax(output, dim = -1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)) / torch.log(torch.tensor(256.0, device = output.device))

        temp = max(0.1, float(self.temp * (1.0 - self.temp * entropy)))
        logits = output / temp
        return int(torch.distributions.Categorical(logits = logits).sample().item())

    def reset(self):
        for layer in self.layers:
            layer.decay.data.zero_()
            layer.states.zero_()

            layer.decaytrace.zero_()
            layer.embedtrace.zero_()

    def step(self, c: torch.Tensor, dummies: list[torch.Tensor]):
        enc = self.encoder(c)
        x = enc
            
        states, decays = [], []

        for i, layer in enumerate(self.layers):
            x, state, decay = layer(enc, x, dummies[i])

            states.append(state)
            decays.append(decay)

        return (x, states, decays), self.decoder(x)

    def __call__(self, currb: int, nextb: int | None, end: bool, notrace: bool = False):
        c = torch.tensor(currb, device = self.device, dtype = torch.long)

        if notrace:
            with torch.no_grad():
                dummies = [torch.zeros((self.dim, ), device = self.device) for _ in range(self.layercount)]
                _, (output, stop) = self.step(c, dummies)
                return self.sample(output), stop.item()

        dummies = [torch.zeros((self.dim, ), device = self.device, requires_grad = True) for _ in range(self.layercount)]
        (x, states, decays), (output, stop) = self.step(c, dummies)

        loss = torch.relu(1.0 - torch.sqrt(torch.var(x, unbiased = False) + 1e-4)) # variance
        if nextb is not None:
            n = torch.tensor(nextb, device = self.device, dtype = torch.long)
            tgt = self.encoder(n).detach()

            loss = loss + torch.mean((x - tgt) ** 2) # pred mse
            loss = loss - output[n] + torch.logsumexp(output, dim = -1) # ce

            stop_target = torch.tensor([1.0 if end else 0.0], device = self.device)
            loss = loss + torch.mean((stop - stop_target) ** 2) # stop mse

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        all_grads = torch.autograd.grad(loss, trainable_params + dummies, retain_graph = False)

        param_grads = all_grads[:len(trainable_params)]
        dlds_s = all_grads[len(trainable_params):]

        self.optimizer.zero_grad()
        for p, g in zip(trainable_params, param_grads):
            p.grad = g.clone()

        one_hot_c = (torch.arange(256, device = self.device) == c).unsqueeze(1).float()

        for i, layer in enumerate(self.layers):
            dlds = dlds_s[i]
            d_i = decays[i].detach()

            embedtrace = (layer.embedtrace * d_i) + one_hot_c
            embed_grad_corr = dlds.unsqueeze(0) * (layer.embedtrace * d_i)
            self.encoder.embed.weight.grad.add_(embed_grad_corr)
            
            decaytrace = (d_i * layer.decaytrace) + (d_i * (1.0 - d_i) * layer.states)
            layer.decay.grad = dlds * decaytrace

            layer.states.copy_(states[i].detach())

            layer.decaytrace.copy_(decaytrace.detach())
            layer.embedtrace.copy_(embedtrace.detach())

        self.optimizer.step()

        return self.sample(output.detach()), stop.detach().item()

    def save(self, path: str):
        import os
        from safetensors.torch import save_file

        data = {}
        for k, v in self.state_dict().items():
            data[f"m.{k}"] = v.contiguous().cpu()

        opt_state = self.optimizer.state_dict()
        for p_idx, s in opt_state.get('state', {}).items():
            for s_key, s_val in s.items():
                if isinstance(s_val, torch.Tensor):
                    data[f"o.{p_idx}.{s_key}"] = s_val.contiguous().cpu()
                elif isinstance(s_val, (int, float)):
                    data[f"o.{p_idx}.{s_key}"] = torch.tensor(s_val)

        for i, layer in enumerate(self.layers):
            data[f"state.{i}"] = layer.states.contiguous().cpu()
            data[f"decaytrace.{i}"] = layer.decaytrace.contiguous().cpu()
            data[f"embedtrace.{i}"] = layer.embedtrace.contiguous().cpu()

        tmp = 'temporary-' + path
        save_file(data, tmp)
        os.replace(tmp, path)

    def load(self, path: str):
        import os
        from safetensors.torch import load_file
        if not os.path.exists(path): return

        data = load_file(path, device = str(self.device))
        model_state = {}
        opt_state_tensors = {}
        
        for k, v in data.items():
            if k.startswith("m."):
                model_state[k[2:]] = v
            elif k.startswith("o."):
                parts = k[2:].split('.')
                if len(parts) == 2:
                    p_idx = int(parts[0])
                    s_key = parts[1]
                    if p_idx not in opt_state_tensors:
                        opt_state_tensors[p_idx] = {}
                    opt_state_tensors[p_idx][s_key] = v
            elif k.startswith("state."):
                idx = int(k.split('.')[1])
                if idx < len(self.layers): self.layers[idx].states.copy_(v)
            elif k.startswith("decaytrace."):
                idx = int(k.split('.')[1])
                if idx < len(self.layers): self.layers[idx].decaytrace.copy_(v)
            elif k.startswith("embedtrace."):
                idx = int(k.split('.')[1])
                if idx < len(self.layers): self.layers[idx].embedtrace.copy_(v)
            
        if model_state:
            curr_state = self.state_dict()
            filtered_state = {k: v for k, v in model_state.items() if k in curr_state and curr_state[k].shape == v.shape}
            self.load_state_dict(filtered_state, strict = False)

        if opt_state_tensors:
            try:
                opt_sd = self.optimizer.state_dict()
                opt_sd['state'] = opt_state_tensors
                self.optimizer.load_state_dict(opt_sd)
            except Exception:
                pass

class Runtime:
    def __init__(self, path: str, threshold: float, **kwargs):
        self.model = Model(**kwargs)
        device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.model.to(device)

        self.path = path
        self.threshold = threshold

        self.step = 0
        self.prevtime = None

    def save(self):
        self.step += 1
        if self.step % 500 == 0: self.model.save(self.path)

    def call(self, c: int, n: int | None, end: bool, readonly: bool = False, notrace: bool = False):
        outputs = self.model(c, n, end, notrace)
        if not readonly: self.save()
        return outputs

    def write(self, b: int):
        import sys
        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def chat(self, readonly: bool = False, notrace: bool = False):
        import itertools, time

        while True:
            text = input(f'\n[{self.now()} | {0 if self.prevtime is None else time.time() - self.prevtime:.4f}s]\nUser >> ')
            self.prevtime = time.time()

            data = (text + '\n').encode('utf-8')
            
            for i, (c, n) in enumerate(itertools.pairwise(data)):
                b, _ = self.call(c, n, i == len(data) - 2, readonly, notrace)

            print(f'\n[{self.now()}]\nModel >> ', end = '', flush = True)

            b = data[-1]
            while True:
                b, stop = self.call(b, None, False, readonly, notrace)
                self.write(b)
                if stop > self.threshold:
                    print()
                    break

    def dataset(self):
        import glob, itertools
        files = ['cleaned_merged_fairy_tales_without_eos.txt']

        while True:
            for file in files:
                with open(file, 'r', encoding = 'utf-8', errors = 'ignore') as f:
                    for line in f:
                        data = line.encode('utf-8')
                        for i, (c, n) in enumerate(itertools.pairwise(data)):
                            b, _ = self.call(c, n, i == len(data) - 2)
                            self.write(b)

    def now(self):
        from datetime import datetime
        return datetime.now().strftime('%d/%m/%Y, %H:%M:%S')

    def __call__(self):
        modes = ['train', 'chat', 'chatreadonly', 'chatnotrace']

        try: mode = modes.index(input(f'\nthe \'chatnotrace\' mode is there for bug testing. \'chatreadonly\' is for chatting without overriding weights.\n[{self.now()}]\nmode: {modes} >> ').lower())
        except ValueError:
            print('\nInvalid mode.')
            return

        self.model.load(self.path)
        print()

        try:
            match mode:
                case 0: self.dataset()
                case 1: self.chat()
                case 2: self.chat(readonly = True)
                case 3: self.chat(readonly = True, notrace = True)
        except KeyboardInterrupt:
            print('\nInterrupted.')
        finally:
            if mode < 2: self.model.save(self.path)

if __name__ == '__main__':
    # Runtime(path = 'larger-130m.safetensors', threshold = 0.35, dim = 2048, layers = 32, temp = 0.75, lr = 5e-4)()
    Runtime(path = 'experimental-4.5m.safetensors', threshold = 0.35, dim = 512, layers = 16, temp = 0.75, lr = 5e-4)()
    # param count = (256 * dim) + (dim * dim + dim * 2 + dim) + (256 * dim + dim + 1)

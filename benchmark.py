import torch
import torch.nn as nn
import torch.optim as opt

from main import Model

class Classification(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor):
        return self.proj(x)

def cola(filepath: str):
    data = []

    try:
        with open(filepath, 'r', encoding = 'utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4: data.append((parts[3].encode('utf-8'), int(parts[1])))

    except FileNotFoundError: pass
    
    return data

def mcc(tp, tn, fp, fn):
    import math
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    score = (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0
    return score * 100

def run(path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = Model(dim = 512, layers = 16, temp = 0.75, lr = 5e-4, device = device)
    model.load(path)
    model.freeze()

    head = Classification(model.dim).to(device)
    headopt = opt.AdamW(head.parameters(), lr = 1e-3)

    data = cola('CoLA/original/raw/in_domain_train.tsv')

    if data == []:
        print('invalid CoLA dataset.')
        return

    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        print(f'\nEpoch {epoch + 1}')

        dummies = [torch.zeros((model.dim, ), device = device) for _ in range(model.layercount)]
        tp, tn, fp, fn = 0, 0, 0, 0
        
        for i, (b_s, label) in enumerate(data):
            model.reset()

            final = None
            with torch.no_grad():
                for b in b_s:
                    enc = model.encoder(torch.tensor(b, device = device, dtype = torch.long))
                    x = enc

                    for j, layer in enumerate(model.layers):
                        x, state, _ = layer(enc, x, dummies[j])
                        layer.states.copy_(state)

                    final = model.layers[-1].states

            headopt.zero_grad()
            choice = head(final)
            target = torch.tensor([label], device = device, dtype = torch.long)
            loss = criterion(choice.unsqueeze(0), target)
            loss.backward()
            headopt.step()

            predicted_class = torch.argmax(choice).item()
            if predicted_class == 1 and label == 1: tp += 1
            elif predicted_class == 0 and label == 0: tn += 1
            elif predicted_class == 1 and label == 0: fp += 1
            elif predicted_class == 0 and label == 1: fn += 1

            score = mcc(tp, tn, fp, fn)

            if i > 0 and i % 500 == 0: print(f'{i}: T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({score:.4f})')

        print(f'{i}: T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({score:.4f})')

if __name__ == '__main__':
    run('experimental-4.5m.safetensors')
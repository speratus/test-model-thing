import os
import tempfile
import torch
import torch.nn as nn
from main import Encoder, Decoder, Layer, Model
from benchmark import Classification, mcc

def test_encoder():
    dim = 64
    encoder = Encoder(dim)
    x = torch.tensor(65, dtype=torch.long)
    out = encoder(x)
    assert out.shape == (dim,)
    assert not torch.isnan(out).any()

def test_decoder():
    dim = 64
    decoder = Decoder(dim)
    x = torch.randn(dim)
    logits, stop = decoder(x)
    assert logits.shape == (256,)
    assert stop.shape == (1,)
    assert 0.0 <= stop.item() <= 1.0

def test_layer():
    dim = 64
    layer = Layer(dim)
    enc = torch.randn(dim)
    x = torch.randn(dim)
    dummy = torch.zeros(dim)
    out, state, decay = layer(enc, x, dummy)
    assert out.shape == (dim,)
    assert state.shape == (dim,)
    assert decay.shape == (dim,)
    assert (decay >= 0.0).all() and (decay <= 1.0).all()

def test_model_step_and_train():
    dim = 64
    layers = 2
    model = Model(dim=dim, layers=layers, temp=0.75, lr=1e-3, device="cpu")
    
    # Test step
    dummies = [torch.zeros(dim) for _ in range(layers)]
    (x, states, decays), (logits, stop) = model.step(torch.tensor(65, dtype=torch.long), dummies)
    assert x.shape == (dim,)
    assert len(states) == layers
    assert len(decays) == layers
    assert logits.shape == (256,)
    assert stop.shape == (1,)

    # Test training step (forward + backward + trace update)
    b_out, stop_val = model(currb=65, nextb=66, end=False)
    assert isinstance(b_out, int)
    assert 0 <= b_out < 256
    assert isinstance(stop_val, float)

    # Test notrace inference
    b_out_nt, stop_val_nt = model(currb=65, nextb=None, end=False, notrace=True)
    assert 0 <= b_out_nt < 256
    assert isinstance(stop_val_nt, float)

def test_model_reset():
    dim = 64
    model = Model(dim=dim, layers=2, temp=0.75, lr=1e-3, device="cpu")
    model(currb=65, nextb=66, end=False)
    assert not torch.allclose(model.layers[0].states, torch.zeros(dim))
    model.reset()
    assert torch.allclose(model.layers[0].states, torch.zeros(dim))
    assert torch.allclose(model.layers[0].decaytrace, torch.zeros(dim))
    assert torch.allclose(model.layers[0].embedtrace, torch.zeros(256, dim))

def test_safetensors_save_load():
    dim = 64
    model = Model(dim=dim, layers=2, temp=0.75, lr=1e-3, device="cpu")
    model(currb=65, nextb=66, end=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.safetensors")
        model.save(path)
        assert os.path.exists(path)

        model2 = Model(dim=dim, layers=2, temp=0.75, lr=1e-3, device="cpu")
        model2.load(path)

        # Check parameter equality
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert k1 == k2
            assert torch.allclose(v1, v2)

def test_benchmark_components():
    dim = 64
    head = Classification(dim)
    x = torch.randn(dim)
    logits = head(x)
    assert logits.shape == (2,)

    # Test MCC score calculation
    score = mcc(tp=10, tn=10, fp=2, fn=2)
    assert isinstance(score, float)
    assert -100.0 <= score <= 100.0

if __name__ == '__main__':
    test_encoder()
    print("✓ test_encoder passed")
    test_decoder()
    print("✓ test_decoder passed")
    test_layer()
    print("✓ test_layer passed")
    test_model_step_and_train()
    print("✓ test_model_step_and_train passed")
    test_model_reset()
    print("✓ test_model_reset passed")
    test_safetensors_save_load()
    print("✓ test_safetensors_save_load passed")
    test_benchmark_components()
    print("✓ test_benchmark_components passed")
    print("All unit tests passed successfully!")

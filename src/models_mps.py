"""Arquitecturas de clasificación con un backend PyTorch compartido.

La CNN conserva las capas de la referencia Keras. El padding explícito replica
SAME_UPPER para las dimensiones intermedias fijas de una entrada de 224 píxeles.
"""
import numpy as np
import torch
from torch import nn


class ReferenceNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        input_channels = 3
        specifications = (
            (32, 5, (1, 2, 1, 2)),
            (64, 5, (1, 2, 1, 2)),
            (128, 3, (0, 1, 0, 1)),
            (256, 3, (1, 1, 1, 1)),
        )
        for filters, kernel, padding in specifications:
            self.blocks.append(nn.Sequential(
                nn.ZeroPad2d(padding),
                nn.Conv2d(input_channels, filters, kernel, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ))
            input_channels = filters
        self.hidden = nn.Linear(256, 128)
        self.output = nn.Linear(128, 4)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        features = inputs
        for block in self.blocks:
            features = block(features)
        return self.output(self.hidden(torch.flatten(features, 1)))


class SharedTorchClassifier:
    def __init__(self, seed, architecture):
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.hardware = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.hardware = 'Apple M3 GPU via MPS'
        else:
            self.device = torch.device('cpu')
            self.hardware = 'CPU'
        self.architecture = architecture
        if architecture == 'cnn':
            self.model = ReferenceNetwork()
        else:
            from ultralytics.nn.tasks import ClassificationModel
            self.model = ClassificationModel('yolo26n-cls.yaml', nc=4, verbose=False)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, eps=1e-7)
        self.loss = nn.CrossEntropyLoss()
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())

    def tensor(self, images):
        array = np.array(images, copy=True)
        return torch.from_numpy(array).to(self.device).permute(0, 3, 1, 2).contiguous().float() / 255

    def train(self, images, labels):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(self.tensor(images))
        targets = torch.as_tensor(labels, device=self.device, dtype=torch.long)
        loss = self.loss(logits, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def predict(self, images):
        self.model.eval()
        with torch.inference_mode():
            output = self.model(self.tensor(images))
            probabilities = output.softmax(dim=1) if self.architecture == 'cnn' else output[0]
        return probabilities.cpu().numpy()

    def save(self, path):
        temporary = path / 'best.pt.incomplete'
        torch.save(self.model.state_dict(), temporary)
        temporary.replace(path / 'best.pt')

    def load(self, path):
        weights = torch.load(path / 'best.pt', map_location=self.device, weights_only=True)
        self.model.load_state_dict(weights)

    def gradcam(self, image, label):
        self.model.eval()
        inputs = self.tensor(image[None])
        features = inputs
        for block in self.model.blocks[:2]:
            features = block(features)
        third = self.model.blocks[2]
        activation = third[2](third[1](third[0](features)))
        activation.retain_grad()
        features = third[3](activation)
        features = self.model.blocks[3](features)
        logits = self.model.output(self.model.hidden(torch.flatten(features, 1)))
        probabilities = logits.softmax(dim=1)
        self.model.zero_grad(set_to_none=True)
        probabilities[0, label].backward()
        weights = activation.grad[0].mean(dim=(1, 2))
        heatmap = torch.relu((activation[0] * weights[:, None, None]).sum(dim=0))
        maximum = heatmap.max()
        if maximum.item() > 0:
            heatmap = heatmap / maximum
        return heatmap.detach().cpu().numpy()


class ReferenceCNN(SharedTorchClassifier):
    def __init__(self, seed):
        super().__init__(seed, 'cnn')


class YOLOClassifier(SharedTorchClassifier):
    def __init__(self, seed):
        super().__init__(seed, 'yolo')

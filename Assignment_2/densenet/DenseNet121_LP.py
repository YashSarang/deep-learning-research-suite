import timm
import torch.nn as nn

class DenseNet121_LP(nn.Module):
    def __init__(self, num_classes, enable_probe=False):
        super().__init__()

        self.enable_probe = enable_probe

        # Load pretrained DenseNet-121 backbone
        self.backbone = timm.create_model(
            "densenet121",
            pretrained=True,
            num_classes=0  # removes classifier
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # DenseNet-121 feature dimension = 1024
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)
        self.probe_features = {}

        if self.enable_probe:
            self._register_hooks()
    
    def _register_hooks(self):
        # Hook into DenseNet blocks instead of ResNet layers
        self.backbone.features.denseblock1.register_forward_hook(self._hook("block1"))
        self.backbone.features.denseblock2.register_forward_hook(self._hook("block2"))
        self.backbone.features.denseblock3.register_forward_hook(self._hook("block3"))
        self.backbone.features.denseblock4.register_forward_hook(self._hook("block4"))

    def _hook(self, name):
        def fn(module, input, output):
            self.probe_features[name] = output.detach()
        return fn

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features
    
    def forward_probe(self, x):
        self.probe_features = {}
        _ = self.backbone(x)
        return self.probe_features
    
class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
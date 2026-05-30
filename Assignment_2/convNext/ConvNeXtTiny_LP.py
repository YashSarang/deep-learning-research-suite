import timm
import torch.nn as nn

class ConvNeXtTiny_LP(nn.Module):
    def __init__(self, num_classes, enable_probe=False):
        super().__init__()

        self.enable_probe = enable_probe

        # Load pretrained backbone from Huggingface/timm
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            num_classes=0  # removes classifier
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Feature dimension (ConvNeXt-Tiny final dim = 768)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)
        self.probe_features = {}

        if self.enable_probe:
            self._register_hooks()
    
    def _register_hooks(self):
        # ConvNeXt uses stages[0] through stages[3] instead of layer1 through layer4
        self.backbone.stages[0].register_forward_hook(self._hook("stages.0"))
        self.backbone.stages[2].register_forward_hook(self._hook("stages.2"))
        self.backbone.stages[3].register_forward_hook(self._hook("stages.3"))

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
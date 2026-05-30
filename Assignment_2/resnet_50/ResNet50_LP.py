import timm
import torch.nn as nn

'''
Class name
    ResNet50_LP
Description
    Pre-trained ResNet-50 model taken from 
'''
class ResNet50_LP(nn.Module):
    def __init__(self, num_classes, enable_probe=False):
        super().__init__()

        self.enable_probe = enable_probe

        # Load pretrained backbone
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=0  # removes classifier
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Feature dimension (ResNet50 = 2048)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)
        self.probe_features = {}

        if self.enable_probe:
            self._register_hooks()
    
    def _register_hooks(self):
        self.backbone.layer1.register_forward_hook(self._hook("layer1"))
        self.backbone.layer3.register_forward_hook(self._hook("layer3"))
        self.backbone.layer4.register_forward_hook(self._hook("layer4"))

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
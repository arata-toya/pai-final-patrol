from torch import nn

LABELS=['normal','stain','crack']

def make_model():
    return nn.Sequential(
        nn.Conv2d(1,12,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(24,48,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),
        nn.Linear(48*4*4,64),nn.ReLU(),nn.Linear(64,3))

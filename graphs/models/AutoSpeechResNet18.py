from graphs.models.AutoSpeechResNet import *


class AutoSpeechResNet18(ResNet):
    def __init__(self, device, num_classes, loss='xent', pretrained=True, **kwargs):
        super(AutoSpeechResNet18, self).__init__(
            device,
            num_classes=num_classes,
            loss=loss,
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            last_stride=2,
            fc_dims=None,
            dropout_p=None,
            **kwargs
        )

        if pretrained:
            init_pretrained_weights(self, model_urls['resnet18'])

from graphs.models.AutoSpeechResNet import *


class AutoSpeechResNet34(ResNet):
    def __init__(self, device, num_classes, loss='xent', pretrained=True, **kwargs):
        super(AutoSpeechResNet34, self).__init__(
            device,
            num_classes=num_classes,
            loss=loss,
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            last_stride=2,
            fc_dims=None,
            dropout_p=None,
            **kwargs
        )

        if pretrained:
            init_pretrained_weights(self, model_urls['resnet34'])

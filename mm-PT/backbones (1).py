# Used to specify and instantiate the backbone that's going to be used for training
import timm
from huggingface_hub import login


def get_backbone(backbone_name: str, architecture: str, num_classes: int, pretrained: bool):
    access_token = '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'
    login(access_token)
    match backbone_name:
        case 'resnet18':
            backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)
            num_features = backbone.get_classifier().in_features
            return backbone, num_features

        case 'dino':
            print(backbone_name)
            print(architecture)
            backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)
            num_features = backbone.get_classifier().in_features
            return backbone, num_features

        case 'uni':
            backbone = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5,
                                              dynamic_img_size=True, num_classes=num_classes)
            num_features = backbone.get_classifier().in_features
            return backbone, num_features

        case 'dinov2':
            backbone = timm.create_model(architecture, pretrained=True, num_classes=num_classes)
            num_features = backbone.get_classifier().in_features
            return backbone, num_features
        case 'prov':
            backbone = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, num_classes=num_classes)
            num_features = backbone.get_classifier().in_features
            return backbone, num_features
    raise ValueError(f"backbone_name {backbone_name} did not match any of the expected architectures")

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import transformers.image_transforms
from torch import nn as nn
from torchvision.models import VGG19_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights
import timm, torchmetrics
from transformers import AutoImageProcessor, Swinv2ForImageClassification
import torch.nn.init as init


# Load pre-trained VGG19 model


class RexNex150(nn.Module):
    def __init__(self, num_classes):
        super(RexNex150, self).__init__()
        self.model = timm.create_model("rexnet_150", pretrained=True, num_classes=num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


class RexNetTransfer(nn.Module):
    def __init__(self, num_classes):
        super(RexNetTransfer, self).__init__()
        self.feature_extractor = timm.create_model("rexnet_150", pretrained=True, features_only=True)

        # Assume output sizes as provided and add adaptive pooling layers for each
        self.adaptive_pool1 = nn.AdaptiveAvgPool2d((1, 1))  # for layer 1
        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((1, 1))  # for layer 2
        self.adaptive_pool3 = nn.AdaptiveAvgPool2d((1, 1))  # for layer 3
        self.adaptive_pool4 = nn.AdaptiveAvgPool2d((1, 1))  # for layer 4
        self.adaptive_pool5 = nn.AdaptiveAvgPool2d((1, 1))  # for layer 5

        # Combine the features from all layers after pooling
        total_feature_size = 24 + 58 + 92 + 193 + 277
        self.fc1 = nn.Linear(total_feature_size, total_feature_size)
        self.fc2 = nn.Linear(total_feature_size, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)

        # Pool each layer's features to size (1, 1) and flatten
        f1 = self.adaptive_pool1(features[0]).view(x.size(0), -1)
        f2 = self.adaptive_pool2(features[1]).view(x.size(0), -1)
        f3 = self.adaptive_pool3(features[2]).view(x.size(0), -1)
        f4 = self.adaptive_pool4(features[3]).view(x.size(0), -1)
        f5 = self.adaptive_pool5(features[4]).view(x.size(0), -1)

        # Concatenate all features
        x = torch.cat([f1, f2, f3, f4, f5], dim=1)

        # Further processing with fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define your model architecture
class VGGTransfer(nn.Module):
    vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT)

    def __init__(self, num_classes):
        super(VGGTransfer, self).__init__()
        # Feature extractor (VGG19)
        self.feature_extractor = self.vgg19.features
        # Fully connected layers for further processing
        self.fc1 = nn.Linear(512 * 7 * 7, 512 * 7)
        self.fc2 = nn.Linear(512 * 7, 512 * 7)
        self.fc3 = nn.Linear(512 * 7, num_classes)

    def forward(self, x):
        # Extract features using VGG19
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        # Further processing with fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNScratch(nn.Module):

    def __init__(self, num_classes):
        super(CNNScratch, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 800 * 600, num_classes)  # Adjusted output size for 5 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = x.view(-1, 16 * 800 * 600)  # Adjusted the input size after the convolutional layer
        x = self.fc1(x)
        return x


class TestCNN(nn.Module):
    def __init__(self, num_classes, input_size=4):
        super(TestCNN, self).__init__()
        self.f1 = nn.Linear(input_size, 4)
        self.f2 = nn.Linear(4, num_classes)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        return x


class CombinedVGGResNet(nn.Module):
    def __init__(self, num_classes):
        super(CombinedVGGResNet, self).__init__()
        self.vgg = VGGTransfer(num_classes)
        self.resnet = RexNex150(num_classes)
        self.final = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        x1 = self.vgg(x)
        x2 = self.resnet(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.final(x)
        return x


class CombinedVGGEfNet(nn.Module):
    def __init__(self, num_classes):
        super(CombinedVGGEfNet, self).__init__()
        self.transform_224 = transforms.Resize((224, 224))
        self.vgg = VGGTransfer(num_classes)
        self.EfNet = EfficientNetV2L(num_classes)

        self.final = nn.Linear(num_classes * 2, num_classes, bias=False)
        init.constant_(self.final.weight, 0.5)

    def forward(self, x):
        vggx = self.transform_224(x)
        vggx = self.vgg(vggx).softmax(dim=1)
        EfNet = self.EfNet(x).softmax(dim=1)
        x = torch.cat((vggx, EfNet), dim=1)
        x = self.final(x)
        return x

    def load_model(self):
        self.vgg.load_state_dict(torch.load('VGG.pth'))
        self.EfNet.load_state_dict(torch.load('EfficientNetV2L.pth'))

    def only_train_final_layer(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.final.parameters():
            param.requires_grad = True


class VGG19Modified(nn.Module):
    vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT)

    def __init__(self, num_classes):
        super(VGG19Modified, self).__init__()
        # Feature extractor (VGG19)
        self.adjustment = nn.Sequential(
            # 1st conv block
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),  # Output: (3, 448, 448)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.feature_extractor = self.vgg19.features
        # Fully connected layers for further processing
        self.fc1 = nn.Linear(512 * 7 * 7, 512 * 7)
        self.fc2 = nn.Linear(512 * 7, 512 * 7)
        self.fc3 = nn.Linear(512 * 7, num_classes)

    def forward(self, x):
        # Extract features using VGG19
        x = self.adjustment(x)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        # Further processing with fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SwimTrans(nn.Module):
    def __init__(self, num_classes):
        super(SwimTrans, self).__init__()
        self.model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.model.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.model(x).logits


class EfficientNetV2L(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetV2L, self).__init__()
        self.model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)

        # Replace the classifier layer to match the number of classes
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class SwinEfficientNetComb(nn.Module):
    def __init__(self, num_classes):
        super(SwinEfficientNetComb, self).__init__()
        self.transform_224 = transforms.Resize((224, 224))
        self.EfNet = EfficientNetV2L(num_classes=num_classes)
        self.swin = SwimTrans(num_classes=num_classes)

        self.final_0 = nn.Linear(num_classes * 2, num_classes, bias=False)
        self.final_1 = nn.Linear(num_classes, num_classes)
        init.constant_(self.final_0.weight, 0.5)
        init.constant_(self.final_1.weight, 0.5)


    def load_model(self):
        self.swin.load_state_dict(torch.load('Swin.pth'))
        self.EfNet.load_state_dict(torch.load('EfficientNetV2L.pth'))

    def forward(self, x):
        x_224 = self.transform_224(x)
        x_swin = self.swin(x_224).softmax(dim=1)
        x_efnet = self.EfNet(x).softmax(dim=1)
        x = torch.cat((x_swin, x_efnet), dim=1)
        return self.final_0(x)

    def only_train_final_layer(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.final_0.parameters():
            param.requires_grad = True
        for param in self.final_1.parameters():
            param.requires_grad = True

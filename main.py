from torchvision import transforms
from DataLoader import DataLoader
from ModelRunner import LearningModel

cut_transform_224 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4286, 0.4982, 0.3136], std=[0.2363, 0.2394, 0.2277]),
])

cut_transform_224_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4286, 0.4982, 0.3136], std=[0.2363, 0.2394, 0.2277]),
])

cut_transform_448 = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4286, 0.4982, 0.3136], std=[0.2363, 0.2394, 0.2277]),
])

cut_transform_480 = transforms.Compose([
    transforms.Resize((480, 480)),  # Resize the image to the required input size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.4286, 0.4982, 0.3136], std=[0.2363, 0.2394, 0.2277]),
])

normal_transformer = transforms.Compose([
    transforms.ToTensor()
])

save_model_location = 'saved_models/'


def train_model(model, epoch, batch_size, weights=None, sampled=False, save_model=None, transformer=normal_transformer):
    data = DataLoader(transformer, generate_new=False, replace_dataset=True)
    if sampled:
        training_data, validation_data, test_data = data.load_data_with_sampler(batch_size=batch_size)
    else:
        training_data, validation_data, test_data = data.load_data(batch_size=batch_size)
    normal_training, normal_validation, normal_test = data.load_data(batch_size=batch_size)

    model = LearningModel(training_data, model=model, number_of_classes=5, epochs=1, class_weights=weights)
    model.model.load_model()
    model.model.only_train_final_layer()
    for i in range(epoch):
        model.train()
        print(model.calculate_global_metrics(normal_validation))
    model.evaluate(normal_test)
    print('Top 2 Global Accuracy: ', model.evaluate_top_k_accuracy(test_data, 2))
    if save_model is not None:
        model.save_model(save_model_location + save_model)
        print('ModelSaved')
    return model.loss_values


def test_model(model_name, transform):
    print('Testing Model: ', model_name)
    data = DataLoader(transform=transform, generate_new=False, replace_dataset=True)

    training_data, _, _ = data.load_data_with_sampler(batch_size=7)
    normal_training, normal_validation, normal_test = data.load_data(batch_size=5)
    model = LearningModel(training_data, model=model_name, number_of_classes=5, epochs=1)
    model.load_model(model_name + '.pth')
    model.model.eval()

    model.evaluate(normal_test)
    print(model.calculate_global_metrics(normal_test))
    print('Top 2 Global Accuracy: ', model.evaluate_top_k_accuracy(normal_test, 2))


def train_rex_net():
    output = train_model('CombinedVGGResNet', 30, 32, None, True, "CombinedVGGResNet.pth", cut_transform_224)


def train_swin():
    output = train_model('Swin', 35, 26, None, True, "Swin.pth", transformer=cut_transform_224)


def train_rex_net():
    output = train_model('ResNet', 10, 32, None, True, "ResNet.pth", cut_transform_224)


def train_Efnet_Swin():
    output = train_model('SwinEfficientNetComb', 20, 30, None, True, "SwinEfficientNetComb.pth", cut_transform_480)


def train_VGG_EfNet():
    output = train_model('CombinedVGGEfNet', 5, 5, None, True, "CombinedVGGEfNet.pth", cut_transform_480)


def train_vgg_mod():
    output = train_model('VGG', 10, 60, None, True, "VGG.pth", cut_transform_224)


def train_EfficientNetV2L():
    output = train_model('EfficientNetV2L', 20, 10, None, True, "Test.pth", cut_transform_480)


if __name__ == '__main__':
    test_model('SwinEfficientNetComb', transform=cut_transform_480)
    test_model('VGG', transform=cut_transform_224_test)
    test_model('ResNet', transform=cut_transform_224_test)
    test_model('Swin', transform=cut_transform_224_test)
    test_model('EfficientNetV2L', transform=cut_transform_480)
    test_model('CombinedVGGEfNet', transform=cut_transform_480)

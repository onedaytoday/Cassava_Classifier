import numpy as np
import os

import torch.nn.functional
from PIL import Image

from torchvision.transforms import transforms

from ModelRunner import LearningModel


class PlantDiseaseClassification:
    model_name = 'ResNet'
    model_location = 'ResNet.pth'
    number_of_classes = 5
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4286, 0.4982, 0.3136], std=[0.2363, 0.2394, 0.2277]),
    ])

    transform_224 = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4286, 0.4982, 0.3136], std=[0.2363, 0.2394, 0.2277]),
    ])

    def __init__(self):
        print("Welcome to the Plant Disease Detection application.")
        self.class_names = ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)",
                            "Cassava Green Mottle (CGM)", "Cassava Mosaic Disease (CMD)", "Healthy"]
        self.model = self.load_model()

    def load_model(self):
        model = LearningModel(None, self.model_name, number_of_classes=self.number_of_classes)
        model.load_model(self.model_location)
        return model

    @staticmethod
    def is_picture_file(file_path):
        if not os.path.isfile(file_path):
            return False
        # Define a tuple of acceptable picture file extensions
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        # Check if the file extension is in the list of valid extensions
        return file_path.lower().endswith(valid_extensions)

    def _classify_image(self, image_name):
        image = Image.open(image_name)
        image = self.transform(image)
        image = torch.unsqueeze(image, 0)
        output = self.model.predict(image)
        return torch.softmax(output, dim=1)

    def classify_image(self, image_name):
        # List of class names corresponding to each possible plant disease
        print("Evaluating Picture: ", image_name)

        if PlantDiseaseClassification.is_picture_file(image_name):
            # Call the evaluate_picture function which simulates the model prediction
            probabilities = self._classify_image(image_name).squeeze(0).cpu()
            # Display the probabilities for each class
            print("\nProbabilities for each class:")
            for name, prob in zip(self.class_names, probabilities):
                print(f"{name}: {prob:.2%}")

            # Determine the class with the highest probability
            predicted_class = torch.argmax(probabilities)
            output = self.class_names[predicted_class]
            print(f"\nThe model has classified the picture as: {self.class_names[predicted_class]}")
        else:
            print("File does not exist or is not a valid picture file. Please enter a correct file name.")
            raise Exception()

        return output

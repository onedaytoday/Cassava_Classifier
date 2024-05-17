import os
import shutil
import torch.utils.data
from torch.utils.data import DataLoader as DL, random_split, RandomSampler, WeightedRandomSampler

import pandas as pd
from torchvision import datasets, transforms

import os
import shutil
from torchvision import datasets
from torch.utils.data import random_split


class DataLoader:
    def __init__(self, transform, shuffle=False, train_split=[0.7, .2, .1], generate_new=False, replace_dataset=False):
        self.train_split = train_split
        self.training_data_set_location = 'images'
        self.training_data_set_labels_location = 'train.csv'
        self.test_data_set_location = ''
        self.save_dataset_location = 'datasets'
        self.transform = transform
        self.generate_new = generate_new
        self.replace_dataset = replace_dataset
        self.train_dataset, self.validation_dataset, self.test_dataset = self.load_dataset(train_split, transform)
        self.shuffle = shuffle

    def load_dataset(self, train_split, transform):
        train_dataset, validation_dataset, test_dataset = self.load_image_dataset(self.save_dataset_location)
        if train_dataset and test_dataset and validation_dataset \
                and self.generate_new is False and self.replace_dataset is False:
            print("Saved Data Found and Returned")
            return train_dataset, validation_dataset, test_dataset

        dataset = datasets.ImageFolder(root=self.training_data_set_location, transform=transform)
        if self.replace_dataset is True:
            print("Indices retrieved and Dataset replaced")
            train_dataset.dataset = dataset
            validation_dataset.dataset = dataset
            test_dataset.dataset = dataset
            return train_dataset, validation_dataset, test_dataset

        print("Generating New Set")

        # Split the dataset into training and test sets
        train_dataset, validation_dataset, test_dataset = random_split(dataset, train_split)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset
        self.save_image_dataset()
        return train_dataset, validation_dataset, test_dataset

    def get_dataset(self):
        return self.train_dataset, self.validation_dataset, self.test_dataset

    def save_image_dataset(self):
        train_file = os.path.join(self.save_dataset_location, 'train_data.pth')
        validation_file = os.path.join(self.save_dataset_location, 'validation_data.pth')
        test_file = os.path.join(self.save_dataset_location, 'test_data.pth')

        if os.path.exists(train_file) and os.path.exists(test_file) \
                and os.path.exists(validation_file) and self.generate_new is False:
            print("Files already exist. Skipping saving.")
            return

        train_data, validation_data, test_data = self.get_dataset()

        if not os.path.exists(self.save_dataset_location):
            os.makedirs(self.save_dataset_location)

        torch.save(train_data, train_file)
        torch.save(validation_data, validation_file)
        torch.save(test_data, test_file)

    def load_image_dataset(self, filepath):
        train_file = os.path.join(filepath, 'train_data.pth')
        validation_file = os.path.join(filepath, 'validation_data.pth')
        test_file = os.path.join(filepath, 'test_data.pth')

        if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(validation_file)):
            return None, None, None

        self.train_dataset = torch.load(train_file)
        self.test_dataset = torch.load(test_file)
        self.validation_dataset = torch.load(validation_file)
        return self.get_dataset()

    def load_data(self, batch_size=64):
        train_dataset, validation_dataset, test_dataset = self.get_dataset()
        train_loader = DL(train_dataset, batch_size=batch_size)
        validation_loader = DL(validation_dataset, batch_size=batch_size)
        test_loader = DL(test_dataset, batch_size=batch_size)

        return train_loader, validation_loader, test_loader

    def load_data_with_sampler(self, batch_size=64):
        train_dataset, validation_dataset, test_dataset = self.get_dataset()

        # sampler
        training_weight = self.calculate_sample_weights_from_dataset(train_dataset)
        validation_weight = self.calculate_sample_weights_from_dataset(validation_dataset)
        test_weight = self.calculate_sample_weights_from_dataset(test_dataset)

        training_sampler = WeightedRandomSampler(training_weight, len(train_dataset) // 2, replacement=False)
        test_sampler = WeightedRandomSampler(test_weight, len(test_dataset), replacement=True)
        validation_sampler = WeightedRandomSampler(validation_weight, len(validation_dataset), replacement=True)

        # Create DataLoader instances for training and test sets
        train_loader = DL(train_dataset, batch_size=batch_size, sampler=training_sampler)
        validation_loader = DL(validation_dataset, batch_size=batch_size, sampler=validation_sampler)
        test_loader = DL(test_dataset, batch_size=batch_size, sampler=test_sampler)

        return train_loader, validation_loader, test_loader

    def get_mean_std(self, dataloader):
        # Initialize sums and squared sums
        channel_sum, channel_squared_sum, num_batches = 0, 0, 0

        # Iterate over the DataLoader
        for data, _ in dataloader:
            # Assume data is a batch of images with shape [batch_size, channels, height, width]
            # Rearrange data to be in shape [channels, batch_size, height, width]
            data = data.permute(1, 0, 2, 3)

            # Update total number of images
            num_batches += data.shape[1]

            # Sum up all the pixel values for each channel
            channel_sum += torch.mean(data, dim=[1, 2, 3]) * data.shape[1]

            # Sum up all the squared pixel values for each channel
            channel_squared_sum += torch.mean(data ** 2, dim=[1, 2, 3]) * data.shape[1]

        # Calculate the mean and std dev for each channel
        mean = channel_sum / num_batches
        std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5

        return mean, std

    def load_small_data(self, transform, subset_size=1):
        # Define transformations to be applied to the images

        # Load the dataset using ImageFolder and apply transformations
        dataset = datasets.ImageFolder(root=self.training_data_set_location, transform=transform)
        sampler = RandomSampler(dataset, num_samples=subset_size)
        sample = DL(dataset, sampler=sampler, batch_size=subset_size)
        output = next(iter(sample))
        features, label = output
        return features, label

    def load_small_data_with_sampler(self, transform, subset_size=1):
        # Define transformations to be applied to the images

        # Load the dataset using ImageFolder and apply transformations
        dataset = datasets.ImageFolder(root=self.training_data_set_location, transform=transform)
        sampler = WeightedRandomSampler(dataset, weights=self.sample_weights, num_samples=subset_size)
        sample = DL(dataset, sampler=sampler, batch_size=subset_size)
        output = next(iter(sample))
        features, label = output
        return features, label

    def calculate_sample_weights_from_dataset(self, dataset):
        # Extract target labels from the dataset
        targets = torch.tensor([target for _, target in dataset])

        # Calculate the frequency of each class
        class_counts = torch.bincount(targets)

        # Calculate the total number of samples
        total_samples = len(targets)

        # Calculate the weight of each class
        class_weights = 1.0 / class_counts.float()

        # Calculate the weight of each sample based on its class
        weights = torch.zeros(total_samples, dtype=torch.float)
        for i, target in enumerate(targets):
            weights[i] = class_weights[target]

        return weights

    def load_image_folder(self):
        loaded_images = datasets.ImageFolder(root=self.training_data_set_location)
        return loaded_images

    def get_data_root(self):
        return self.training_data_set_location

    def categorize_images(self):
        training_labels = pd.read_csv(self.training_data_set_labels_location)

        labels_dict = dict(zip(training_labels['image_id'], training_labels['label']))
        categorizer = CategorizeImages('images', 'train_images', labels_dict)
        categorizer.categorize()

    def load_test_data(self):
        test_data = pd.read_csv(self.test_data_set_location)

        test_data_labels = test_data.iloc[:, 0]
        test_data_features = test_data.drop('label', axis=1)

        return test_data_features, test_data_labels

    def print_training_data(self):
        print(pd.read_csv(self.training_data_set_location))

    def print_testing_data(self):
        print(pd.read_csv(self.test_data_set_location))


class CategorizeImages:
    def __init__(self, output, picture_location, labels):
        self.output = output
        self.picture_location = picture_location
        self.labels = labels

    def categorize(self):
        # Create output root folder if it doesn't exist
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        # Create class directories in the output root folder
        for label in set(self.labels.values()):
            class_folder = os.path.join(self.output, f'class_{label}')
            os.makedirs(class_folder, exist_ok=True)

        # Copy images to corresponding class directories

        total_image_copied = 0
        for image_name, label in self.labels.items():
            try:
                source_path = os.path.join(self.picture_location, image_name)

                destination_folder = os.path.join(self.output, f'class_{label}')
                destination_path = os.path.join(destination_folder, image_name)
                shutil.copyfile(source_path, destination_path)
                total_image_copied += 1
            except Exception():
                print(source_path)
        print("Total Images Copied: ", total_image_copied)

from torcheval.metrics import MulticlassConfusionMatrix
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import transforms
from Models import VGGTransfer, CNNScratch, TestCNN, RexNetTransfer, RexNex150, CombinedVGGResNet, VGG19Modified, \
    EfficientNetV2L, CombinedVGGEfNet, SwimTrans, SwinEfficientNetComb


class LearningModel:
    learning_rate = 0.001
    target_height = 800
    target_width = 600

    shuffle = False
    criterion = nn.CrossEntropyLoss()
    padding_size = 0

    def __init__(self, dataset, model, number_of_classes, class_weights=None, epochs=1, batch_size=64):

        if class_weights is not None and len(class_weights) != number_of_classes:
            print("Invalid weight size")
            raise Exception()
        self.training_data = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.number_of_classes = number_of_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.get_model(model)
        self.model.to(self.device)
        self.transform = self.get_transformer()
        self.class_weight = class_weights
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.number_of_classes, device=self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.loss_values = []

    def get_transformer(self):
        return transforms.Compose([
            transforms.Pad(self.padding_size),
            transforms.Resize((self.target_height, self.target_width)),  # Resize if needed
            transforms.ToTensor(),  # Convert to tensor
        ])

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def get_model(self, model_name):
        if model_name == 'CNNScratch':
            model = CNNScratch(num_classes=self.number_of_classes)
        elif model_name == 'VGG':
            model = VGGTransfer(num_classes=self.number_of_classes)
        elif model_name == 'ResNet':
            model = RexNetTransfer(num_classes=self.number_of_classes)
        elif model_name == 'VGG19Modified':
            model = VGG19Modified(num_classes=self.number_of_classes)
        elif model_name == 'EfficientNetV2L':
            model = EfficientNetV2L(num_classes=self.number_of_classes)
        elif model_name == 'RexNet150':
            model = RexNex150(num_classes=self.number_of_classes)
        elif model_name == 'CombinedVGGResNet':
            model = CombinedVGGResNet(num_classes=self.number_of_classes)
        elif model_name == 'CombinedVGGEfNet':
            model = CombinedVGGEfNet(num_classes=self.number_of_classes)
        elif model_name == 'Swin':
            model = SwimTrans(num_classes=self.number_of_classes)
        elif model_name == 'SwinEfficientNetComb':
            model = SwinEfficientNetComb(num_classes=self.number_of_classes)
        elif model_name == "Test":
            model = TestCNN(num_classes=self.number_of_classes)
        else:
            raise Exception()
        print('Model: ', model_name)
        return model

    def set_epoch(self, epoch):
        self.epochs = epoch

    def set_batch_size(self, size):
        self.batch_size = size

    def set_param_to_zero(self):
        for param in self.model.parameters():
            param.data.fill_(0)

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle

    def train(self):
        self.model.train()
        return self.optimize()

    def get_loss(self):
        return self.loss_values

    def print_loss(self):
        print(self.loss_values)
        return self.loss_values

    def predict(self, x):
        self.model.eval()
        x = x.to(self.device)
        return self.model(x)

    def classify(self, x):
        self.model.eval()
        x = x.to(self.device)
        return self.predict(x).argmax(axis=1)

    def calculate_loss(self, outputs, labels):
        return self.criterion(outputs, labels)

    def optimize(self):
        if self.training_data is None:
            print("Invalid training data")
            raise Exception()
        print_every = 100
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.training_data):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.calculate_loss(outputs, labels)
                self.loss_values.append(loss.item())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if (i + 1) % print_every == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}], Batch [{i + 1}/{len(self.training_data)}], "
                        f"Loss: {running_loss / print_every:.4f}")
                    running_loss = 0.0

    def evaluate(self, dataloader):
        self.model.eval()  # Set the model to evaluation mode
        metrics = self.calculate_confusion_matrix(dataloader)
        self.clear_metrics()
        self.print_calculated_loss(dataloader)
        return self.print_pytorch_metrics(metrics)

    def clear_metrics(self):
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.number_of_classes, device=self.device)

    def print_calculated_loss(self, loader):
        loss = 0.0
        sample_size = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                self.optimizer.zero_grad()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                cur_loss = self.calculate_loss(outputs, labels)
                sample_size += len(inputs)
                loss += cur_loss.item() * len(inputs)
        print('Loss: ', loss / sample_size, ' with ', sample_size, ' samples')

    def calculate_global_metrics(self, dataloader):
        self.model.eval()  # Set the model to evaluation mode
        confusion_matrix = self.calculate_confusion_matrix(dataloader)
        TP = torch.diag(confusion_matrix).float()
        FP = torch.sum(confusion_matrix, dim=1).float() - TP
        FN = torch.sum(confusion_matrix, dim=0).float() - TP
        TN = torch.sum(confusion_matrix).float() - (TP + FP + FN)

        # Calculate Accuracy
        accuracy = torch.sum(TP) / torch.sum(confusion_matrix)

        # Calculate Precision, Recall, and F1 score
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
        macro_f1_score = torch.mean(f1_score)

        return accuracy.item(), macro_f1_score.item()

    def print_pytorch_metrics(self, confusion_matrix):
        # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
        TP = torch.diag(confusion_matrix).float()
        FP = torch.sum(confusion_matrix, dim=1).float() - TP
        FN = torch.sum(confusion_matrix, dim=0).float() - TP
        TN = torch.sum(confusion_matrix).float() - (TP + FP + FN)
        total_input_class_dist = confusion_matrix.int().sum(dim=1)
        total_pre_class_dist = confusion_matrix.int().sum(dim=0)

        # Calculate Accuracy
        accuracy = torch.sum(TP) / torch.sum(confusion_matrix)

        # Calculate Precision, Recall, and F1 score
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
        accuracy_per_class = (TP + TN) / (TP + TN + FP + FN)

        # Calculate False Positive Rate (FPR)
        FPR = FP / (FP + TN)

        # Calculate percentage of TP, TN, FP, FN for each class
        total_instances = torch.sum(confusion_matrix).float()
        tp_percentage = (TP / total_instances) * 100
        tn_percentage = (TN / total_instances) * 100
        fp_percentage = (FP / total_instances) * 100
        fn_percentage = (FN / total_instances) * 100

        # Handle division by zero errors
        precision = torch.nan_to_num(precision)
        recall = torch.nan_to_num(recall)
        f1_score = torch.nan_to_num(f1_score)

        # Print metrics for each class
        for i in range(len(TP)):
            print(f"Metrics for class {i}:")
            print(f"  Total Number of True Class {i}: {total_input_class_dist[i]}")
            print(f"  Total Number of Predicted Class {i}: {total_pre_class_dist[i]}")
            print(f"  True Positives (TP): {TP[i]} ({tp_percentage[i]}%)")
            print(f"  True Negatives (TN): {TN[i]} ({tn_percentage[i]}%)")
            print(f"  False Positives (FP): {FP[i]} ({fp_percentage[i]}%)")
            print(f"  False Negatives (FN): {FN[i]} ({fn_percentage[i]}%)")
            print(f"  Precision: {precision[i]}")
            print(f"  Accuracy: {accuracy_per_class[i]}")
            print(f"  Recall: {recall[i]}")
            print(f"  F1 Score: {f1_score[i]}")
            print(f"  False Positive Rate (FPR): {FPR[i]}")
            print()

        # Calculate macro-averaged metrics
        macro_precision = torch.mean(precision)
        macro_recall = torch.mean(recall)
        macro_f1_score = torch.mean(f1_score)

        # Print global metrics
        print("Global Metrics:")
        print(f"  Total Count: {total_instances.item()}")
        print(f"  Accuracy: {accuracy.item()}")
        print(f"  Macro Precision: {macro_precision.item()}")
        print(f"  Macro Recall: {macro_recall.item()}")
        print(f"  Macro F1 Score: {macro_f1_score.item()}")
        print()

        # Print confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix.int())

        # Print Data Distribution
        print("Data Class Distribution Matrix:")
        print(confusion_matrix.int().sum(dim=1))

        return accuracy.item(), accuracy.item()

    def calculate_confusion_matrix(self, dataloader):
        self.model.eval()
        self.confusion_matrix.reset()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs, 1)

                # Update confusion matrix
                self.confusion_matrix.update(predicted, labels)

        metrics = self.confusion_matrix.compute()
        self.clear_metrics()
        return metrics

    def calculate_accuracy_and_f1(self, dataloader):
        #
        # Resource: https://www.evidentlyai.com/classification-metrics/multi-class-metrics#
        correct_predictions = 0
        total_predictions = 0
        true_positives = [0] * self.number_of_classes
        false_positives = [0] * self.number_of_classes
        false_negatives = [0] * self.number_of_classes
        class_accuracies = [0] * self.number_of_classes
        f1_scores = [0] * self.number_of_classes
        class_weights = [0] * self.number_of_classes

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                for i in range(self.number_of_classes):
                    true_positives[i] += ((predicted == i) & (labels == i)).sum().item()
                    false_positives[i] += ((predicted == i) & (labels != i)).sum().item()
                    false_negatives[i] += ((predicted != i) & (labels == i)).sum().item()

            accuracy = correct_predictions / total_predictions

            for i in range(self.number_of_classes):
                precision = true_positives[i] / (true_positives[i] + false_positives[i] + 1e-15)
                recall = true_positives[i] / (true_positives[i] + false_negatives[i] + 1e-15)
                f1_scores[i] = 2 * (precision * recall) / (precision + recall + 1e-15)

                # Calculate accuracy for class i
                class_total = true_positives[i] + false_positives[i] + false_negatives[i]
                class_accuracies[i] = true_positives[i] / (class_total + 1e-15)

                # Calculate class weight
                class_weights[i] = class_total / total_predictions

        # Calculate weighted global F1 score
        global_f1 = sum(f1 * weight for f1, weight in zip(f1_scores, class_weights))

        self.print_metrics(accuracy, global_f1, true_positives,
                           false_positives, false_negatives, f1_scores, class_accuracies)

        return accuracy, global_f1

    def top_k_accuracy(self, output, target, k):
        with torch.no_grad():
            # Get the indices of top k predictions from the last dimension
            pred = torch.topk(output, k, dim=1).indices
            # Check if the target is in the predicted top-k indices
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            # Calculate the top-k accuracy
            correct_k = correct.view(-1).float().sum(0, keepdim=True)
            return correct_k

    def evaluate_top_k_accuracy(self, dataloader, k):
        self.model.eval()  # Set the model to evaluation mode
        correct_topk = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                # Increment the number of correct top-k predictions
                correct_topk += self.top_k_accuracy(outputs, labels, k)
                # Increment the total number of samples
                total += labels.size(0)

        # Calculate the top-k accuracy percentage
        top_k_accuracy = correct_topk / total * 100
        return top_k_accuracy.item()

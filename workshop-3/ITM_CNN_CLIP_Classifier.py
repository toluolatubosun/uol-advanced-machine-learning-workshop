################################################################################
# Image-Text Matching Classifier: baseline system for visual question answering
# 
# This program has been adapted and rewriten from the CMP9137 materials of 2026.
# 
# It treats the task of multi-choice visual question answering as a binary
# classification task. This is possible by rewriting the questions from this format:
# v7w_2358727.jpg	When was this?  Nighttime. | Daytime. | Dawn. Sunset.
# 
# to the following format:
# v7w_2358727.jpg	When was this? Nighttime. 	match
# v7w_2358727.jpg	When was this?  Daytime. 	no-match
# v7w_2358727.jpg	When was this?  Dawn. 	no-match
# v7w_2358727.jpg	When was this?  Sunset.	no-match
#
# The list above contains the image file name, the question-answer pairs, and the labels.
# Only question types "when", "where" and "who" were used due to compute requirements. In
# this folder, files v7w.*Images.itm.txt are used and v7w.*Images.txt are ignored. The 
# two formats are provided for your information and convenience.
# 
# To enable the above this implementation provides the following classes and functions:
# - Class ITM_Dataset() to load the multimodal data (image & text (question and answer)).
# - Function load_sentence_embeddings() to load pre-generated sentence embeddings of questions 
#   and answers, which were generated using SentenceTransformer('sentence-transformers/gtr-t5-large').
# - Class ITM_Model() to create a model combining the vision and text encoders above. 
# - Class ITM_Model_CLIP() to create a model combining the vision & text encoders with CLIP. 
# - Function train_model trains/finetunes one of two possible models: CNN or ViT. The CNN 
#   model is based on resnet18; see further details of this architecture in the link below.
# - Function evaluate_model() calculates the accuracy of the selected model using test data. 
# - The last block of code brings everything together calling all classes & functions above.
# 
# info of resnet18: https://pytorch.org/vision/main/models/resnet.html
# info of SentenceTransformer: https://huggingface.co/sentence-transformers/gtr-t5-large
# info of CLIP: https://openai.com/index/clip/
#
# This program was tested on Windows 11 using WSL and does not generate any plots. 
# Feel free to use and extend this program as part of your our assignment work.
#
# Version 1.0, main functionality in tensorflow tested with COCO data 
# Version 1.2, extended functionality for Flickr data
# Version 1.3, ported to pytorch and tested with visual7w data
# Version 1.4, extended to support the CLIP approach
# Contact: {hcuayahuitl}@lincoln.ac.uk
################################################################################

import os
import time
import pickle
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset


# custom Dataset
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, sentence_embeddings, data_split, train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
        self.sentence_embeddings = sentence_embeddings
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        self.question_embeddings_data = []
        self.answer_embeddings_data = []
        self.label_data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # standard for pretrained models on ImageNet
        ])

        self.load_data()

    def load_data(self):
        print("LOADING data from "+str(self.data_file))
        print("=========================================")

        random.seed(42)

        with open(self.data_file) as f:
            lines = f.readlines()

            # apply train_ratio only for training data
            if self.data_split == "train":
                random.shuffle(lines)  # shuffle before selecting
                num_samples = int(len(lines) * self.train_ratio)
                lines = lines[:num_samples]

            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("\t")  
                img_path = os.path.join(self.images_path, img_name.strip())

                question_answer_text = text.split("?")
                question_text = question_answer_text[0].strip() + '?'
                answer_text = question_answer_text[1].strip()

                # get binary labels from match/no-match answers
                label = 1 if raw_label == "match" else 0
                self.image_data.append(img_path)
                self.question_data.append(question_text)
                self.answer_data.append(answer_text)
                self.question_embeddings_data.append(self.sentence_embeddings[question_text])
                self.answer_embeddings_data.append(self.sentence_embeddings[answer_text])
                self.label_data.append(label)

        print("|image_data|="+str(len(self.image_data)))
        print("|question_data|="+str(len(self.question_data)))
        print("|answer_data|="+str(len(self.answer_data)))
        print("done loading data...")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = self.image_data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  
        question_embedding = torch.tensor(self.question_embeddings_data[idx], dtype=torch.float32)
        answer_embedding = torch.tensor(self.answer_embeddings_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        return img, question_embedding, answer_embedding, label

# load sentence embeddings from an existing file -- generated a priori
def load_sentence_embeddings(file_path):
    print("READING sentence embeddings...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def retrieve_vision_model(PRETRAINED, embedding_dim):
    vision_model = models.resnet18(pretrained=PRETRAINED)
    #vision_model = models.resnet50(pretrained=PRETRAINED)
    if PRETRAINED:
		# freeze all layers 
        for param in vision_model.parameters():
            param.requires_grad = False # no calculation of gradients

        # unfreeze the last two layers
        for param in list(vision_model.children())[-2:]:
            for p in param.parameters():
                p.requires_grad = True # gradients are calculated
    else:
        for param in vision_model.parameters():
            param.requires_grad = True # gradients are calculated
    
    vision_model.fc = nn.Linear(vision_model.fc.in_features, embedding_dim) # change output's dimensionality
    return vision_model

# Image-Text Matching model
class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, EMBEDDING_DIM=None, ARCHITECTURE=None, PRETRAINED=None):
        print(f'BUILDING %s model, pretrained=%s' % (ARCHITECTURE, PRETRAINED))
        super(ITM_Model, self).__init__()

        self.vision_model = retrieve_vision_model(PRETRAINED, 128)
        self.question_embedding_layer = nn.Linear(768, EMBEDDING_DIM)  # adjust question dimension 
        self.answer_embedding_layer = nn.Linear(768, EMBEDDING_DIM)  # adjust answer dimension
        self.fc = nn.Sequential(nn.Linear(EMBEDDING_DIM*3, 128), nn.ReLU(), nn.Linear(128, 2)) # small MLP for the joint embeddings

    def forward(self, img, question_embedding, answer_embedding):
        img_features = self.vision_model(img)
        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)
        output = self.fc(combined_features)
        return output, None, None


class ITM_Model_CLIP(nn.Module):
    def __init__(self, num_classes=2, EMBEDDING_DIM=None, ARCHITECTURE=None, PRETRAINED=None):
        print(f'BUILDING %s model, pretrained=%s' % (ARCHITECTURE, PRETRAINED))
        super(ITM_Model_CLIP, self).__init__()

        self.vision_model = retrieve_vision_model(PRETRAINED, EMBEDDING_DIM)
        self.question_embedding_layer = nn.Linear(768, EMBEDDING_DIM)  # adjust question dimension 
        self.answer_embedding_layer = nn.Linear(768, EMBEDDING_DIM)  # adjust answer dimension
        self.text_projection = nn.Linear(2 * EMBEDDING_DIM, EMBEDDING_DIM) # single embedding for question and answer
        self.fc = nn.Sequential(nn.Linear(EMBEDDING_DIM*2 + 1, 128), nn.ReLU(), nn.Linear(128, 2)) # small MLP for the joint embeddings
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07)) # learnable temperature parameter

    def forward(self, img, question_embedding, answer_embedding):
        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        qa_concat = torch.cat([question_features, answer_features], dim=-1) # concatenation of text features

        text_features = F.normalize(self.text_projection(qa_concat), dim=-1) # L2 regularisation of text features
        img_features = F.normalize(self.vision_model(img), dim=-1) # L2 regularisation of image features
        similarity = (img_features * text_features).sum(dim=-1, keepdim=True) # dot product between vectors
        similarity *= self.logit_scale.exp()

        combined_features = torch.cat([img_features, text_features, similarity], dim=-1) # concatenate along the feature dimension
        output = self.fc(combined_features)
        return output, img_features, text_features
    
    def CLIP_loss(self, img_features, text_features):
        logits = self.logit_scale.exp() * torch.matmul(img_features, text_features.t())  # [N, N] similarity matrix
        labels = torch.arange(len(img_features), device=img_features.device)
        loss_img = F.cross_entropy(logits, labels)      # error in predicting text given image
        loss_text = F.cross_entropy(logits.t(), labels) # error in predicting image given text
        return (loss_img + loss_text) / 2


# class used to prevent overfitting VQA models
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_count = 0

    def check_early_stopping(self, val_loss):
        improved = False
        if val_loss < self.best_loss-1e-3: # check for improvements larger than a threshold 1e-3
            self.best_loss = val_loss
            self.patience_count = 0
            improved = True
            stop =  False
        else:
            self.patience_count += 1
            improved = False
            stop = True if self.patience_count >= self.patience else False

        print("Validation loss=%.5f, improved=%s: patience=%s" % (val_loss, improved, self.patience_count))

        if stop: print("Stopping early!")
        return stop

    def reset(self):
        self.best_loss = float('inf')
        self.patience_count = 0


# general training procedure used by different deep learning architectures
def train_model(model, ARCHITECTURE, train_loader, val_loader, criterion, optimiser, num_epochs=10):
    print(f'TRAINING %s model' % (ARCHITECTURE))
    model.train()
    alpha = 0.2

    early_stopping = EarlyStopping(patience=3)

    # track the overall loss for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        start_time = time.time()

        for batch_idx, (images, question_embeddings, answer_embeddings, labels) in enumerate(train_loader):
            # move images/text/labels to the GPU (if available)
            images = images.to(device)          
            question_embeddings = question_embeddings.to(device) 
            answer_embeddings = answer_embeddings.to(device)  
            labels = labels.to(device)

            # forward pass -- given input data to the model
            outputs, img_features, text_features = model(images, question_embeddings, answer_embeddings)

            # calculate loss (error)
            loss = criterion(outputs, labels)  # output should be raw logits
            if hasattr(model, "CLIP_loss"):
                contrastive_loss = model.CLIP_loss(img_features, text_features)
                loss = loss + alpha * contrastive_loss

            # backward pass -- given loss above
            optimiser.zero_grad() # clear the gradients
            loss.backward() # computes gradient of the loss/error
            optimiser.step() # updates parameters using gradients
            running_loss += loss.item()

            # print progress every X batches
            if batch_idx % 100 == 0:
                #print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}')
                print('Epoch [%s/%s], Batch [%s/%s], Loss: %.4f' % (epoch+1, num_epochs, batch_idx, total_batches, loss.item()))

        if val_loader is not None:
            # check whether to carry on training or not
            val_loss = running_loss / len(val_loader)  # scalar to monitor
            if early_stopping.check_early_stopping(val_loss):
                break
        
        # print average loss for the epoch
        avg_loss = running_loss / total_batches
        elapsed_time = time.time() - start_time
        #print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}, {elapsed_time:.2f} seconds')
        print('Epoch [%s/%s] Average Loss: %.4f, %.2f seconds' % (epoch+1, num_epochs, avg_loss, elapsed_time))

def evaluate_model(model, ARCHITECTURE, test_loader, device):
    print(f'EVALUATING %s model' % (ARCHITECTURE))
    model.eval()
    total_test_loss = 0
    all_labels = []
    all_predictions = []
    start_time = time.time()

    with torch.no_grad():
        for images, question_embeddings, answer_embeddings, labels in test_loader:
            # move images/text/labels to the GPU (if available)
            images = images.to(device)          
            question_embeddings = question_embeddings.to(device) 
            answer_embeddings = answer_embeddings.to(device)  
            labels = labels.to(device)  # Labels are single integers (0 or 1)
			
            # perform forward pass on our data
            outputs, _, _ = model(images, question_embeddings, answer_embeddings)
			
            # accumulate loss on test data
            total_test_loss += criterion(outputs, labels)  

            # aince outputs are logits, apply softmax to get probabilities
            predicted_probabilities = torch.softmax(outputs, dim=1)  # use softmax for multi-class output
            predicted_class = predicted_probabilities.argmax(dim=1)  # get the predicted class index (0 or 1)

            # store labels and predictions for later analysis
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())

    # convert to numpy arrays for easier calculations
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # calculate true positives, true negatives, false positives, false negatives
    tp = np.sum((all_predictions == 1) & (all_labels == 1))  # True positives
    tn = np.sum((all_predictions == 0) & (all_labels == 0))  # True negatives
    fp = np.sum((all_predictions == 1) & (all_labels == 0))  # False positives
    fn = np.sum((all_predictions == 0) & (all_labels == 1))  # False negatives

    # calculate sensitivity, specificity, and balanced accuracy
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # true positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # 1 - false positive rate
    balanced_accuracy = (sensitivity + specificity) / 2.0

    elapsed_time = time.time() - start_time
    print("Balanced Accuracy=%.4f, %.2f seconds" % (balanced_accuracy, elapsed_time))
    print("Total Test Loss=%.4f" % (total_test_loss))

# main Execution
if __name__ == '__main__':
    # Cceck GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # paths and files
    IMAGES_PATH = "./visual7w-images"
    train_data_file = "./visual7w-text/v7w.TrainImages.itm.txt"
    dev_data_file = "./visual7w-text/v7w.DevImages.itm.txt"
    test_data_file = "./visual7w-text//v7w.TestImages.itm.txt"
    sentence_embeddings_file = "./v7w.sentence_embeddings-gtr-t5-large.pkl"
    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)
    USE_EARLY_STOPPING = True
    EMBEDDING_LEN = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    EPOCHS = 1000 # only use a large number if USE_EARLY_STOPPING is True

    # create datasets and loaders
    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test")  # whole test data
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if USE_EARLY_STOPPING:
        # create validation data by splitting training data into training and validation
        val_dataset = ITM_Dataset(IMAGES_PATH, dev_data_file, sentence_embeddings, data_split="val")  # whole dev data
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        val_loader = None
    
    # create the model using one of the two supported architectures
    MODEL_ARCHITECTURE = "CNN" # options are "CNN" or "CNN_CLIP"
    USE_PRETRAINED_MODEL = True
    if MODEL_ARCHITECTURE == "CNN_CLIP":
        model = ITM_Model_CLIP(num_classes=2, EMBEDDING_DIM=EMBEDDING_LEN, ARCHITECTURE=MODEL_ARCHITECTURE, PRETRAINED=USE_PRETRAINED_MODEL).to(device)
    else:
        model = ITM_Model(num_classes=2, EMBEDDING_DIM=EMBEDDING_LEN, ARCHITECTURE=MODEL_ARCHITECTURE, PRETRAINED=USE_PRETRAINED_MODEL).to(device)
    print("\nModel Architecture:")
    print(model)

    # print the parameters of the model selected above
    total_params = 0
    print("\nModel Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:  # print trainable parameters
            num_params = param.numel() # number of trainable weights/params. 
            total_params += num_params
            print("%s: %s | Number of parameters: %s" % (name, param.data.shape, num_params))
    print("\nTotal number of parameters in the model=%s" % (total_params))
    print("\nUSE_PRETRAINED_MODEL=%s\n" % (USE_PRETRAINED_MODEL))

    # define loss function and optimiser 
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # train and evaluate the model
    train_model(model, MODEL_ARCHITECTURE, train_loader, val_loader, criterion, optimiser, num_epochs=EPOCHS)
    evaluate_model(model, MODEL_ARCHITECTURE, test_loader, device)

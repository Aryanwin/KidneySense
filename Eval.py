import torch
import os
from  sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from DataOrganization import transformed_dataset_test
from Train import model

net = model.network
dataiter = iter(transformed_dataset_test)
predictions = []
correct = []

for each in transformed_dataset_test:
    sample = next(dataiter)
    images, labels = sample["image"].to(torch.float), sample["label"]

    output = model.predict(images)
    _, predicted = torch.max(output, 1)

    classes = [0, 1, 2, 3]
    predictions.append(int(classes[predicted[0]]))
    correct.append(labels)

model_dir = 'models'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
PATH = os.path.join(model_dir, "model")
torch.save(model.network.state_dict(), PATH)

def classificationreport():
    print(classification_report(correct, predictions))

def confusionmatrix():
    cm = confusion_matrix(correct, predictions, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
    disp.plot()
    plt.show()

def predictionout(imagein):
    torchimage = imagein.to(torch.float)
    outputs = model.predict(torchimage)
    _, predict1 = torch.max(outputs, 1)
    class1 = [0, 1, 2, 3]
    return int(class1[predict1[0]])
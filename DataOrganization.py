import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from   sklearn.model_selection import train_test_split
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import glob
global device

if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('Running on the GPU')
else:
  device = torch.device('cpu')
  print('Running on the CPU')

data_dir = "/Users/aryanmansingh/Documents/Machine learning/Inspirit stuff/data"
subdirs = [x[0] for x in os.walk(data_dir)]
subdirs = subdirs[1::]
imgdict = {}
for i in subdirs:
    label = ""

    if "Normal" in i:
        label = 0
    if "Tumor" in i:
        label = 1
    if "Stone" in i:
        label = 2
    if "Cyst" in i:
        label = 3

    filenames = glob.glob(i + '/*.jpg')
    for j in filenames:
        imgdict[j] = label

df = pd.DataFrame(imgdict.items(), columns=['path', 'label'])
X=df["path"]
y=df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
X_train.head()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42, stratify = y_train)

df_train = pd.concat([X_train, y_train], axis=1)
df_val = pd.concat([X_val, y_val], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

class KidneyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform
        #self.img_dim = (128, 128)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        image = io.imread(img_name)
        #print(image.shape)
        #image = image.resize(self.img_dim)

        #image = cv2.imread(img_name)
        #image = cv2.resize(image, self.img_dim)

        #img_tensor = torch.from_numpy(image)
        #img_tensor = img_tensor.permute(2, 0, 1)
        #class_id = torch.tensor([label])

        sample = {"image":image, "label":label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        label = sample["label"]

        return {'image': img, 'label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, (h - new_h + 1))
        left = np.random.randint(0, (w - new_w + 1))

        image = image[top: top + new_h,
                      left: left + new_w]

        label = label

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label}

transformed_dataset_train = KidneyDataset(df_train, transform = transforms.Compose([Rescale(32), RandomCrop(32),ToTensor()]))
transformed_dataset_val = KidneyDataset(df_val, transform = transforms.Compose([Rescale(32), RandomCrop(32),ToTensor()]))
transformed_dataset_test = KidneyDataset(df_test, transform = transforms.Compose([Rescale(32), RandomCrop(32),ToTensor()]))

for i, sample in enumerate(transformed_dataset_train):
    #print(i, sample['image'].size(), sample["label"])

    if i ==3:
        break

dataloader = DataLoader(transformed_dataset_train, batch_size=4,
                        shuffle=True, num_workers=0)


# Helper function to show a batch

def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.title('Batch from dataloader')
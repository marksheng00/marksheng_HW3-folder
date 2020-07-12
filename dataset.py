from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from file_train import file_train
from read_pmf import readPFM


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def default_loader(path):
    img_pil = readPFM(path)
    img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor


class trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.images = file_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        return img

    def __len__(self):
        return len(self.images)


train_data = trainset()
trainloader = DataLoader(train_data, batch_size=4,shuffle=True)
###
Traceback (most recent call last):
  File "E:/flownets/dataset.py", line 43, in <module>
    trainloader = DataLoader(train_data, batch_size=4,shuffle=True)
  File "D:\AI\Anaconda\lib\site-packages\torch\utils\data\dataloader.py", line 213, in __init__
    sampler = RandomSampler(dataset)
  File "D:\AI\Anaconda\lib\site-packages\torch\utils\data\sampler.py", line 92, in __init__
    if not isinstance(self.num_samples, int) or self.num_samples <= 0:
  File "D:\AI\Anaconda\lib\site-packages\torch\utils\data\sampler.py", line 100, in num_samples
    return len(self.data_source)
  File "E:/flownets/dataset.py", line 39, in __len__
    return len(self.images)
TypeError: object of type 'function' has no len()

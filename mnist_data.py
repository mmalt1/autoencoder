import pickle
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

x_train = train_dataset.data.numpy()
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

with open('mnist_train.pkl', 'wb') as f:
    pickle.dump((x_train, y_train), f)

with open('mnist_valid.pkl', 'wb') as f:
    pickle.dump((x_test, y_test), f)
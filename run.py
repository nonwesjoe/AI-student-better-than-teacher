import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 制造部分正确标签
def corrupt_labels(dataset, correct_ratio=1, num_classes=10):
    labels = dataset.targets.clone()
    n_samples = len(labels)
    n_correct = int(correct_ratio * n_samples)

    keep_idx = set(random.sample(range(n_samples), n_correct))
    for i in range(n_samples):
        if i not in keep_idx:
            labels[i] = random.randint(0, num_classes - 1)
    dataset.targets = labels


# 简单 CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 训练
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# 测试
def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = correct / len(test_loader.dataset)
    return acc

# 不同标签正确率下的训练 & 测试
ratios = [0.05, 0.1, 0.2, 0.4,0.8, 1.0]
# 存储不同 ratio 下的干净和噪声验证集准确率
dic_clean = {}
dic_noisy = {}

for r in ratios:
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset_clean = datasets.MNIST('./data', train=False, transform=transform)
    noisy_test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # 制造训练集和 noisy 验证集的标签噪声
    corrupt_labels(train_dataset, correct_ratio=r)
    corrupt_labels(noisy_test_dataset, correct_ratio=r)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader_clean = torch.utils.data.DataLoader(test_dataset_clean, batch_size=1000, shuffle=False)
    test_loader_noisy = torch.utils.data.DataLoader(noisy_test_dataset, batch_size=1000, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    accs_clean = []
    accs_noisy = []

    for epoch in range(1, 21):  # 可以先改成 5
        train(model, train_loader, optimizer, epoch)

        # 干净验证集
        acc_clean = test(model, test_loader_clean)
        # 被污染验证集
        acc_noisy = test(model, test_loader_noisy)

        accs_clean.append(acc_clean)
        accs_noisy.append(acc_noisy)

        print(f"Ratio={r}, Epoch={epoch}, Clean Acc={acc_clean:.4f}, Noisy Acc={acc_noisy:.4f}")

    dic_clean[r] = accs_clean
    dic_noisy[r] = accs_noisy


# 画图
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 左图：干净验证集
for r, accs in dic_clean.items():
    axs[0].plot(range(1, len(accs)+1), accs, marker='o', label=f"ratio={r}")
axs[0].set_title("Accuracy on Clean Validation Set")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Accuracy")
axs[0].grid(True)
axs[0].legend()

# 右图：被污染验证集
for r, accs in dic_noisy.items():
    axs[1].plot(range(1, len(accs)+1), accs, marker='x', linestyle='--', label=f"ratio={r}")
axs[1].set_title("Accuracy on Noisy Validation Set")
axs[1].set_xlabel("Epoch")
axs[1].grid(True)
axs[1].legend()

plt.suptitle("Effect of Label Noise on MNIST")
plt.tight_layout()
plt.show()

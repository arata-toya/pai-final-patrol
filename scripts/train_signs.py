from pathlib import Path
import csv
import random
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["normal", "inspect", "background"]
SIZE = 64

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(2)

class SignDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.counts = []
        for index, label in enumerate(LABELS):
            files = sorted((ROOT / "data" / label).glob("*.png"))
            if not files:
                raise RuntimeError(f"{label} の画像がありません")
            self.counts.append(len(files))
            for path in files:
                image = cv2.imread(str(path))
                if image is None:
                    raise RuntimeError(f"読み込み失敗: {path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (SIZE, SIZE))
                self.samples.append((image, index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, label = self.samples[index]
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1), label

# 推論時にも、この構成を使います。
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 3),
)

dataset = SignDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# 背景画像が多いため、少ないクラスの損失を重くします。
weights = 1.0 / torch.tensor(dataset.counts, dtype=torch.float32)
weights = weights / weights.mean()
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

output = ROOT / "models"
output.mkdir(parents=True, exist_ok=True)
history = []

print("使用画像:", dict(zip(LABELS, dataset.counts)), flush=True)
print("CPUで100エポック学習します", flush=True)

for epoch in range(1, 101):
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        # 明るさと左右反転を変えた画像でも学習します。
        factors = 0.8 + 0.4 * torch.rand(len(images), 1, 1, 1)
        images = (images * factors).clamp(0, 1)
        flip = torch.rand(len(images)) < 0.5
        images[flip] = images[flip].flip(-1)

        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)

    model.eval()
    correct = [0, 0, 0]
    with torch.no_grad():
        for images, targets in DataLoader(dataset, batch_size=16):
            predictions = model(images).argmax(1)
            for target, prediction in zip(targets.tolist(), predictions.tolist()):
                correct[target] += int(target == prediction)

    accuracy = sum(correct) / len(dataset)
    mean_loss = total_loss / len(dataset)
    history.append([epoch, mean_loss, accuracy])

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"epoch {epoch:3d}/100  loss={mean_loss:.4f}  "
            f"学習画像の正解率={accuracy:.1%}",
            flush=True
        )

torch.save({
    "model_state": model.state_dict(),
    "labels": LABELS,
    "image_size": SIZE,
    "architecture": "sign_cnn_v1",
    "seed": 42,
    "training_counts": dataset.counts,
}, output / "sign_classifier.pt")

with (output / "training_log.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss", "training_accuracy"])
    writer.writerows(history)

print("\n学習画像での結果:")
for label, count, ok in zip(LABELS, dataset.counts, correct):
    print(f"  {label}: {ok}/{count}")
print("保存先:", output / "sign_classifier.pt")
print("注意: これは学習画像の成績です。未知の画像の評価は次に行います。")

[toc]

# Transforms：裁剪

在 PyTorch 中，`RandomResizedCrop` 和 `Resize` 是用于图像预处理的两个重要变换操作。它们各自有不同的功能和用途。下面是这两个操作的用法及其区别：

## RandomResizedCrop

**功能：**

- `RandomResizedCrop` 在给定的范围内随机裁剪图像，并调整裁剪后的图像到指定的尺寸。它通常用于数据增强，以增加模型对不同图像缩放和裁剪的鲁棒性。

**用法：**
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
    transforms.ToTensor(),
])
```
- **`size`**：裁剪后的图像尺寸（宽度和高度）。如果是整数，则裁剪的区域将是正方形；如果是元组 `(height, width)`，则裁剪的区域将是矩形。
- **`scale`**：裁剪区域相对于原始图像的面积比例范围，应该是一个元组 `(min, max)`。例如，`(0.5, 1.0)` 意味着裁剪区域的面积将占原始图像面积的 50% 到 100%。

**特点：**
- **随机性**：每次调用 `RandomResizedCrop` 都会在指定的范围内随机裁剪图像的区域，因此每次生成的裁剪区域可能不同。
- **增强数据**：通过随机裁剪和调整尺寸，可以使模型对不同的图像尺度和裁剪位置更加鲁棒，有助于提高模型的泛化能力。

## Resize

**功能：**
- `Resize` 直接将图像调整到指定的尺寸，无论原始图像的尺寸如何。它不会裁剪图像，只是进行尺寸调整。

**用法：**
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.ToTensor(),
])
```
- **`size`**：目标尺寸，可以是一个整数或一个元组。如果是整数，图像会被调整为相同的宽度和高度（正方形）；如果是元组 `(height, width)`，则调整为指定的高度和宽度。

**特点：**
- **非随机性**：`Resize` 不涉及随机裁剪或变换，每次操作都会将图像调整到相同的目标尺寸。
- **标准化尺寸**：对于要求输入图像有统一尺寸的模型，`Resize` 可以将图像调整为所需的标准尺寸。

## 总结

- **`RandomResizedCrop`**：用于数据增强，通过随机裁剪和调整图像尺寸来增强模型对图像变换的鲁棒性。每次处理图像时都会产生不同的裁剪区域。
- **`Resize`**：用于将图像调整为指定的固定尺寸，适用于需要统一输入尺寸的场景。

选择使用哪种方法取决于你的具体需求。如果需要数据增强和增加模型的泛化能力，可以使用 `RandomResizedCrop`。如果你只需要统一的输入尺寸，则 `Resize` 更为直接和简单。
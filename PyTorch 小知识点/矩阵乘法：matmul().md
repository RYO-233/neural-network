[toc]

# 矩阵乘法之 matmul() 的使用

官方文档：[torch.matmul — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul)

`torch.matmul()` 函数几乎可以用于所有矩阵 / 向量相乘的情况，其乘法规则视参与乘法的两个张量的维度而定。

`torch.matmul()` 将两个张量相乘划分成了五种情形：一维 × 一维、二维 × 二维、一维 × 二维、二维 × 一维、涉及到三维及三维以上维度的张量的乘法。

## 一维 × 一维

如果两个张量都是一维的，即 `torch.Size([n])`，此时返回两个向量的点积。**作用与 `torch.dot()` 相同，同样要求两个一维张量的元素个数相同。**

```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([2, 3, 4])
z = torch.tensor([3, 4, 5, 6])

rs1 = torch.matmul(x, y)
print(f'rs1 = {rs1}')  # 20
rs2 = torch.dot(x, y)
print(f'rs2 = {rs2}')  # 20
```

```python
# 两个一维张量的元素个数要相同！
rs = torch.matmul(x, z)
print(f'rs = {rs}')
"""
Traceback (most recent call last):
  File "D:\Codes\neuralnetwork\pytorch\matmul_use.py", line 12, in <module>
    rs = torch.matmul(x, z)
RuntimeError: inconsistent tensor size, expected tensor [3] and src [4] to have the same number of elements, but got 3 and 4 elements respectively
"""
```

## 二维 × 二维

如果两个参数都是二维张量，那么将返回矩阵乘积。**作用与 `torch.mm()` 相同，同样要求两个张量的形状需要满足矩阵乘法的条件，即 $(n×m) × (m×p) = (n×p)$**

```python
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[-1], [2]])
print(x.shape, y.shape)  # torch.Size([2, 2]) torch.Size([2, 1])

rs1 = torch.matmul(x, y)
print(f'rs1 = {rs1}')
rs2 = torch.mm(x, y)
print(f'rs2 = {rs2}')
"""
rs1 = tensor([[3],
        [5]])
rs2 = tensor([[3],
        [5]])
"""
```

## 一维 × 二维

如果第一个参数是一维张量，第二个参数是二维张量，那么**在一维张量的前面增加一个维度，然后进行矩阵乘法，矩阵乘法结束后移除添加的维度。**

> 文档原文为：“a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.”

```python
x = torch.tensor([-1, 2])
y = torch.tensor([[1, 2], [3, 4]])

rs = torch.matmul(x, y)
print(f'rs = {rs}')  # tensor([[5, 6]])

x = torch.unsqueeze(x, 0)  # 在一维张量前增加一个维度
print(x, x.shape)
rs = torch.mm(x, y)
print(f'rs = {rs}')  # tensor([[5, 6]])
x = torch.squeeze(x, 0)
print(x, x.shape)  # tensor([-1,  2]) torch.Size([2])
```

## 二维 × 一维

如果第一个参数是二维张量（矩阵），第二个参数是一维张量（向量），那么将返回矩阵×向量的积。**作用与 `torch.mv()` 相同。另外要求矩阵的形状和向量的形状满足矩阵乘法的要求。**

```python
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([-1, 2])

rs1 = torch.matmul(x, y)
print(f'rs1 = {rs1}')  # tensor([3, 5])
rs2 = torch.mv(x, y)
print(f'rs2 = {rs2}')  # tensor([3, 5])
```

## 高维 × 高维

如果两个参数均至少为一维，且其中一个参数的 `ndim > 2`，那么……（一番处理），然后进行批量矩阵乘法。

这条规则将所有涉及到三维张量及三维以上的张量（下文称为高维张量）的乘法分为三类：一维张量 × 高维张量、高维张量 × 一维张量、二维及二维以上的张量 × 二维及二维以上的张量。

1. 如果第一个参数是一维张量，那么在此张量**之前**增加一个维度。

	> 文档原文为：“ If the first argument is 1-dimensional, a 1 is **prepended** to its dimension for the purpose of the batched matrix multiply and removed after.”

2. 如果第二个参数是一维张量，那么在此张量**之后**增加一个维度。

	> 文档原文为：“If the second argument is 1-dimensional, a 1 is **appended** to its dimension for the purpose of the batched matrix multiple and removed after. ”

3. 由于上述两个规则，**所有涉及到一维张量和高维张量的乘法都被转变为二维及二维以上的张量 × 二维及二维以上的张量。**

	然后除掉最右边的两个维度，对剩下的维度进行广播。原文为：“The non-matrix dimensions are broadcasted.”

	然后就可以进行批量矩阵乘法。

	`For example, if input is a (j × 1 × n × n) tensor and other is a (k × n × n) tensor, out will be a (j × k × n × n) tensor.`

举例说明：根据第一条规则，先对 `x` 增加维度；由于 `y.shape = torch.Size([3, 4, 1])` ，根据广播的规则，`x1` 要被广播为 `torch.Size([3, 1, 4])` ，也就是 `x2`。最后使用乘法函数 `torch.bmm()` 来进行批量矩阵乘法。

由于在第一条规则中对一维张量增加了维度，因此矩阵计算结束后要移除这个维度。移除之后和前面使用 `torch.matmul()` 的结果相同。

```python
x = torch.tensor([1, 2, 3, 4])
y = torch.randint(-2, 3, size=(3, 4, 1))  # 返回一个张量，填充了在低（含）和高（不含）之间均匀生成的随机整数。

rs = torch.matmul(x, y)
print(f'rs =\n{rs}')
print(rs.shape)

x1 = torch.unsqueeze(x, 0)
print(x1, x1.shape)  # tensor([[1, 2, 3, 4]]) torch.Size([1, 4])

x2 = torch.tensor([[[1, 2, -1, 1]], [[1, 2, -1, 1]], [[1, 2, -1, 1]]])
print(x2.shape)  # torch.Size([3, 1, 4])

rs = torch.bmm(x2, y)
print(f'rs =\n{rs}')

rs1 = torch.squeeze(rs, -1)
print(rs1.shape)
```
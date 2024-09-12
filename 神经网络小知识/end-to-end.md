[toc]

# 如何理解深度学习领域中的端到端 （end to end）

[如何理解深度学习领域中的端到端 （end to end）](https://zhuanlan.zhihu.com/p/686887845)

## 端到端（end-to-end）

端到端的含义涉及到不同的领域。比如，在计算机科学和信息技术领域中，端到端的概念指的是一种通信方式，数据从发送方直接传输到接受方，而不需要中间环境对数据内容进行解析和处理。在通信领域内，端到端的模式强调的是数据传输过程中的直接性和完整性。

类似的，这个概念引申到深度学习和人工智能领域。端到端的概念表示**模型可以直接利用输入数据而不需要其他处理**。

因此，<u>端到端</u>或者<u>非端到端</u>，往往是形容一个模型对输入数据的要求。如果模型可以直接通过输入原始数据来得到输出，那么我们就说这个模型是端到端的（可以理解为从输入端直接到输出端的）。

与之相反的，传统机器学习方法往往不能直接利用原始数据，而需要提前对原始数据进行一定的处理。比如降维、[特征提取](https://zhida.zhihu.com/search?q=特征提取&zhida_source=entity&is_preview=1)等方法，那么这种方法就不能称之为端到端的学习方法。

例如，我们可以用下面的两张图直观表示端到端和非端到端：

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart LR
     原始数据-- 输入 ---> 端到端模型 -- 输出 ---> 输出结果


```

```mermaid
flowchart LR
     原始数据-- 数据处理 ---> 处理后的数据-- 输入 ---> 非端到端模型 -- 输出 ---> 输出结果
```

即，[端到端模型](https://zhida.zhihu.com/search?q=端到端模型&zhida_source=entity&is_preview=1)直接将原始数据作为输出返回输出结果，[非端到端模型](https://zhida.zhihu.com/search?q=非端到端模型&zhida_source=entity&is_preview=1)需要使用经过数据处理后的处理数据作为模型输入。

## 非端到端模型与端到端模型示例

首先先介绍数据处理的方法，也即对于非端到端模型需要去做的而端到端模型不需要做的步骤。

数据处理是机器学习和数据分析中的一个关键步骤，尤其是在非端到端模型中，这些步骤对于提高模型性能至关重要。以下是一些常见的数据处理方法，这些方法在非端到端模型中尤为重要，而在端到端模型中可能不那么必要或完全不需要：

1. **[特征工程](https://zhida.zhihu.com/search?q=特征工程&zhida_source=entity&is_preview=1)**：包括[特征选择](https://zhida.zhihu.com/search?q=特征选择&zhida_source=entity&is_preview=1)、特征提取和特征构造。这些步骤旨在从原始数据中识别、选择和转换出对模型预测最有用的信息。
2. **[数据清洗](https://zhida.zhihu.com/search?q=数据清洗&zhida_source=entity&is_preview=1)**：移除或填补缺失值、修正错误或不一致的数据、处理[异常值](https://zhida.zhihu.com/search?q=异常值&zhida_source=entity&is_preview=1)等，以确保数据的质量和准确性。
3. **数据转换**：将非数值型数据转换为数值型数据（如通过[独热编码](https://zhida.zhihu.com/search?q=独热编码&zhida_source=entity&is_preview=1)或标签编码），或者将连续数据离散化。
4. **[归一化](https://zhida.zhihu.com/search?q=归一化&zhida_source=entity&is_preview=1)/标准化**：调整数据的尺度，使其落在特定的范围内（如[0, 1]或均值为0，[标准差](https://zhida.zhihu.com/search?q=标准差&zhida_source=entity&is_preview=1)为1），以便于模型更好地学习。
5. **降维**：通过PCA、LDA等方法减少数据的维度，以减少计算复杂度和避免[过拟合](https://zhida.zhihu.com/search?q=过拟合&zhida_source=entity&is_preview=1)。
6. **去相关**：减少特征之间的相关性，避免[多重共线性](https://zhida.zhihu.com/search?q=多重共线性&zhida_source=entity&is_preview=1)问题，提高模型的稳定性。
7. **数据平衡**：在处理不平衡数据集时，通过过采样少数类别或[欠采样](https://zhida.zhihu.com/search?q=欠采样&zhida_source=entity&is_preview=1)多数类别来平衡各类别的样本数量。

在端到端模型中，如深度学习网络（例如CNN、RNN等），很多这些预处理步骤可以被简化或完全省略，因为这些模型能够自动从原始数据中学习到有用的特征表示。端到端模型的设计目标是直接从输入数据到输出结果，减少人为干预和预处理的需求。**然而，即使在端到端模型中，适当的[数据预处理](https://zhida.zhihu.com/search?q=数据预处理&zhida_source=entity&is_preview=1)仍然可能有助于提高模型的性能和训练效率。**

### 典型的端到端模型 - CNN

CNN，[卷积神经网络](https://zhida.zhihu.com/search?q=卷积神经网络&zhida_source=entity&is_preview=1)，算是我进入深度学习领域接触到的第一个应用级别的神经网络，他以及众多[神经网络模型](https://zhida.zhihu.com/search?q=神经网络模型&zhida_source=entity&is_preview=1)的端到端体现在于可以直接向模型输入原始图像，而不需要如提取特征这样的处理，根本原因在于，CNN 一连串的[隐藏层](https://zhida.zhihu.com/search?q=隐藏层&zhida_source=entity&is_preview=1)在不断训练和学习的过程中，已经学会了自动识别输入图像的特征，这也是深度学习神经网络里最强大的能力之一，就是抽象输入原始数据特征。

<img src="https://pic3.zhimg.com/80/v2-bfdda5c2e76e90dcd98896779899eb22_1440w.webp" alt="img" style="zoom:50%;" />

### 典型的非端到端模型 - SVM

[支持向量机](https://zhida.zhihu.com/search?q=支持向量机&zhida_source=entity&is_preview=1)（SVM）是一种监督学习方法，主要用于分类和回归任务。SVM 的过程主要包括数据预处理、选择[核函数](https://zhida.zhihu.com/search?q=核函数&zhida_source=entity&is_preview=1)、训练 SVM 模型、模型评估等方面。SVM使用核技巧来将数据映射到高维特征空间，以便能够找到一个[超平面](https://zhida.zhihu.com/search?q=超平面&zhida_source=entity&is_preview=1)来分隔数据。选择合适的核函数（如线性核、多项式核、[径向基函数](https://zhida.zhihu.com/search?q=径向基函数&zhida_source=entity&is_preview=1)（RBF）核等）是很重要的。而核函数的主要功能就是将原始数据转换为一个新的空间，使得数据的分布变得更加线性可分，从而简化 SVM 的优化问题。因此，SVM 不能直接用原始数据，需要数据处理过程，因此，他不是端到端的模型。

<img src="https://pic3.zhimg.com/80/v2-1f1201ea991ba4c27dab913deff4ee06_1440w.webp" alt="img" style="zoom:50%;" />

其余案例：

1. Nvidia的基于CNNs的[end-end](https://zhida.zhihu.com/search?q=end-end&zhida_source=entity&is_preview=1)自动驾驶，输入图片，直接输出steering angle。从视频来看效果拔群，但其实这个系统目前只能做简单的[follow lane](https://zhida.zhihu.com/search?q=follow+lane&zhida_source=entity&is_preview=1)，与真正的自动驾驶差距较大。亮点是证实了end-end在自动驾驶领域的可行性，并且对于数据集进行了augmentation。链接：[https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/](https://link.zhihu.com/?target=https%3A//devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
2. Google的paper: Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection，也可以算是end-end学习：输入图片，输出控制机械手移动的指令来抓取物品。这篇论文很赞，推荐：[https://arxiv.org/pdf/1603.02199v4.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1603.02199v4.pdf)
3. DeepMind神作Human-level control through deep reinforcement learning，其实也可以归为end-end，深度增强学习开山之作，值得学习：[http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html](https://link.zhihu.com/?target=http%3A//www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
4. Princeton大学有个Deep Driving项目，介于end-end和传统的[model based](https://zhida.zhihu.com/search?q=model+based&zhida_source=entity&is_preview=1)的自动驾驶之间，输入为图片，输出一些有用的affordance（实在不知道这词怎么翻译合适…）例如车身姿态、与前车距离、距路边距离等，然后利用这些数据通过公式计算所需的具体驾驶指令如加速、刹车、转向等。链接：[http://deepdriving.cs.princeton.edu/](https://link.zhihu.com/?target=http%3A//deepdriving.cs.princeton.edu/)

> 参考：[https://geek.csdn.net/65d6b0b3b8e5f01e1e4660c8.html](https://link.zhihu.com/?target=https%3A//geek.csdn.net/65d6b0b3b8e5f01e1e4660c8.html)

## 端到端学习的意义

端到端学习（End-to-End Learning）的意义在于它简化了机器学习模型的设计和训练过程，同时在许多情况下能够提高模型的性能和[泛化能力](https://zhida.zhihu.com/search?q=泛化能力&zhida_source=entity&is_preview=1)。以下是端到端学习的几个关键意义：

1. **简化流程**：端到端学习模型直接从原始输入数据到最终输出进行学习，无需复杂的特征工程或预处理步骤。这大大简化了模型的开发流程，减少了人工干预的需求。
2. **自动特征学习**：端到端模型能够自动发现并学习数据中的关键特征，这些特征对于完成特定任务（如分类、回归、序列预测等）是至关重要的。这种自动特征学习的能力减少了对领域专家知识的依赖。
3. **提高性能**：在许多应用中，端到端学习模型已经证明能够达到或超过传统机器学习模型的性能。特别是在图像识别、[自然语言处理](https://zhida.zhihu.com/search?q=自然语言处理&zhida_source=entity&is_preview=1)和语音识别等领域，深度学习模型（一种端到端学习的方法）已经取得了突破性的成果。
4. **泛化能力**：端到端学习模型通常具有较好的泛化能力，因为它们能够从大量数据中学习到复杂的模式和结构，而不是依赖于有限的、人为设计的特征。
5. **处理复杂数据**：端到端学习模型特别适合处理高维和复杂的数据类型，如图像、视频和音频数据。这些模型能够捕捉到数据中的细微差别和深层次的关系。
6. **[可扩展性](https://zhida.zhihu.com/search?q=可扩展性&zhida_source=entity&is_preview=1)**：端到端学习模型可以很好地扩展到大规模数据集和复杂任务，这在大数据时代尤为重要。
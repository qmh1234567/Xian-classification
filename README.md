## Xian-classification
### 目录说明
目录名称 | 含义
---|---
src_pytorch | 本地运行代码
src_pytorch_efficientE7 | 线上代码，使用efficientE7模型
src_pytorch_resnext101_32x8d | 线上代码，使用resnext101_32x8d模型
test_20 | 网上找的测试数据，每类20张

### 参考项目


### 模型提交过程
模型名称 | 主要内容  | 得分 
---|--- | --- 
model | Resnet50，修改图片预处理方式、等比缩放等 |90
model1 | 降低学习率到0.01，将resnet50改为resnext101_32x8d | 91.5
model2 | 在model1的基础上，修改预处理方法 | 94.2
model3 | 在model2的基础上，增加dropout 和最后一层 |95.2
model4 | 在model3的基础上,修改损失函数、添加注意力机制等 | 94.2
model5 | 使用effecientE4模型，参考别人的代码 |95.5
model6 | 更换成effecientE7模型 | 95.6
model7 | 过滤和新增数据集 | 95.7

### 模型训练心得

##### keras和tensorflow 
使用上次比赛的代码，修改了图片预处理方式，对图片进行了等比缩放，使得分从89提高到90.  
##### pytorch
1. 修改学习率从0.1降低至0.01 ,最后测试acc有91.5
2. 更换模型，将resnet50更换为resnext101_32x8d，注意加载模型的代码需要修改，不然总是从外网下载模型：
```
## 原代码：
model = models.__dict__[args.arch](pretrained=True)
# 加载模型结构
        model = models.__dict__[args.arch](pretrained=False)
        # 加载模型参数
        model.load_state_dict(torch.load(os.path.abspath('../pre-trained_model/pytorch/resnext101_32x8d-8ba56ff5.pth')))
```

3. 更换图片处理方式。注意这里的transformations是调用另一脚本的函数实现的。而且对应的也要更改推理文件的图片处理方式。

```
 train_dataset = datasets.ImageFolder(
        traindir,
        transformations['val_train'],
    )
    val_dataset = datasets.ImageFolder(
        valdir, 
        transformations['val_test'],
    )
```
二三步之后，acc=94.20

4. 修改模型：在最后一层之后新增几层。  
 之前在这个地方一直卡住，因为推理的时候提示有未知的层出现，之前一直以为是sequential层不对，后来才发现自己没有好好读代码，那个推理文件的代码直接加载的原来的网络模型，没有考虑新加的层，所以结构不对应。在推理脚本中应该这样修改：

```
self.model = models.__dict__['resnext101_32x8d']()
self.model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(256,self.num_classes)
)
```
最后的acc = 95.2（经过后面的多个步骤测试后，发现起作用的是dropout,没有它则会变到94.几）

5. epoch设置的小于5时，保存模型为空，这里需要调整代码，使其每个epoch之后都能保存最优的模型。否则不能快速的进行测试。
注意不是not is_best ，而是 is_best
```
def save_checkpoint(state, is_best, filename, args):
    if  is_best:
        # print("&"*100)
        torch.save(state, filename)
        if args.train_url.startswith('s3'):
            mox.file.copy(filename,
                          args.train_url + '/' + os.path.basename(filename))
            os.remove(filename)
```

6. 进行标签平滑 并修改loss函数  

由于pytorch的crossentropyloss只能接收非one-hot标签，而标签平滑又是对one-hot标签进行操作的，故需要重写交叉熵损失函数.

最后的acc=94.3 

7. 修改优化器，并采用双损失函数

acc也是94左右

9. 添加注意力机制
最后的acc 是94左右


10.在训练过程中，发现加入其他预处理方式，如mixup,cutout等，都会导致测试结果降低。这个目前还不清楚原因。

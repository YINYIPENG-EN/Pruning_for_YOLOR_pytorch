clone代码至本地后可以先测试一下：

```shell
python detect.py --source inference/images/horses.jpg --cfg cfg/yolor_csp.cfg --weights yolor_csp.pt --conf 0.25 --img-size 640 --device 0
```

# 训练自己的数据集

在工程下面中的dataset文件下放入自己的数据集。目录形式如下：

> dataset
>  |-- Annotations
>  |-- ImageSets
>  |-- images
>  |-- labels

Annotations是存放xml标签文件的，images是存放图像的，ImageSets存放四个txt文件【后面运行代码的时候会自动生成】，labels是将xml转txt文件。

1.运行makeTXT.py。这将会在ImageSets文件夹下生成 trainval.txt，test.txt，train.txt，val.txt四个文件【如果你打开这些txt文件，里面仅有图像的名字】。

2.**打开voc_label.py，并修改代码 classes**=[""]填入自己的类名，比如你的是训练猫和狗，那么就是classes=["dog","cat"]，然后运行该程序。此时**会在labels文件下**生成对应每个图像的txt文件，形式如下：【最前面的0是类对应的索引，我这里只有一个类，后面的四个数为box的参数，均归一化以后的，分别表示box的左上和右下坐标，等训练的时候会处理成center_x，center_y，w, h】

> ```
> 0 0.4723557692307693 0.5408653846153847 0.34375 0.8990384615384616
> 0 0.8834134615384616 0.5793269230769231 0.21875 0.8221153846153847 
> ```

 3.在data文件夹下新建一个mydata.yaml文件。内容如下【你也可以把coco.yaml复制过来】。

你只需要修改nc以及names即可，nc是类的数量，names是类的名字。

> ```
> train: ./dataset/train.txt
> val: ./dataset/val.txt
> test: ./dataset/test.txt
> 
> # number of classes
> nc: 1
> 
> # class names
> names: ['target']
> ```

4.以yolor_csp为例。打开cfg下的yolor_csp.cfg，搜索两个内容，搜索classes【有三个地方】，将classes修改为自己的类别数量 。再继续搜索255【6个地方】，这个255指的是coco数据集，为3 * (5 + 80)，如果是你自己的类，你需要自己计算一下,3*(5+你自己类的数量)。比如我这里是1个类，就是改成18.

5.在data/下新建一个myclasses.names文件，写入自己的类【这个是为了后面检测的时候读取类名】

6.终端输入参数，开始训练。

```bash
python train.py --weights yolor_csp.pt --cfg cfg/yolor_csp.cfg --data data/mydata.yaml --batch-size 8 --device 0
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

训练的权重会存储在当前工程下runs/train/exp的weights中。每次运行train.py的时候会在runs/train/下生成exp,exp1,exp2...为的防止权重的覆盖。 

# 检测推理

终端输入参数，开始检测。

```python
python detect.py --source 【你的图像路径】 --cfg cfg/yolor_csp.cfg --weights 【你训练好的权重路径】 --conf 0.2 --img-size 640 --device 0
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

------

# 剪枝

 在利用剪枝功能前，需要安装一下剪枝的库。需要安装0.2.7版本，0.2.8有些人说有问题。

> pip install torch_pruning==0.2.7

## 1.保存完整的权重文件

剪枝之前先要保存一下网络的权重和网络结构【非剪枝训练的权重仅含有权值，也就是通过torch.save(model.state_dict())形式保存的】。

修改tools/save_whole_model.py中的--weights，改为自己的训练后的权权重路径，修改--save_whole_model 为**True**，运行代码后会生成一个whole_model.pt，如果还想得到onnx模型，可以将--onnx设置为**True**。

## 2.网络剪枝

剪枝之前需要自己熟悉网络结构，也可以通过tools/printmodel.py 打印网络结构。

这里的剪枝操作支持两种类型：

### 1.单独卷积的剪枝

在Conv_pruning这个函数中，修改三个地方：

通过keys指筛选层：

> ```
> if k == 'module_list.22.Conv2d.weight':  # 筛选出该层 (根据自己需求)
> ```

amount是剪枝率，可以按续修改。 

> ```
> pruning_idxs = strategy(v, amount=0.4)  # or manually selected pruning_idxs=[2, 6, 9, ...]
> ```

修改DG.get_pruning_plan model参数，改为需要剪枝的层 

> ```
> # 放入要剪枝的层
> pruning_plan = DG.get_pruning_plan(model.module_list[22].Conv2d, tp.prune_conv, idxs=pruning_idxs)
> ```

我这里是仅对第22个卷积进行剪枝。看到如下参数有变化就是剪枝成功了，如果参数没变就说明你剪的不对【可以看到你参数变化不大，因为你仅仅对一个层剪枝了而已，当然变化不大】

```python
-------------
[ <DEP: prune_conv => prune_conv on module_list.22.Conv2d (Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False))>, Index=[1, 5, 8, 10, 11, 12, 13, 14, 15, 18, 20, 21, 22, 23, 25, 28, 32, 38, 40, 44, 52, 54, 55, 57, 58, 59, 61, 63, 64, 66, 72, 73, 77, 82, 84, 86, 88, 96, 97, 98, 99, 101, 102, 103, 109, 113, 114, 120, 123, 126, 127], NumPruned=6528]
[ <DEP: prune_conv => prune_batchnorm on module_list.22.BatchNorm2d (BatchNorm2d(128, eps=0.0001, momentum=0.03, affine=True, track_running_stats=True))>, Index=[1, 5, 8, 10, 11, 12, 13, 14, 15, 18, 20, 21, 22, 23, 25, 28, 32, 38, 40, 44, 52, 54, 55, 57, 58, 59, 61, 63, 64, 66, 72, 73, 77, 82, 84, 86, 88, 96, 97, 98, 99, 101, 102, 103, 109, 113, 114, 120, 123, 126, 127], NumPruned=102]
[ <DEP: prune_batchnorm => _prune_elementwise_op on _ElementWiseOp()>, Index=[1, 5, 8, 10, 11, 12, 13, 14, 15, 18, 20, 21, 22, 23, 25, 28, 32, 38, 40, 44, 52, 54, 55, 57, 58, 59, 61, 63, 64, 66, 72, 73, 77, 82, 84, 86, 88, 96, 97, 98, 99, 101, 102, 103, 109, 113, 114, 120, 123, 126, 127], NumPruned=0]
[ <DEP: _prune_elementwise_op => _prune_elementwise_op on _ElementWiseOp()>, Index=[1, 5, 8, 10, 11, 12, 13, 14, 15, 18, 20, 21, 22, 23, 25, 28, 32, 38, 40, 44, 52, 54, 55, 57, 58, 59, 61, 63, 64, 66, 72, 73, 77, 82, 84, 86, 88, 96, 97, 98, 99, 101, 102, 103, 109, 113, 114, 120, 123, 126, 127], NumPruned=0]
[ <DEP: _prune_elementwise_op => prune_related_conv on module_list.23.Conv2d (Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))>, Index=[1, 5, 8, 10, 11, 12, 13, 14, 15, 18, 20, 21, 22, 23, 25, 28, 32, 38, 40, 44, 52, 54, 55, 57, 58, 59, 61, 63, 64, 66, 72, 73, 77, 82, 84, 86, 88, 96, 97, 98, 99, 101, 102, 103, 109, 113, 114, 120, 123, 126, 127], NumPruned=58752]
65382 parameters will be pruned
-------------

2022-09-19 11:01:55.563 | INFO     | __main__:Conv_pruning:42 -   Params: 52497868 => 52432486

2022-09-19 11:01:56.361 | INFO     | __main__:Conv_pruning:55 - 剪枝完成
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

剪枝完在model_data/下会保存一个 Conv_pruning.pt权重。这个就是剪枝后的权重。

### 2.卷积层(某个模块)的剪枝

通过运行layer_pruning()函数。修改两个地方：

included_layers是需要剪枝的层，比如我这里是对前60层进行剪枝。

```python
included_layers = [layer.Conv2d for layer in model.module_list[:61] if
                      type(layer) is torch.nn.Sequential and layer.Conv2d]
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

修改amount剪枝率。 

```python
pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=strategy(m.weight, amount=0.9))
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

看到如下参数的变化说明剪枝成功了。 将会在model_data/下生成一个**layer_pruning.pt**。

> 2022-09-19 11:12:40.519 | INFO   | __main__:layer_pruning:81 - 
>  \-------------
>  [ <DEP: prune_conv => prune_conv on module_list.60.Conv2d (Conv2d(26, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))>, Index=[], NumPruned=0]
>  0 parameters will be pruned
>  \-------------
>
> 2022-09-19 11:12:40.522 | INFO   | __main__:layer_pruning:87 -  Params: 52497868 => 43633847
>
> 2022-09-19 11:12:41.709 | INFO   | __main__:layer_pruning:102 - 剪枝完成
>  
>
>  

 

------

## 3.剪枝后的微调训练 

 与上面的训练一样，只不过weights需要改为自己的剪枝后的权重路径，同时再加一个--pt参数，如下：

参数--pt：指剪枝后的训练

这里默认的epochs还是300，自己可以修改。

```bash
python train.py --weights model_data/layer_pruning.pt --cfg cfg/yolor_csp.cfg --data data/mydata.yaml --batch-size 8 --device 0 --pt
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

## 4.剪枝后的推理检测

 --weights 为剪枝后的权重，在加一个--pd表示剪枝后的检测。

```bash
python detect.py --source 【你的图像路径】 --cfg cfg/yolor_csp.cfg --weights 【剪枝的权重路径】 --conf 0.2 --img-size 640 --device 0 --pd
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

# 所遇问题：

## **1.剪枝训练期间在测mAP的时候报错**：

```python
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

解决办法：在models/modes.py中的第416行和417行，强行修改 :

> ```
> io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
> io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
> ```
>
> 改为：
>
> ```
> io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid.cuda())
> io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh.cuda()
> ```

在训练完需要推理测试的时候需要把上面的再改回去(不是不能检测，只是会特别的不准)
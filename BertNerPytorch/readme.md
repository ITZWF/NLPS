## pytorch版NER, 基于bert+crf实现

#### 模型结构

![image](https://user-images.githubusercontent.com/42050378/117459934-86f6e680-af7e-11eb-9c81-63eb8e3c8cbe.png)

  - 训练数据准备在pick???.py下, 是标准的ner训练数据输入格式
  - 训练是train???.py
  - 解码是eval???.py
  - 参考的是一个大神的crf代码

### 一个简单的bert的文本分类代码

数据集选择的THUCNews，自行下载并整理出10w条数据。news_title_dataset.csv链接：https://pan.baidu.com/s/1jok0ohyTVBn2HueZrynKng

提取码：7941

下载预训练模型权重，这里下载的是 chinese_roberta_wwm_ext_pytorch 。bert_model链接: https://pan.baidu.com/s/1I9-I0ukHnDYQMvHM_uFrgA 

提取码: a6ud


'''

 Test Accuracy = 0.9736 

              precision    recall  f1-score   support

           0     1.0000    0.9939    0.9970       990
           1     0.9764    0.9910    0.9836      1001
           2     0.9788    0.9826    0.9807      1033
           3     0.9784    0.9636    0.9709       988
           4     0.9687    0.9851    0.9768      1005
           5     0.9888    0.9888    0.9888       985
           6     0.9688    0.9688    0.9688       962
           7     0.9649    0.9668    0.9658      1023
           8     0.9544    0.9618    0.9581      1022
           9     0.9576    0.9334    0.9453       991

    accuracy                         0.9736     10000
   macro avg     0.9737    0.9736    0.9736     10000
weighted avg     0.9736    0.9736    0.9736     10000

'''

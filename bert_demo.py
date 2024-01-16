import pandas as pd 
import numpy as np 
import json, time 
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, get_cosine_schedule_with_warmup
from transformers import AdamW
import warnings
from typing import Union, Optional
warnings.filterwarnings('ignore')

bert_path = "/root/yunzhi/retrieval/bert_model"    # 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
tokenizer = BertTokenizer.from_pretrained(bert_path)   # 初始化分词器
BATCH_SIZE = 64
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5

# 定义模型
class Bert_Model(nn.Module):
    def __init__(self,  bert_path, num_classes=10):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path) # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path) # 导入预训练模型权重 
        self.fc = nn.Linear(self.config.hidden_size, num_classes) # 分类器

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1] # 池化后的输出 [bs, config.hidden_size]
        logit = self.fc(out_pool) # 分类器
        return logit

def data_processor(data_path: str)->Union[list, list, list, list]:
    """
    :param data: 原始数据
    :return: 返回处理后的数据, 包括input_ids, input_mask, token_type_id, label
    """
    # 输出预处理
    input_ids, input_masks, input_types = [], [], []
    label = []
    maxlen = 30

    with open('./news_title_dataset.csv', encoding='utf-8') as f:

        for i, line in tqdm(enumerate(f)):
            title, y = line.strip().split('\t')

            # encode_plus会输出一个字典，分别为'input_ids', 'token_type_ids', 'attention_mask'对应的编码
            # 根据参数会短则补齐，长则切断
            encoder_dict = tokenizer.encode_plus(
                text=title,
                max_length=maxlen,
                padding='max_length',
                truncation=True
            )
            input_ids.append(encoder_dict['input_ids'])
            input_masks.append(encoder_dict['attention_mask'])
            input_types.append(encoder_dict['token_type_ids'])
            label.append(int(y))

    input_ids, input_types, input_masks, label = np.array(input_ids), np.array(input_types), np.array(input_masks), np.array(label)
    print(input_ids.shape, input_types.shape, input_masks.shape, label.shape)
    return input_ids, input_types, input_masks, label


# 切分训练集、验证集、测试集
def data_split_and_build_loader(input_ids: list, 
                                input_types: list, 
                                input_masks: list, 
                                label: list) -> Union[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], list]:
    """
    :param input_ids: 输入的id
    :param input_types: token_type_id
    :param input_masks: attention_mask
    :param label: 标签
    :return: 返回训练集、验证集、测试集， y_test
    """
    # 随机打乱索引
    idxes = np.arange(input_ids.shape[0])
    np.random.seed(1234)
    np.random.shuffle(idxes)
    print(idxes.shape, idxes[:10])

    input_ids_train, input_ids_valid, input_ids_test= input_ids[idxes[:80000]], input_ids[idxes[80000:90000]], input_ids[idxes[90000:]]
    input_types_train, input_types_valid, input_types_test = input_types[idxes[:80000]], input_types[idxes[80000:90000]], input_types[idxes[90000:]]
    input_masks_train, input_masks_valid, input_masks_test = input_masks[idxes[:80000]], input_masks[idxes[80000:90000]], input_masks[idxes[90000:]]
    y_train, y_valid, y_test = label[idxes[:80000]], label[idxes[80000:90000]], label[idxes[90000:]]

    print(input_ids_train.shape, y_train.shape, input_ids_valid.shape, y_valid.shape, 
        input_ids_test.shape, y_test.shape)
    
    # 训练集
    trian_dataset = TensorDataset(torch.LongTensor(input_ids_train),
                                torch.LongTensor(input_masks_train),  
                                torch.LongTensor(input_types_train), 
                                torch.LongTensor(y_train))
    train_sampler = RandomSampler(trian_dataset)
    train_dataloader = DataLoader(trian_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

    # 验证集
    valid_dataset = TensorDataset(torch.LongTensor(input_ids_valid), 
                                torch.LongTensor(input_masks_valid), 
                                torch.LongTensor(input_types_valid), 
                                torch.LongTensor(y_valid))

    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)

    # 测试集
    test_dataset = TensorDataset(torch.LongTensor(input_ids_test), 
                                torch.LongTensor(input_masks_test), 
                                torch.LongTensor(input_types_test))
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

    return train_dataloader, valid_dataloader, test_dataloader, y_test
    
# 实例化bert模型
def get_parameter_number(model: Optional[BertModel]) -> None:
    """
    :param model: 模型
    :return: 返回模型参数数量
    """
    # 打印模型参数数量，包括总训练的和不可训练的
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters:{}, Trainable parameters:{}'.format(total_num, trainable_num)

# 定义训练函数和验证测试函数
def evaluate(model: Optional[BertModel], data_loader: Optional[DataLoader], device: str) -> float:
    """
    :param model: 模型
    :param data_loader: 数据加载器
    :param device: 设备
    :return: 返回accuracy
    """
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, (ids, att, tpe, y) in (enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(y.squeeze().cpu().numpy().tolist())
    
    return accuracy_score(val_true, val_pred) # 返回accuracy

# 测试集没有标签，需要预测提交
def predict(model: Optional[BertModel], 
            data_loader: Optional[DataLoader], 
            device: str) -> list:
    """
    :param model: 模型
    :param data_loader: 数据加载器
    :param device: 设备
    :return: 返回预测结果
    """
    model.eval()
    val_pred = []
    with torch.no_grad():
        for idx, (ids, att, tpe) in (enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
    return val_pred

def trian_and_eval(model: Optional[BertModel], 
                train_loader: Optional[DataLoader], 
                valid_loader: Optional[DataLoader],   
                optimizer: Optional[AdamW], 
                scheduler: Optional[get_cosine_schedule_with_warmup], 
                device: str, 
                epoch: int) -> None:    
    best_acc = 0.0
    patience = 0
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        """训练模型"""
        start = time.time()
        model.train()
        print("***** Running training epoch {} *****".format(i+1))
        train_loss_sum = 0.0
        for idx, (ids, att, tpe, y) in (enumerate(train_loader)):
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            y_pred = model(ids, att, tpe)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step() # 学习率变化

            train_loss_sum += loss.item()
            if (idx + 1) % (len(train_loader)//5) ==0: # 只打印五次结果
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}s".format(
                    i + 1, idx + 1, len(train_loader), train_loss_sum/(idx + 1), time.time() - start))
                
        """验证模型"""
        model.eval()
        acc = evaluate(model, valid_loader, device) # 验证模型的性能

        #保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'the_best_bert.pth')
        
        print("current acc is {:.4f}, best acc is {:.4f}".format(acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))

if __name__ == '__main__':
    input_ids, input_types, input_masks, label = data_processor('./news_title_dataset.csv')
    train_dataloader, valid_dataloader, test_dataloader ,y_test = data_split_and_build_loader(input_ids, input_types, input_masks, label)
    
    # 实例化bert模型
    model = Bert_Model(bert_path).to(DEVICE)
    print(get_parameter_number(model))

    # 定义优化器
    # 学习率先线性warmup一个epoch，然后cosine式下降。
    # 这里给个小提示，一定要加warmup（学习率从0慢慢升上去），如果把warmup去掉，可能收敛不了。
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=len(train_dataloader), 
                                                num_training_steps=len(train_dataloader)*EPOCHS)
    
    # 训练和验证模型
    trian_and_eval(model, train_dataloader, valid_dataloader, optimizer, scheduler, DEVICE, EPOCHS)

    # 加载最优权重对测试集进行测试
    model.load_state_dict(torch.load('the_best_bert.pth'))
    pred_test = predict(model, test_dataloader, DEVICE)
    print("\n Test Accuracy = {} \n".format(accuracy_score(y_test, pred_test)))
    print(classification_report(y_test, pred_test, digits=4))
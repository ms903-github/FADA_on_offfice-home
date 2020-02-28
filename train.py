import argparse
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sch
import torchvision.transforms as transforms
import os
import sys
import time
import yaml
import shutil

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from addict import Dict
from sklearn.metrics import f1_score
from torchvision.transforms import Compose, ToTensor, RandomResizedCrop
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, Normalize

from libs import *

#パラメタの取得
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path of config file")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--csv_path", type=str, default="./csv/csv65")
    return parser.parse_args()
   
#device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_g(dataloader, model_g, model_h, criterion, optimizer, epoch, metric=False):
    # 平均を計算してくれるクラス
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # 進捗状況を表示してくれるクラス
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch)
    )

    gts = []
    preds = []

    model_g.train()
    model_h.train()

    end = time.time()
    #エポック開始
    for i, sample in enumerate(dataloader):
        #ロード時間を計測
        data_time.update(time.time() - end)
        #推論
        data, label = sample
        label = torch.tensor(label, dtype=torch.long)
        data, label = data.to(device), label.to(device)
        _, _, feat = model_g(data)
        if metric:
            pred = model_h(feat, label)
        else:
            pred = model_h(feat)
        loss = criterion(pred, label)
        #ロス・精度を記録
        acc1 = accuracy(pred, label, topk=(1,))
        tmp_batchsize = data.shape[0]
        losses.update(loss.item(), n=tmp_batchsize)
        top1.update(acc1[0].item(), n=tmp_batchsize)
        #推論結果・正解ラベルを記録
        _, predict = pred.max(dim=1)
        gts += list(label.to("cpu").numpy())
        preds += list(predict.to("cpu").numpy())
        #逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #計算時間を計測
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and i % 10 == 0:
            progress.display(1)

    f1s = f1_score(gts, preds, average="macro")
    return(losses.avg, top1.avg, f1s)

def train_DCD(dataloader, model_DCD, model_g, criterion, optimizer, epoch, dcd_mode="l"):
    # 平均を計算してくれるクラス
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # 進捗状況を表示してくれるクラス
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch)
    )

    gts = []
    preds = []

    model_DCD.train()
    model_g.eval()

    end = time.time()
    for i, sample in enumerate(dataloader):
        #ロード時間を計測
        data_time.update(time.time() - end)
        #推論
        data1, data2, label = sample["image1"], sample["image2"], sample["label"]
        data1, data2, label = data1.to(device), data2.to(device), label.to(device)
        feat1_h, feat1_m, feat1_l = model_g(data1)
        feat2_h, feat2_m, feat2_l = model_g(data2)
        if dcd_mode == "h":
            data = torch.cat([feat1_h, feat2_h], dim=1)
        if dcd_mode == "m":
            data = torch.cat([feat1_m, feat2_m], dim=1)
        if dcd_mode == "l":
            data = torch.cat([feat1_l, feat2_l], dim=1)
        pred = model_DCD(data)
        loss = criterion(pred, label)
        #ロス・精度を記録
        acc1 = accuracy(pred, label, topk=(1,))
        tmp_batchsize = data.shape[0]
        losses.update(loss.item(), tmp_batchsize)
        top1.update(acc1[0].item(), tmp_batchsize)
        #推論結果・正解ラベルを記録
        _, prediction = pred.max(dim=1)
        gts += list(label.to("cpu").numpy())
        preds += list(prediction.to("cpu").numpy())
        #逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #計算時間を計測
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and i % 10 == 0:
            progress.display(1)

    f1s = f1_score(gts, preds, average="macro")
    return(losses.avg, top1.avg, f1s)

#DCDを固定、g,hをtrainする関数

def train_adv_gh(dataloader, model_DCD, model_g, model_h, criterion_gh, criterion_DCD, optimizer, epoch, dcd_mode="l", alpha=1, beta=3, gamma=0.3, metric=False):
    # 平均を計算してくれるクラス
    batch_time_g = AverageMeter('Time_g', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_g = AverageMeter('Loss_g', ':.4e')
    top1_g_s = AverageMeter('Acc1_g@source', ':6.2f')
    top1_g_t = AverageMeter('Acc1_g@target', ':6.2f')
    
    # 進捗状況を表示してくれるクラス
    progress = ProgressMeter(
        len(dataloader),
        [batch_time_g, data_time, losses_g, top1_g_s, top1_g_t],
        prefix="Epoch: [{}]".format(epoch)
    )

    model_g.train().to(device)
    model_h.train().to(device)
    model_DCD.to(device)     #本当はここは.eval()にしなければいけない
    
    end = time.time()

    for i, sample in enumerate(dataloader):
        #ロード時間を計測
        data_time.update(time.time() - end)
        #推論
        data1, data2, label, s_data, s_label, t_data, t_label = [
            sample["image1"].to(device),
            sample["image2"].to(device), 
            sample["label"].to(device), 
            sample["s_traindata"].to(device), 
            sample["s_trainlabel"].to(device), 
            sample["t_traindata"].to(device), 
            sample["t_trainlabel"].to(device)
        ]
        #DCDに判別されるロス
        feat1_h, feat1_m, feat1_l = model_g(data1)
        feat2_h, feat2_m, feat2_l = model_g(data2)
        if dcd_mode == "h":
            data = torch.cat([feat1_h, feat2_h], dim=1)
        if dcd_mode == "m":
            data = torch.cat([feat1_m, feat2_m], dim=1)
        if dcd_mode =="l":
            data = torch.cat([feat1_l, feat2_l], dim=1)
        pred = model_DCD(data)
        loss_DCD = criterion_DCD(pred, label)
        #ソースにおける分類ロス
        _, _, feat = model_g(s_data)
        if metric:
            s_pred = model_h(feat, s_label)
        else:
            s_pred = model_h(feat)
        s_loss = criterion_gh(s_pred, s_label)
        #ターゲットにおける分類ロス
        _, _, feat = model_g(t_data)
        if metric:
            t_pred = model_h(feat, t_label)               
        else:
            t_pred = model_h(feat)
        t_loss = criterion_gh(t_pred, t_label)
        
        #全ロス
        loss = alpha * s_loss + beta * t_loss + gamma * loss_DCD
        #記録
        acc_s = accuracy(s_pred, s_label, topk=(1,))
        acc_t = accuracy(t_pred, t_label, topk=(1,))
        tmp_s_batchsize = s_data.shape[0]
        tmp_t_batchsize = t_data.shape[0]
        top1_g_s.update(acc_s[0].item(), tmp_s_batchsize)
        top1_g_t.update(acc_t[0].item(), tmp_t_batchsize)
        losses_g.update(loss.item(), tmp_s_batchsize)
        #逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #計算時間を計測
        batch_time_g.update(time.time() - end)
        end = time.time()
    
    return(losses_g.avg, top1_g_s.avg, top1_g_t.avg)

#metricモジュール
def train_g_metric(model_g, s_dataloader, t_dataloader, optimizer, n_class, metric="l1_ave"):
    optimizer.zero_grad()
    if metric == "l1_ave":
        smloss = dist_distance(model_g, s_dataloader, t_dataloader, n_class, dist=metric)
        smloss.backward()
    elif metric == "anchor":
        s_feat_l = []
        s_label_l = []
        for data, label in s_dataloader:
            _, _, s_feat = model_g(data.to(device))
            s_feat_l.append(s_feat)
            s_label_l.append(label)
        s_feats = torch.cat(s_feat_l)
        s_labels = torch.cat(s_label_l)
        s_kmeans = KMeans(n_clusters=n_class)
        s_kmeans.fit(s_feats.detach().cpu().numpy())
        s_centers = s_kmeans.cluster_centers_
        s_centers = torch.FloatTensor(s_centers).to(device)
        for data, label in t_dataloader:
            _, _, t_feat = model_g(data.to(device))
            loss = (s_centers[int(label)] - t_feat).pow(2).sum(1)
            loss /= len(data)
            loss.backward()
    
    optimizer.step()


#MCDモジュール
def train_h_MCD(model_g, model_h1, model_h2, s_dataloader, t_dataloader, optimizer_g, optimizer_h1, optimizer_h2, loss_func, discrepancy):
    for data_t, label_t in t_dataloader:
        #ソースによるg, h の更新
        data_s, label_s = iter(s_dataloader).next()
        data_s, label_s, data_t, label_t = data_s.to(device), label_s.to(device), data_t.to(device), label_t.to(device)
        optimizer_g.zero_grad()
        optimizer_h1.zero_grad()
        optimizer_h2.zero_grad()
        pred1 = model_h1(model_g(data_s))
        loss1 = loss_func(pred1, label_s)
        pred2 = model_h2(model_g(data_s))
        loss2 = loss_func(pred2, label_s)
        loss_sum = loss1 + loss2
        loss_sum.backward()
        optimizer_g.step()
        optimizer_h1.step()
        optimizer_h2.step()
        
        #h1, h2更新(discrepancy使用)
        optimizer_g.zero_grad()
        optimizer_h1.zero_grad()
        optimizer_h2.zero_grad()
        pred1 = model_h1(model_g(data_s))
        loss1 = loss_func(pred1, label_s)
        pred2 = model_h2(model_g(data_s))
        loss2 = loss_func(pred2, label_s)
        loss_sum = loss1 + loss2
        loss_dis = discrepancy(pred1, pred2)
        loss = loss_sum - loss_dis
        loss.backward()
        optimizer_h1.step()
        optimizer_h2.step()
        optimizer_h1.zero_grad()
        optimizer_h2.zero_grad()
        
        #G更新(num_k : h更新につきgを何回更新するか)
        num_k = 3
        for i in range(num_k):
            pred1 = model_h1(model_g(data_t))
            pred2 = model_h2(model_g(data_t))
            loss_dis = discrepancy(pred1, pred2)
            loss_dis.backward()
            optimizer_g.step()
            optimizer_g.zero_grad()


def validate(s_dataloader, t_dataloader, model_g, model_h, criterion, metric=False):
    s_losses = AverageMeter("s_loss", ":.4e")
    t_losses = AverageMeter("t_loss", ":.4e")
    s_acc = AverageMeter("s_acc", ":6.2f")
    t_acc = AverageMeter("t_acc", ":6.2f")

    s_gts = []
    t_gts = []
    s_preds = []
    t_preds = []

    model_g.eval().to(device)
    model_h.eval().to(device)

    with torch.no_grad():
        for i, sample in enumerate(s_dataloader):
            data, label = sample
            label = torch.tensor(label, dtype=torch.long)
            data, label = data.to(device), label.to(device)
            tmp_batchsize = data.shape[0]
            _, _, feat = model_g(data)
            if metric:
                pred = model_h(feat, label)
            else:
                pred = model_h(feat)
            loss = criterion(pred, label)

            acc = accuracy(pred, label, topk=(1,))
            s_losses.update(loss.item(), tmp_batchsize)
            s_acc.update(acc[0].item(), tmp_batchsize)

            _, prediction = pred.max(dim=1)
            s_gts += list(label.to("cpu").numpy())
            s_preds += list(prediction.to("cpu").numpy())

    with torch.no_grad():
        for i, sample in enumerate(t_dataloader):
            data, label = sample
            label = torch.tensor(label, dtype=torch.long)
            data, label = data.to(device), label.to(device)
            tmp_batchsize = data.shape[0]
            _, _, feat = model_g(data)
            if metric:
                pred = model_h(feat, label)
            else:
                pred = model_h(feat)
            loss = criterion(pred, label)

            acc = accuracy(pred, label, topk=(1,))
            t_losses.update(loss.item(), tmp_batchsize)
            t_acc.update(acc[0].item(), tmp_batchsize)

            _, prediction = pred.max(dim=1)
            t_gts += list(label.to("cpu").numpy())
            t_preds += list(prediction.to("cpu").numpy())

    s_f1 = f1_score(s_gts, s_preds, average="macro")
    t_f1 = f1_score(t_gts, t_preds, average="macro")

    return s_losses.avg, t_losses.avg, s_acc.avg, t_acc.avg, s_f1, t_f1

def main():
    args = get_arguments()

    CONFIG = Dict(yaml.safe_load(open(os.path.join(args.config, "config.yaml"))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #データロード
    # resize_tensor = transforms.Compose([
    #     transforms.RandomResizedCrop(size=(CONFIG.height, CONFIG.width)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=get_mean(), std=get_std())
    # ])
    resize_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=get_mean(), std=get_std())
    ])

    batch_size = CONFIG.batch_size
    num_workers = CONFIG.num_workers
    n_classes = CONFIG.n_classes
    s_trainset = load_pict2(os.path.join(args.csv_path, "Product_train.csv"), transform = resize_tensor)
    s_testset = load_pict2(os.path.join(args.csv_path, "Product_test.csv"), transform = resize_tensor)
    t_trainset = load_pict2(os.path.join(args.csv_path, "RealWorld_few.csv"), transform = resize_tensor)
    t_testset = load_pict2(os.path.join(args.csv_path, "RealWorld_test.csv"), transform = resize_tensor)
    s_trainloader = DataLoader(s_trainset, batch_size = batch_size, shuffle = True, num_workers=num_workers, pin_memory=True, drop_last=True)
    s_testloader = DataLoader(s_testset, batch_size = batch_size, shuffle = True,  num_workers=num_workers, pin_memory=True)
    t_trainloader = DataLoader(t_trainset, batch_size = batch_size, shuffle = True,  num_workers=num_workers, pin_memory=True, drop_last=True)
    t_testloader = DataLoader(t_testset, batch_size = batch_size, shuffle = True,  num_workers=num_workers, pin_memory=True)
    
    #DCD用のデータロード
    dataset = load_pair(os.path.join(args.csv_path, "G1.csv"), os.path.join(args.csv_path, "G2.csv"), os.path.join(args.csv_path, "G3.csv"), os.path.join(args.csv_path, "G4.csv"), transform = resize_tensor)
    DCD_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #g,hの敵対訓練用のデータロード
    adv_dataset = load_pair(os.path.join(args.csv_path, "adv_G2.csv"), os.path.join(args.csv_path, "adv_G4.csv"), transform=resize_tensor)
    cat_dataset = concat_dataset(adv_dataset, s_trainset, t_trainset)
    adv_dataloader = DataLoader(cat_dataset, batch_size=batch_size, shuffle=True)
    
    #モデル、更新器を定義
    #g
    if CONFIG.model == "googlenet":
        net_g = Encoder_google().to(device)
    elif CONFIG.model == "resnet":
        net_g = Encoder_resnet().to(device)
    #h
    if CONFIG.metric == "arc":
        net_h = classifier_arc(n_classes).to(device)
    elif CONFIG.metric == "cos":
        net_h = classifier_cos(n_classes).to(device)
    elif CONFIG.metric == "sphere":
        net_h = classifier_sphere(n_classes).to(device)
    elif CONFIG.metric == "adacos":
        net_h = classifier_adacos(n_classes).to(device)
    else:
        net_h = classifier(n_classes).to(device)
    optimizer_gh = optim.Adam(list(net_g.parameters()) + list(net_h.parameters()), lr=CONFIG.lr_gh)
    if CONFIG.MCD:
        net_h2 = classifier(n_classes).to(device)
        optimizer_gh2 = optim.Adam(list(net_g.parameters()) + list(net_h2.parameters()), lr=CONFIG.lr_gh)
    #scheduler
    if CONFIG.scheduler == "onplateau":
        scheduler_gh = sch.ReduceLROnPlateau(optimizer_gh)
    #DCD
    if CONFIG.DCD_h_activate:
        net_DCD_h = DCD_head().to(device)
        optimizer_DCD_h = optim.Adam(net_DCD_h.parameters(), lr=CONFIG.lr_DCD)    
    if CONFIG.DCD_m_activate:
        net_DCD_m = DCD_middle().to(device)
        optimizer_DCD_m = optim.Adam(net_DCD_m.parameters(), lr=CONFIG.lr_DCD)
    if CONFIG.DCD_l_activate:
        net_DCD_l = DCD_last().to(device)
        optimizer_DCD_l = optim.Adam(net_DCD_l.parameters(), lr=CONFIG.lr_DCD)
    
    #resumerは今回は割愛

    #損失関数
    if CONFIG.class_weight:
        criterion_gh = nn.CrossEntropyLoss(
            weight=get_class_weight(n_classes=n_classes).to(device)
        )
    else:
        criterion_gh = nn.CrossEntropyLoss()

    criterion_DCD = nn.CrossEntropyLoss()

    print("\n----------------start training-----------------")

    #動かすパラメタの設定
    reqg_config(net_g, CONFIG.model, CONFIG, reqg_lastonly=CONFIG.reqg_init_lastonly)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
        
    print("\nstep1 : training g&h on source\n")
    #ログ定義
    best_acc1 = 0
    log = pd.DataFrame(
        columns=[
            'epoch', 
            'train_loss', 's_val_loss', 't_val_loss',
            'train_acc', 's_val_acc', 't_val_acc',
            'train_f1', 's_val_f1s', 't_val_f1s'
        ]
    )
    for epoch in range(CONFIG.num_ep_init):
        #train
        train_loss, train_acc, train_f1 = train_g(s_trainloader, net_g, net_h, criterion_gh, optimizer_gh, epoch, metric=CONFIG.metric)
        #eval
        s_val_loss, t_val_loss, s_val_acc, t_val_acc, s_val_f1s, t_val_f1s = validate(
            s_testloader, t_testloader, net_g, net_h, criterion_gh, metric=CONFIG.metric
            )

        #精度が良ければモデル保存
        if best_acc1 < s_val_acc:
            best_acc1 = s_val_acc
            torch.save(
                net_g.state_dict(),
                os.path.join(CONFIG.result_path, 'best_acc1_model_g.prm')
            )
            torch.save(
                net_h.state_dict(),
                os.path.join(CONFIG.result_path, 'best_acc1_model_h.prm')
            )
        #記録
        tmp = pd.Series([
            epoch,
            train_loss, s_val_loss, t_val_loss,
            train_acc, s_val_acc, t_val_acc,
            train_f1, s_val_f1s, t_val_f1s
        ], index=log.columns
        )

        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(CONFIG.result_path, 'log_step1.csv'), index=False)
        
        print(
            'epoch: {}\ttrain loss: {:.4f}\ts_val loss: {:.4f}\tt_val loss: {:.4f}\ts_val_acc: {:.5f}\tt_val_acc: {:.5f}\ts_val_f1s: {:.5f}\tt_val_f1s: {:.5f}'
            .format(epoch, train_loss,
                    s_val_loss, t_val_loss,
                    s_val_acc, t_val_acc, 
                    s_val_f1s, t_val_f1s)
        )
        if CONFIG.scheduler:
            scheduler_gh.step(s_val_loss)

    print("\nstep2 : training DCD\n")
    #モデルロード
    net_g.load_state_dict(torch.load(os.path.join(CONFIG.result_path, 'best_acc1_model_g.prm')))
    net_h.load_state_dict(torch.load(os.path.join(CONFIG.result_path, 'best_acc1_model_h.prm')))

    #ログ定義
    best_acc1_h, best_acc1_m, best_acc1_l = 0, 0, 0
    log = pd.DataFrame(
        columns=[
            'epoch', 
            'train_loss_h', "train_acc_h", "train_f1_h",
            'train_loss_m', "train_acc_m", "train_f1_m",
            'train_loss_l', "train_acc_l", "train_f1_l"
        ]
    )
    for epoch in range(CONFIG.num_ep_DCD):
        train_loss_h, train_acc_h, train_f1_h, train_loss_m, train_acc_m, train_f1_m, train_loss_l, train_acc_l, train_f1_l = 0, 0, 0, 0, 0, 0, 0, 0, 0
        #train
        if CONFIG.DCD_h_activate:
            train_loss_h, train_acc_h, train_f1_h = train_DCD(DCD_dataloader, net_DCD_h, net_g, criterion_DCD, optimizer_DCD_h, epoch, dcd_mode="h")
            #精度が良ければモデル保存
            if best_acc1_h < train_acc_h:
                best_acc1_h = train_acc_h
                torch.save(net_DCD_h.state_dict(),os.path.join(CONFIG.result_path, 'best_acc1_model_DCD_h.prm'))
        if CONFIG.DCD_m_activate:
            train_loss_m, train_acc_m, train_f1_m = train_DCD(DCD_dataloader, net_DCD_m, net_g, criterion_DCD, optimizer_DCD_m, epoch, dcd_mode="m")
            #精度が良ければモデル保存
            if best_acc1_m < train_acc_m:
                best_acc1_m = train_acc_m
                torch.save(net_DCD_m.state_dict(),os.path.join(CONFIG.result_path, 'best_acc1_model_DCD_m.prm'))    
        if CONFIG.DCD_l_activate:
            train_loss_l, train_acc_l, train_f1_l = train_DCD(DCD_dataloader, net_DCD_l, net_g, criterion_DCD, optimizer_DCD_l, epoch, dcd_mode="l")
            #精度が良ければモデル保存
            if best_acc1_l < train_acc_l:
                best_acc1_l = train_acc_l
                torch.save(net_DCD_l.state_dict(),os.path.join(CONFIG.result_path, 'best_acc1_model_DCD_l.prm'))
            
        #記録
        tmp = pd.Series([
            epoch,
            train_loss_h, train_acc_h, train_f1_h,
            train_loss_m, train_acc_m, train_f1_m,
            train_loss_l, train_acc_l, train_f1_l
        ], index=log.columns
        )

        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(CONFIG.result_path, 'log_step2.csv'), index=False)
        print(
            'epoch: {}\ttrain acc l: {:.4f}\ttrain acc m: {:.5f}\ttrain acc h: {:.5f}'
            .format(epoch, train_acc_l, train_acc_m, train_acc_h)
        )    
        
            
    print("\nstep3 : adversarial training of g,h and DCD\n")

    #モデルロード
    net_g.load_state_dict(torch.load(os.path.join(CONFIG.result_path, 'best_acc1_model_g.prm')))
    
    net_h.load_state_dict(torch.load(os.path.join(CONFIG.result_path, 'best_acc1_model_h.prm')))
    if CONFIG.MCD:
        net_h2.load_state_dict(torch.load(os.path.join(CONFIG.result_path, 'best_acc1_model_h.prm')))
    
    if CONFIG.DCD_h_activate:
        net_DCD_h.load_state_dict(torch.load(os.path.join(CONFIG.result_path, 'best_acc1_model_DCD_h.prm')))
    if CONFIG.DCD_m_activate:
        net_DCD_m.load_state_dict(torch.load(os.path.join(CONFIG.result_path, 'best_acc1_model_DCD_m.prm')))
    if CONFIG.DCD_l_activate:
        net_DCD_l.load_state_dict(torch.load(os.path.join(CONFIG.result_path, 'best_acc1_model_DCD_l.prm')))
    
    #動かすパラメタの設定
    reqg_config(net_g, CONFIG.model, CONFIG, reqg_lastonly=False)
    
    #ログ定義
    log = pd.DataFrame(
        columns=[
            'epoch', 
            'train_loss', 's_val_loss', 't_val_loss',
            's_tr_acc', 't_tr_acc',
            's_val_acc', 't_val_acc',
            's_val_f1s', 't_val_f1s',
            "DCD_loss", "DCD_acc", "DCD_f1s"
        ]
    )
    for epoch in range(CONFIG.num_ep_train):
        #metric train g if dist_align==True
        if CONFIG.dist_align:
            train_g_metric(net_g, s_trainloader, t_trainloader, optimizer_gh, n_classes, metric=CONFIG.dist_method)

        #train g,h
        for k in range(CONFIG.num_k):
            if CONFIG.MCD:
                if CONFIG.DCD_l_activate:
                    train_loss, s_tr_acc, t_tr_acc = train_adv_gh(
                        adv_dataloader, net_DCD_l, net_g, net_h, criterion_gh, criterion_DCD, optimizer_gh, epoch, dcd_mode="l",
                        alpha=CONFIG.alpha, beta=CONFIG.beta, gamma=CONFIG.gamma, metric=CONFIG.metric)    
                    train_loss, s_tr_acc, t_tr_acc = train_adv_gh(
                        adv_dataloader, net_DCD_l, net_g, net_h2, criterion_gh, criterion_DCD, optimizer_gh2, epoch,
                        alpha=CONFIG.alpha, beta=CONFIG.beta, gamma=CONFIG.gamma, metric=CONFIG.metric)
                if CONFIG.DCD_m_activate:
                    train_loss, s_tr_acc, t_tr_acc = train_adv_gh(
                        adv_dataloader, net_DCD_m, net_g, net_h, criterion_gh, criterion_DCD, optimizer_gh, epoch, dcd_mode="m",
                        alpha=CONFIG.alpha, beta=CONFIG.beta, gamma=CONFIG.gamma, metric=CONFIG.metric)    
                    train_loss, s_tr_acc, t_tr_acc = train_adv_gh(
                        adv_dataloader, net_DCD_m, net_g, net_h2, criterion_gh, criterion_DCD, optimizer_gh2, epoch, dcd_mode="m",
                        alpha=CONFIG.alpha, beta=CONFIG.beta, gamma=CONFIG.gamma, metric=CONFIG.metric)
                if CONFIG.DCD_h_activate:
                    train_loss, s_tr_acc, t_tr_acc = train_adv_gh(
                        adv_dataloader, net_DCD_h, net_g, net_h, criterion_gh, criterion_DCD, optimizer_gh, epoch, dcd_mode="h",
                        alpha=CONFIG.alpha, beta=CONFIG.beta, gamma=CONFIG.gamma, metric=CONFIG.metric)    
                    train_loss, s_tr_acc, t_tr_acc = train_adv_gh(
                        adv_dataloader, net_DCD_h, net_g, net_h2, criterion_gh, criterion_DCD, optimizer_gh2, epoch, dcd_mode="h",
                        alpha=CONFIG.alpha, beta=CONFIG.beta, gamma=CONFIG.gamma, metric=CONFIG.metric)
                
            else:
                if CONFIG.DCD_l_activate:
                    train_loss, s_tr_acc, t_tr_acc = train_adv_gh(
                        adv_dataloader, net_DCD_l, net_g, net_h, criterion_gh, criterion_DCD, optimizer_gh, epoch, dcd_mode="l",
                        alpha=CONFIG.alpha, beta=CONFIG.beta, gamma=CONFIG.gamma, metric=CONFIG.metric)
                if CONFIG.DCD_m_activate:    
                    train_loss, s_tr_acc, t_tr_acc = train_adv_gh(
                        adv_dataloader, net_DCD_m, net_g, net_h, criterion_gh, criterion_DCD, optimizer_gh, epoch, dcd_mode="m",
                        alpha=CONFIG.alpha, beta=CONFIG.beta, gamma=CONFIG.gamma, metric=CONFIG.metric)
                if CONFIG.DCD_h_activate:
                    train_loss, s_tr_acc, t_tr_acc = train_adv_gh(
                        adv_dataloader, net_DCD_h, net_g, net_h, criterion_gh, criterion_DCD, optimizer_gh, epoch, dcd_mode="h",
                        alpha=CONFIG.alpha, beta=CONFIG.beta, gamma=CONFIG.gamma, metric=CONFIG.metric)


        #train DCD
        if CONFIG.DCD_l_activate:
            DCD_loss, DCD_acc, DCD_f1s = train_DCD(DCD_dataloader, net_DCD_l, net_g, criterion_DCD, optimizer_DCD_l, epoch, dcd_mode="l")
        if CONFIG.DCD_m_activate:
            DCD_loss, DCD_acc, DCD_f1s = train_DCD(DCD_dataloader, net_DCD_m, net_g, criterion_DCD, optimizer_DCD_m, epoch, dcd_mode="m")
        if CONFIG.DCD_h_activate:
            DCD_loss, DCD_acc, DCD_f1s = train_DCD(DCD_dataloader, net_DCD_h, net_g, criterion_DCD, optimizer_DCD_h, epoch, dcd_mode="h")

        #MCD if MCD==True
        if CONFIG.MCD:
            optimizer_g = torch.optim.Adam(net_g.parameters(),  lr = 0.001, weight_decay=0.0005, momentum = 0.9)
            optimizer_h1 = torch.optim.Adam(net_h.parameters(),  lr = 0.001, weight_decay=0.0005, momentum = 0.9)
            optimizer_h2 = torch.optim.Adam(net_h2.parameters(),  lr = 0.001, weight_decay=0.0005, momentum = 0.9)
            train_h_MCD(
                net_g, net_h, net_h2, s_trainloader, t_trainloader, optimizer_g, optimizer_h1, optimizer_h2, criterion_gh, discrepancy
                )

        #eval g,h
        s_val_loss, t_val_loss, s_val_acc, t_val_acc, s_val_f1s, t_val_f1s = validate(
            s_testloader, t_testloader, net_g, net_h, criterion_gh, metric=CONFIG.metric
        )
        #記録
        tmp = pd.Series([
            epoch,
            train_loss, s_val_loss, t_val_loss,
            s_tr_acc, t_tr_acc,
            s_val_acc, t_val_acc,
            s_val_f1s, t_val_f1s,
            DCD_loss, DCD_acc, DCD_f1s
        ], index=log.columns
        )
        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(CONFIG.result_path, 'log_step3.csv'), index=False)
        print(
            'epoch: {}\ttrain loss: {:.4f}\ts_val loss: {:.4f}\tt_val loss: {:.4f}\ts_val_acc: {:.5f}\tt_val_acc: {:.5f}\ts_val_f1s: {:.5f}\tt_val_f1s: {:.5f}\tDCD_loss : {:.5f}\tDCD_acc : {:.5f}'
            .format(epoch, train_loss,
                    s_val_loss, t_val_loss,
                    s_val_acc, t_val_acc, 
                    s_val_f1s, t_val_f1s,
                    DCD_loss, DCD_acc)
        )

if __name__ == '__main__':
    main()

"""
CONFIGの変数
height, width, batch_size, num_workers, n_classes, model, optimizer_gh, optimizer_DCD, class_weight, result_path, num_ep_init,
num_ep_DCD, num_ep_train, learning_rate, metric, metric_method

"""


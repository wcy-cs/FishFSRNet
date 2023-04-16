import torch
import torch.optim as optim
from option import args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch.nn as nn
import dataset_parsingnet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import util
import torchvision
from parsingnet import ParsingNet

net = ParsingNet()
net = util.prepare(net)
writer = SummaryWriter('./logs/{}'.format(args.writer_name))
traindata = dataset_parsingnet.Data(root=os.path.join(args.dir_data, args.data_train), args=args, train=True)
trainset = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=16)
valdata = dataset_parsingnet.Data(root=os.path.join(args.dir_data, args.data_val), args=args, train=False)
valset = DataLoader(valdata, batch_size=1, shuffle=False, num_workers=1)

criterion1 = nn.L1Loss()
optimizer = optim.Adam(params=net.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)

for i in range(args.epochs):
    net.train()
    train_loss = 0
    bum = len(trainset)
    for batch, (lr, hr, _) in enumerate(trainset):
        lr, hr = util.prepare(lr), util.prepare(hr)
        sr = net(lr)
        loss = criterion1(sr, hr)
        train_loss = train_loss + loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch：{} loss: {:.3f}".format(i + 1, train_loss / (len(trainset)) * 255))
    writer.add_scalar('train_loss', train_loss / (len(trainset)) * 255, i + 1)
    os.makedirs(os.path.join(args.save_path, args.writer_name), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'model'), exist_ok=True)
    torch.save(net.state_dict(),
               os.path.join(args.save_path, args.writer_name, 'model', 'epoch{}.pth'.format(i + 1)))

    net.eval()
    val_psnr_my = 0
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'result'), exist_ok=True)
    for batch, (lr, hr, filename) in enumerate(valset):
        lr, hr = util.prepare(lr), util.prepare(hr)
        sr = net(lr)
        val_psnr_my = val_psnr_my + util.cal_psnr(hr[0].data.cpu(), sr[0].data.cpu())

    print("Epoch：{} val  psnr: {:.3f}".format(i + 1, val_psnr_my / (len(valset))))
    writer.add_scalar("val_psnr_my", val_psnr_my / len(valset), i + 1)


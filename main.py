from option import args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch
import torch.optim as optim
import torch.nn as nn
from data import dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import util
from model import reduce

epochs = args.epochs
lr = args.lr

net = reduce.FISHNET(args)

net = util.prepare(net)
print(net)

writer = SummaryWriter('./logs/{}'.format(args.writer_name))
traindata = dataset.Data(root=os.path.join(args.dir_data, args.data_train), args=args, train=True)
trainset = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=16)


valdata = dataset.Data(root=os.path.join(args.dir_data, args.data_val), args=args, train=False)
valset = DataLoader(valdata, batch_size=1, shuffle=False, num_workers=1)

criterion1 = nn.L1Loss()
optimizer = optim.Adam(params=net.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8)

for i in range(epochs):
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
    torch.save(net.state_dict(), os.path.join(args.save_path, args.writer_name, 'model', 'epoch{}.pth'.format(i + 1)))

    net.eval()
    val_psnr = 0
    val_ssim = 0

    os.makedirs(os.path.join(args.save_path, args.writer_name, 'result'), exist_ok=True)
    for batch, (lr, hr, filename) in enumerate(valset):
        lr, hr = util.prepare(lr), util.prepare(hr)
        sr = net(lr)
                 
        psnr_c, ssim_c = util.calc_metrics(hr[0].data.cpu(), sr[0].data.cpu())
        val_psnr = val_psnr + psnr_c
        val_ssim = val_ssim + ssim_c
    print("Epoch：{} val  psnr: {:.3f}".format(i + 1, val_psnr / (len(valset))))

    writer.add_scalar("val_psnr", val_psnr / len(valset), i + 1)
    writer.add_scalar("val_ssim", val_ssim / len(valset), i + 1)















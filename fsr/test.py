from option import args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch
import dataset_parsing
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import util
import torchvision
from fishfsrnet import FISHNET
net = FISHNET(args)
net = util.prepare(net)
writer = SummaryWriter('./logs/{}'.format(args.writer_name))
testdata = dataset_parsing.Data(root=os.path.join(args.dir_data, args.data_test), args=args, train=False)
testset = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=1)
pretrained_dict = torch.load('/epoch.pth', map_location='cuda:0')
net.load_state_dict(pretrained_dict)
net = util.prepare(net)
net.eval()
val_psnr = 0
val_ssim = 0
with torch.no_grad():
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'result-test'), exist_ok=True)
    net.eval()
    for batch, (lr, hr, parsing, filename) in enumerate(testset):

        lr, hr, parsing = util.prepare(lr), util.prepare(hr), util.prepare(parsing)
        sr = net(lr, parsing)

        psnr1, _ = util.calc_metrics(hr[0].data.cpu(), sr[0].data.cpu(), crop_border=8)
        val_psnr = val_psnr + psnr1
        torchvision.utils.save_image(sr[0],
                                         os.path.join(args.save_path, args.writer_name, 'result-test',
                                                      '{}'.format(str(filename[0])[:-4] + ".png")))
    print("Test psnr: {:.3f}".format(val_psnr / (len(testset))))
    print(val_ssim / (len(testset)))

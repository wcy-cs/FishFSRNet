from option import args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch
import data_parsingnet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import util
import torchvision
import parsingnet
net = parsingnet.ParsingNet()
net = util.prepare(net)
print(util.get_parameter_number(net))
writer = SummaryWriter('./logs/{}'.format(args.writer_name))
testdata = data_parsingnet.Data(root=os.path.join(args.dir_data, args.data_test), args=args, train=False)
testset = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=1)
pretrained_dict = torch.load('./epoch.pth', map_location='cuda:0')
net.load_state_dict(pretrained_dict)
net = util.prepare(net)
net.eval()
val_psnr = 0
val_ssim = 0
with torch.no_grad():
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'result-test'), exist_ok=True)
    net.eval()
    timer_test = util.timer()
    for batch, (lr, _, filename) in enumerate(testset):
        lr = util.prepare(lr)
        p = net(lr)
        torchvision.utils.save_image(p[0],
                                         os.path.join(args.save_path, args.writer_name, 'result-test',
                                                      '{}'.format(str(filename[0])[:-4] + ".png")))
    print("Tesing over.")

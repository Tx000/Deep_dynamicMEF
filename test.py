import time
import argparse
from torch.autograd import Variable
from model_PE import model_PE_LH
from model_dynamic import MEF_dynamic
from utils import *

torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_PE_dir', type=str, default='checkpoint/pre_en.pth')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/mef.pth')
parser.add_argument('--results_dir', type=str, default='results')
parser.add_argument('--dataset', type=str, default='./dataset/Test')
args = parser.parse_args()

if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)

scene_dirs = sorted(os.listdir(os.path.join(args.dataset, 'Input')))
nScenes = len(scene_dirs) // 2
time_all = 0.0

pre_en = model_PE_LH().cuda()
pre_en.load_state_dict(torch.load(args.checkpoint_PE_dir, map_location=torch.device('cpu')))
pre_en.eval()

mef = MEF_dynamic().cuda()
mef.load_state_dict(torch.load(args.checkpoint_dir, map_location=torch.device('cpu')))
mef.eval()

for idx in range(nScenes):
    print('batch no. %d:' % (idx + 1))
    input1_L, input1_H, label1, input2_L, input2_H, label2 = get_input(args.dataset, idx)

    st = time.time()

    with torch.no_grad():
        input1_en_L = Variable(torch.from_numpy(input1_L / 255.0)).cuda().type(torch.cuda.FloatTensor).permute(2, 0, 1).unsqueeze(0)
        input1_en_H = Variable(torch.from_numpy(input1_H / 255.0)).cuda().type(torch.cuda.FloatTensor).permute(2, 0, 1).unsqueeze(0)
        label1 = Variable(torch.from_numpy(label1 / 255.0)).cuda().type(torch.cuda.FloatTensor).permute(2, 0, 1).unsqueeze(0)

        input2_en_L = Variable(torch.from_numpy(input2_L / 255.0)).cuda().type(torch.cuda.FloatTensor).permute(2, 0, 1).unsqueeze(0)
        input2_en_H = Variable(torch.from_numpy(input2_H / 255.0)).cuda().type(torch.cuda.FloatTensor).permute(2, 0, 1).unsqueeze(0)
        label2 = Variable(torch.from_numpy(label2 / 255.0)).cuda().type(torch.cuda.FloatTensor).permute(2, 0, 1).unsqueeze(0)

        # pre-enahancement
        enhanced1_L, enhanced1_H = pre_en(input1_en_L, input1_en_H)
        enhanced2_L, enhanced2_H = pre_en(input2_en_L, input2_en_H)

    # align under-exposed image to over-exposed image
    enhanced1_L = np.transpose(np.squeeze(enhanced1_L.data.cpu().numpy()), axes=(1, 2, 0)) * 255
    enhanced1_H = np.transpose(np.squeeze(enhanced1_H.data.cpu().numpy()), axes=(1, 2, 0)) * 255
    aligned1_L, h = alignImages(enhanced1_L.astype(np.uint8), enhanced1_H.astype(np.uint8), input1_L)
    cv2.imwrite(os.path.join(args.results_dir, 'aligned_{:03d}-1_L.png'.format(idx + 1)), aligned1_L)
    if abs(h[0, 2]) > 500 or abs(h[1, 2]) > 500:
        print('error h1: {}'.format(h))
        aligned1_L = input1_L

    enhanced2_L = np.transpose(np.squeeze(enhanced2_L.data.cpu().numpy()), axes=(1, 2, 0)) * 255
    enhanced2_H = np.transpose(np.squeeze(enhanced2_H.data.cpu().numpy()), axes=(1, 2, 0)) * 255
    aligned2_L, h = alignImages(enhanced2_L.astype(np.uint8), enhanced2_H.astype(np.uint8), input2_L)
    cv2.imwrite(os.path.join(args.results_dir, 'aligned_{:03d}-2_L.png'.format(idx + 1)), aligned2_L)
    if abs(h[0, 2]) > 500 or abs(h[1, 2]) > 500:
        print('error h2: {}'.format(h))
        aligned2_L = input2_L

    with torch.no_grad():
        input1_L = Variable(torch.from_numpy(aligned1_L / 255.0)).cuda().type(torch.cuda.FloatTensor).permute(2, 0, 1).unsqueeze(0)
        input2_L = Variable(torch.from_numpy(aligned2_L / 255.0)).cuda().type(torch.cuda.FloatTensor).permute(2, 0, 1).unsqueeze(0)

        # Fusion
        out_MEF1 = mef(torch.cat([input1_L, input1_en_H], dim=1))
        out_MEF2 = mef(torch.cat([input2_L, input2_en_H], dim=1))
        torch.cuda.synchronize(0)
        fl = time.time()

    out_MEF1 = np.transpose(np.squeeze(out_MEF1.data.cpu().numpy()), axes=(1, 2, 0)) * 255
    out_MEF2 = np.transpose(np.squeeze(out_MEF2.data.cpu().numpy()), axes=(1, 2, 0)) * 255

    time_all += fl - st

    torch.cuda.empty_cache()
    cv2.imwrite(os.path.join(args.results_dir, 'test_{:03d}_1.png'.format(idx + 1)),
                out_MEF1)
    cv2.imwrite(os.path.join(args.results_dir, 'test_{:03d}_2.png'.format(idx + 1)),
                out_MEF2)

    print("time: %.4f" % (fl - st))
    time_all += (fl - st) / nScenes

print("Average time: %.4f" % (time_all / nScenes))

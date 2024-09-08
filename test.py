import timeit
from datetime import datetime
import yaml
from addict import Dict
import argparse
import cv2
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import imageio
from dataloaders import lane_detect
from dataloaders import augmentation as augment
import torch.nn.functional as F
from models.Event_ld_network import network
from utils import loss as losses
from utils import iou_eval
from utils.metrics import runningScore, averageMeter
from dataloaders.utils import *
CONFIG=Dict(yaml.load(open("./config/testing.yaml"),Loader=yaml.FullLoader))


torch.manual_seed(125)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(125)
ap = argparse.ArgumentParser()
ap.add_argument('--backbone_network', required=False,help = 'name of backbone network',default='')
ap.add_argument('--model_path_resume', required=False,help = 'path to a model to resume from',default= './model_zoo/Event_ld.pth')
ap.add_argument('--nEpochs', required=False,help = 'nEpochs',default=1)
ap.add_argument('--resume_epoch', required=False,help = 'resume_epoch',default=0)
ap.add_argument('--testBatch', required=False,help = 'Batch during testing',default=1)
args = ap.parse_args()
backbone_network=args.backbone_network
nEpochs = args.nEpochs
resume_epoch = args.resume_epoch
dataset_path=CONFIG.DATASET
experiment_id=datetime.now().strftime("%Y-%m-%d_%H_%M")
net = network.build(backbone_network,None,5)
if CONFIG.USING_GPU:
    torch.cuda.set_device(device=CONFIG.GPU_ID)
    net.cuda()

print("Using a weights from training coarse data from: {}...".format(CONFIG.model_path))
net.load_state_dict(torch.load(CONFIG.model_path))
running_metrics_test = runningScore(CONFIG.n_classes)    
modelName = 'LDNet-' + backbone_network + '-lane'
print(modelName)

criterion = losses.cross_entropy2d

if resume_epoch != nEpochs+1:
    composed_transforms_tr = transforms.Compose([
        augment.FixedResize((256,256)),
        augment.ToTensor()])
    lane_detect_test = lane_detect.Lane_detect(root=dataset_path,n_classes=CONFIG.n_classes,split='test',transform=composed_transforms_tr)
    testloader = DataLoader(lane_detect_test, batch_size=args.testBatch, shuffle=False, num_workers=0)

    loaders=[ testloader ]
    num_img_test = len(testloader)
    running_loss_te = 0.0

    previous_miou = -1.0
    global_step = 0
    iev = iou_eval.Eval(CONFIG.n_classes,19)
    test_loss_meter = averageMeter()

    best_iou = -100.0
    i = 0
    flag = True
    test_rlt_f1=[]
    test_rlt_OA=[]
    test_rlt_iou=[]
    best_f1_till_now=0
    best_OA_till_now=0
    best_IOU_till_now=0

    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        total_miou = 0.0
        net.eval()
        for ii, sample_batched in enumerate(testloader):
                inputs, labels = sample_batched['image'], sample_batched['label']
                labels_down = labels.to(torch.uint8)

                image_array = labels_down.squeeze()
                save_path = './output_color_mt'
                image_array[image_array == 1] = 100
                image_array[image_array == 2] = 150
                image_array[image_array == 3] = 200
                image_array[image_array == 4] = 250
                file_name = os.path.join(save_path, f'label_{ii}.png')

                inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                if CONFIG.USING_GPU:
                    inputs, labels = inputs.cuda(), labels.cuda()

                with torch.no_grad():
                    outputs = net.forward(inputs)
                outputs = F.interpolate(outputs, size=(800,1280), mode='bilinear', align_corners=True)
                predictions = torch.max(outputs, 1)[1]
                off=predictions.detach().cpu().numpy()
                pred_color=decode_segmap_cv(off, 'lane')
                pred_color = pred_color.astype(np.uint8)
                save_path = './output_color_mt'
                file_name = os.path.join(save_path, f'image_{ii}.png')
                imageio.imwrite(file_name, pred_color)

                loss = criterion(outputs, labels,reduct='sum',weight=None)
                running_loss_te += loss.item()

                y = torch.ones(labels.size()[2], labels.size()[3]).mul(19).cuda()
                labels=labels.where(labels !=255, y)
                iev.addBatch(predictions.unsqueeze(1).data,labels.cpu())
                running_metrics_test.update(labels, predictions.unsqueeze(1).data)

                if ii % num_img_test == num_img_test - 1:
                    miou=iev.getIoU()[0]
                    running_loss_te = running_loss_te/ num_img_test
                    print('TEST:')
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * args.testBatch + inputs.data.shape[0]))
                    print('Loss: %f' % running_loss_te)
                    print("Predi iou",iev.getIoU())
                    running_loss_te = 0
                    iev.reset()
        score, class_iou = running_metrics_test.get_scores()
        running_metrics_test.reset()


        avg_f1 = score["Mean F1 : \t"]
        OA=score["Overall Acc: \t"]
        IOU=score["Mean IoU : \t"]
        test_rlt_f1.append(avg_f1)
        test_rlt_OA.append(score["Overall Acc: \t"])
        test_rlt_iou.append(score["Mean IoU : \t"])

        if avg_f1 >= best_f1_till_now:
            best_f1_till_now = avg_f1
            correspond_OA = score["Overall Acc: \t"]
            correspond_IOU = score["Mean IoU : \t"]
            best_f1_epoch_till_now = epoch+1
        print("\nBest F1 till now = ", best_f1_till_now)
        print("Correspond OA= ", correspond_OA)
        print("Correspond IOU= ", correspond_IOU)
        print("Best F1 Iter till now= ", best_f1_epoch_till_now)

        if IOU >= best_IOU_till_now:
            best_OA_till_now = IOU
            correspond_f1 = score["Mean F1 : \t"]
            correspond_iou = score["Mean IoU : \t"]
            correspond_acc=score["Overall Acc: \t"]
            best_IOU_epoch_till_now = i+1

            state = {
                "epoch": epoch + 1,
                "best_OA": best_OA_till_now,
            }

        print("Best IOU till now = ", best_IOU_till_now)
        print("Correspond F1= ", correspond_f1)
        print("Correspond OA= ",correspond_acc)
        print("Correspond IOU= ",correspond_iou)
        print("Best IOU Iter till now= ", best_IOU_epoch_till_now)
       
   

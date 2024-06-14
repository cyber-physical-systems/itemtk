# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 classification model on a classification dataset

Usage:
    $ bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
    $ python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate ImageNet

Usage - formats:
    $ python classify/val.py --weights yolov5s-cls.pt                 # PyTorch
                                       yolov5s-cls.torchscript        # TorchScript
                                       yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                       yolov5s-cls_openvino_model     # OpenVINO
                                       yolov5s-cls.engine             # TensorRT
                                       yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                       yolov5s-cls_saved_model        # TensorFlow SavedModel
                                       yolov5s-cls.pb                 # TensorFlow GraphDef
                                       yolov5s-cls.tflite             # TensorFlow Lite
                                       yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                       yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
# from sklearn.metrics import confusion_matrix
import glob

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from models.experimental import EnsembleValidator, FusionComponent, Ensemble
from utils.dataloaders import create_classification_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_img_size, check_requirements, colorstr,
                           increment_path, print_args)
from torch.utils.data import DataLoader
from utils.torch_utils import select_device, smart_inference_mode
from utils.dataloaders import create_classification_dataloader, ClassificationDataset
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import math

per_class_results = defaultdict(list)


def calculate_mcc_for_class(TP, TN, FP, FN):
    # 计算MCC分母，防止除以零
    denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    if denominator == 0:
        return 0  # 如果分母为0，则MCC为0
    else:
        mcc = (TP * TN - FP * FN) / denominator
        return mcc


def calculate_total_mcc(mcc_scores):
    # Filter out the classes with MCC scores to avoid division by zero in case of no predictions for a class
    valid_mcc_scores = [mcc for mcc in mcc_scores.values() if not math.isnan(mcc)]

    # Calculate the average MCC across all valid classes
    total_mcc = sum(valid_mcc_scores) / len(valid_mcc_scores) if valid_mcc_scores else 0
    return total_mcc

@smart_inference_mode()
def run(
    data=ROOT / '../datasets/mnist',  # dataset dir
    weights=ROOT / 'yolov5s-cls.pt',  # model.pt path(s)
    batch_size=128,  # batch size
    imgsz=224,  # inference size (pixels)
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    verbose=False,  # verbose output
    project=ROOT / 'runs/val-cls',  # save to project/name
    name='exp',  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    dataset2 = None, # 融合
    dataset3 = None, # 融合
    dataset4 = None, # 融合
    criterion=None,
    pbar=None,
):
    # Initialize/load model and set device
    dataset2_root_test = './data/train_data_padding_SCA/val'
    dataset3_root_test = './data/train_data_padding_GAF_1/val'
    dataset4_root_test = './data/train_data_padding_INO/val'
    training = model is not None
    if training:  # called by train.py

        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()

    else:  # called directly
        dataset2_root_test = './train_data_padding_SCA/val'
        dataset3_root_test = './train_data_padding_GAF_1/val'
        dataset4_root_test = './data/train_data_padding_INO/val'
        state_dict = torch.load(weights[0], map_location='cpu')
        data = Path(data)

        # dataset1 = ClassificationDataset(root=data, augment=True, imgsz=224, cache=False)
        dataset2 = ClassificationDataset(root=dataset2_root_test, augment=True, imgsz=224, cache=False)
        dataset3 = ClassificationDataset(root=dataset3_root_test, augment=True, imgsz=224, cache=False)
        dataset4 = ClassificationDataset(root=dataset4_root_test, augment=True, imgsz=224, cache=False)


        if 'model' in state_dict and isinstance(state_dict['model'], Ensemble):
            model = state_dict['model']
        else:
            raise TypeError(f"Expected an Ensemble object in checkpoint['model'], but got: {type(state_dict['model'])}")
        model = model.half()

        model.to('cpu')

        # Dataloader
        data = Path(data)
        # test_dir = data / 'test' if (data / 'test').exists() else data / 'val'  # data/test or data/val
        test_dir = data
        torch.manual_seed(0)

        dataloader = create_classification_dataloader(path=test_dir,
                                                      imgsz=imgsz,
                                                      batch_size=batch_size,
                                                      augment=False,
                                                      rank=-1,
                                                      workers=workers)


    model.eval()

    pred, targets, loss, dt = [], [], 0, (Profile(), Profile(), Profile())
    n = len(dataloader)  # number of batches

    action = 'validating' if dataloader.dataset.root.stem == 'val' else 'testing'
    desc = f'{pbar.desc[:-36]}{action:>36}' if pbar else f'{action}'

    all_files_dataset2 = sorted(glob.glob(os.path.join(dataset2_root_test, '**', '*.*'), recursive=True))

    filename_to_index2 = {os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)): i for
                          i, path in enumerate(all_files_dataset2)}

    all_files_dataset3 = sorted(glob.glob(os.path.join(dataset3_root_test, '**', '*.*'), recursive=True))

    filename_to_index3 = {os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)): i for
                          i, path in enumerate(all_files_dataset3)}

    all_files_dataset4 = sorted(glob.glob(os.path.join(dataset4_root_test, '**', '*.*'), recursive=True))

    filename_to_index4 = {os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)): i for
                          i, path in enumerate(all_files_dataset4)}

    top1_classes = []
    top1_confidences = []
    top5_classes = []
    top5_confidences = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 14
    class_stats = {i: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for i in range(num_classes)}
    # paired_loader = zip(dataloader, dataloader2, dataloader3, dataloader4)
    bar = tqdm(dataloader, desc, n, not training, bar_format=TQDM_BAR_FORMAT, position=0)

    with torch.cuda.amp.autocast(enabled=device != 'cpu'):

        for images1, labels1,paths1 in bar:

            images2 = []
            labels2 = []
            images3 = []
            labels3 = []
            images4 = []
            labels4 = []
            for path1 in paths1:

                relative_path = os.path.join(os.path.basename(os.path.dirname(path1)), os.path.basename(path1))
                index2 = filename_to_index2[relative_path]
                image2, label2, path2 = dataset2[index2]
                images2.append(image2)
                labels2.append(label2)

                index3 = filename_to_index3[relative_path]
                image3, label3, path3 = dataset3[index3]
                images3.append(image3)

                index4 = filename_to_index4[relative_path]
                image4, label4, path4 = dataset4[index4]
                images4.append(image4)

            with dt[0]:
                images1, labels1 = images1.to(device, non_blocking=True), labels1.to(device)

                images2 = torch.stack(images2).to(device, non_blocking=True)
                labels2 = torch.tensor(labels2).to(device, non_blocking=True)
                images3 = torch.stack(images3).to(device, non_blocking=True)
                images4 = torch.stack(images4).to(device, non_blocking=True)
            with dt[1]:
                model.to(device)
                images1 = images1.to(device, non_blocking=True)
                images2 = images2.to(device, non_blocking=True)
                images3 = images3.to(device, non_blocking=True)
                images4 = images4.to(device, non_blocking=True)

                labels1 = labels1.to(device, non_blocking=True)

                y= model(images1,images2,images3,images4)

                top5_confidences_batch, top5_classes_batch = y.topk(5, dim=1)
                top1_classes_batch = top5_classes_batch[:, 0]
                top1_confidences_batch = top5_confidences_batch[:, 0]


                top1_classes.append(top1_classes_batch.cpu().numpy())
                top1_confidences.append(top1_confidences_batch.cpu().numpy())
                top5_classes.append(top5_classes_batch.cpu().numpy())
                top5_confidences.append(top5_confidences_batch.cpu().numpy())

            with dt[2]:

                pred.append(y.argsort(1, descending=True)[:, :5])
                targets.append(labels1)
                if criterion:
                    loss += criterion(y, labels1)

            for i in range(images1.size(0)):

                label = labels1[i].item()
                pred_label = top1_classes_batch[i].item()

                other_labels = set(range(num_classes)) - {label}
                if pred_label == label:
                    class_stats[label]['TP'] += 1
                else:
                    class_stats[label]['FN'] += 1
                    class_stats[pred_label]['FP'] += 1

                for label_temp in other_labels:
                    if label_temp != pred_label:
                        class_stats[label_temp]['TN'] += 1

                top5_scores_tensor = torch.tensor(top5_confidences_batch[i])
                top5_scores_softmax = F.softmax(top5_scores_tensor, dim=0).cpu().numpy()
                per_class_results[label].append({
                    'image_index': i,
                    'top1_class': top1_classes_batch[i].item(),
                    'top1_confidence': top1_confidences_batch[i].item(),
                    'top5_classes': top5_classes_batch[i].tolist(),
                    'top5_confidences': top5_scores_softmax.tolist(),
                    'stats': class_stats[label],
                })

            mcc_scores = {label: 0 for label in range(num_classes)}

            for label in range(num_classes):
                TP = class_stats[label]['TP']
                TN = class_stats[label]['TN']
                FP = class_stats[label]['FP']
                FN = class_stats[label]['FN']
                mcc_scores[label] = calculate_mcc_for_class(TP, TN, FP, FN)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    top1_pred = pred[:, 0]

    conf_mat = confusion_matrix(targets.cpu().numpy(), top1_pred.cpu().numpy())
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                          xticklabels=model.names, yticklabels=model.names)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.savefig('./confusion_matrix_pg.png')

    plt.close()

    correct = (targets[:, None] == pred).float()

    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()


    if pbar:

        pbar.desc = f'{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}'


    if verbose:  # all classes
        LOGGER.info(f"{'Class':>24}{'Images':>12}{'top1_acc':>12}{'top5_acc':>12}")
        LOGGER.info(f"{'all':>24}{targets.shape[0]:>12}{top1:>12.3g}{top5:>12.3g}")

        for label, images_info in per_class_results.items():
            acc_i = acc[targets == label]
            top1i, top5i = acc_i.mean(0).tolist()
            # print('per_class_results : label = {}'.format(label))
            class_name = model.names[label]
            LOGGER.info(f'{class_name:>24}{acc_i.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}')

        total_tp = total_fp = total_tn = total_fn = 0
        LOGGER.info(f"{'Class':>24}{'Images':>12}{'TP':>12}{'FP':>12}{'TN':>12}{'FN':>12}")
        for label, stats in class_stats.items():
            tp, fp, tn, fn = stats['TP'], stats['FP'], stats['TN'], stats['FN']
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            # print('class_stats : label = {}'.format(label))
            class_name = model.names[label]
            LOGGER.info(f'{class_name:>24}{tp + fn:>12}{tp:>12}{fp:>12}{tn:>12}{fn:>12}')

        LOGGER.info(f"{'Class':>24}{'Images':>12}{'MCC':>12}")
        for label, mcc in mcc_scores.items():
            class_name = model.names[label]
            LOGGER.info(f'{class_name:>24}{tp + fn:>12}{mcc:>12.3g}')

        total_mcc = calculate_total_mcc(mcc_scores)
        print(f"Total MCC for all classes: {total_mcc}")

        total_precision = total_tp / (total_tp + total_fp)
        total_recall = total_tp / (total_tp + total_fn)
        total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall)

        # Log total metrics
        LOGGER.info(f"{'Total':>24}{'Precision':>12}{'Recall':>12}{'F1 Score':>12}")
        LOGGER.info(f"{'Total':>24}{total_precision:>12.4f}{total_recall:>12.4f}{total_f1_score:>12.4f}")

        # Header for the individual class evaluation metrics
        LOGGER.info(f"{'Class':>24}{'Precision':>12}{'Recall':>12}{'F1 Score':>12}")

        for label, stats in class_stats.items():
            tp, fp, tn, fn = stats['TP'], stats['FP'], stats['TN'], stats['FN']
            class_name = model.names[label]

            # Calculate Precision, Recall, and F1 Score
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            # Log the evaluation metrics
            LOGGER.info(f'{class_name:>24}{precision:>12.4f}{recall:>12.4f}{f1_score:>12.4f}')

        # Print results
        t = tuple(x.t / len(dataloader.dataset.samples) * 1E3 for x in dt)  # speeds per image
        shape = (1, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}' % t)
        # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy

    return top1, top5, loss


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / '../datasets/mnist', help='dataset path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-cls.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--verbose', nargs='?', const=True, default=True, help='verbose output')
    parser.add_argument('--project', default=ROOT / 'runs/val-cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
import os
import sys
import time
from optparse import OptionParser

import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.utils import data

from dataset_domain_test import CMRDataset_test
from model.MMF_net import MMF_net


parser = OptionParser()
parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path',
                  default='./checkpoint2/', help='checkpoint and output path')
parser.add_option('-u', '--unique-name', type='str', dest='unique_name',
                  default='test', help='experiment name')
parser.add_option('-c', '--resume', action='store_true', dest='load',
                  default=True, help='load pretrained model')
parser.add_option('--no-resume', action='store_false', dest='load',
                  help='test without loading a checkpoint')
parser.add_option('-t', '--training-parameter', type='str', dest='params',
                  default='best.pth', help='checkpoint filename')
parser.add_option('-d', '--test-dir', type='str', dest='test_dir',
                  default='./data/', help='test data path')
parser.add_option('--gpu', type='str', dest='gpu', default='0',
                  help='GPU id')
parser.add_option('--slice-gap', type='int', dest='slice_gap', default=1,
                  help='distance between neighbouring slices')
parser.add_option('--inference-batch-size', type='int',
                  dest='inference_batch_size', default=4,
                  help='number of 2.5D windows inferred together')
options, args = parser.parse_args()


def predict_patient_25d(net, volume, slice_gap, batch_size, device):
    """Predict one patient volume with three-slice 2.5D windows.

    Args:
        volume: Tensor with shape [Z, 3, H, W].

    Returns:
        Tensor with shape [Z, 1, H, W].
    """
    pad_z = slice_gap
    blank = torch.zeros(
        (pad_z, *volume.shape[1:]),
        dtype=volume.dtype,
        device=device,
    )
    padded_volume = torch.cat((blank, volume, blank), dim=0)
    predictions = []

    for start in range(0, volume.shape[0], batch_size):
        stop = min(start + batch_size, volume.shape[0])
        windows = []

        for z in range(start, stop):
            centre = z + pad_z
            indices = [centre - slice_gap, centre, centre + slice_gap]
            # [3 modalities, 3 neighbouring slices, H, W]
            window = padded_volume[indices].permute(1, 0, 2, 3)
            windows.append(window)

        input_batch = torch.stack(windows, dim=0)
        pred = net(input_batch)

        if pred.ndim != 4 or pred.shape[1] != 1:
            raise RuntimeError(
                'MMF_net output must have shape [B, 1, H, W], '
                f'but got {tuple(pred.shape)}'
            )
        predictions.append(pred)

    return torch.cat(predictions, dim=0)


def test(net, options, device):
    testset = CMRDataset_test(dir=options.test_dir, mode='test')
    test_loader = data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=0
    )
    mae_loss = nn.L1Loss()
    net.eval()

    with torch.no_grad():
        for img, label, patient in test_loader:
            start_time = time.time()
            patient_name = patient[0]
            print(patient_name)

            # Dataset: [1, 3, H, W, Z] -> patient volume: [Z, 3, H, W]
            inputs = img.squeeze(0).permute(3, 0, 1, 2).to(device)
            labels = label.squeeze(0).permute(3, 0, 1, 2).to(device)

            prediction = predict_patient_25d(
                net,
                inputs,
                slice_gap=options.slice_gap,
                batch_size=options.inference_batch_size,
                device=device,
            )

            loss = mae_loss(prediction, labels)
            print('mae_loss in current case: %.5f' % loss.item())

            inputs_np = inputs.cpu().numpy().transpose(1, 0, 2, 3)
            labels_np = labels.cpu().numpy().transpose(1, 0, 2, 3).squeeze(0)
            prediction_np = (
                prediction.cpu().numpy().transpose(1, 0, 2, 3).squeeze(0)
            )

            experiment_dir = os.path.join(
                options.cp_path, options.unique_name
            )
            label_dir = os.path.join(experiment_dir, 'label', patient_name)
            pred_dir = os.path.join(experiment_dir, 'pred', patient_name)
            os.makedirs(label_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)

            sitk.WriteImage(
                sitk.GetImageFromArray(inputs_np[0]),
                os.path.join(label_dir, 'ct.nii.gz'),
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(inputs_np[1]),
                os.path.join(label_dir, 'oars.nii.gz'),
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(inputs_np[2]),
                os.path.join(label_dir, 'ptvs.nii.gz'),
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(labels_np),
                os.path.join(label_dir, 'dose.nii.gz'),
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(prediction_np),
                os.path.join(pred_dir, 'dose.nii.gz'),
            )

            print('batch_time:%.5f' % (time.time() - start_time))

    print('save done')


if __name__ == '__main__':
    if options.slice_gap < 1:
        parser.error('--slice-gap must be greater than or equal to 1')
    if options.inference_batch_size < 1:
        parser.error('--inference-batch-size must be greater than or equal to 1')

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Using model: MMF_net')
    net = MMF_net().to(device)

    if options.load:
        checkpoint = os.path.join(
            options.cp_path, options.unique_name, options.params
        )
        state_dict = torch.load(checkpoint, map_location='cpu')
        state_dict = {
            key.replace('module.', ''): value
            for key, value in state_dict.items()
        }
        net.load_state_dict(state_dict)
        print('Model loaded from {}'.format(checkpoint))

    test(net, options, device)
    print('done')
    sys.exit(0)

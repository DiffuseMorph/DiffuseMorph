import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
from math import *
import time
from util.visualizer import Visualizer
from PIL import Image
import numpy as np
import torch.nn.functional as F

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase != opt['phase']: continue
        if opt['phase'] == 'train':
            batchSize = opt['datasets']['train']['batch_size']
            train_set = Data.create_dataset_3D(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            training_iters = int(ceil(train_set.data_len / float(batchSize)))
        elif opt['phase'] == 'test':
            test_set = Data.create_dataset_3D(dataset_opt, phase)
            test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    if opt['phase'] == 'train':
        current_step = diffusion.begin_step
        current_epoch = diffusion.begin_epoch
        n_epoch = opt['train']['n_epoch']
        if opt['path']['resume_state']:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

        while current_epoch < n_epoch:
            current_epoch += 1
            for istep, train_data in enumerate(train_loader):
                iter_start_time = time.time()
                current_step += 1

                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if (istep + 1) % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    t = (time.time() - iter_start_time) / batchSize
                    visualizer.print_current_errors(current_epoch, istep + 1, training_iters, logs, t, 'Train')
                    visualizer.plot_current_errors(current_epoch, (istep + 1) / float(training_iters), logs)

                # validation
                if (istep + 1) % opt['train']['val_freq'] == 0:
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.test_registration(continous=False)
                    visuals = diffusion.get_current_visuals()
                    visualizer.display_current_results(visuals, current_epoch, True)

            if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

        # save model
        logger.info('End of training.')
    else:
        from model.deformation_net_2D import Dense2DSpatialTransformer

        registDice = np.zeros((test_set.data_len, 5))
        originDice = np.zeros((test_set.data_len, 5))
        registTime = []
        stn = Dense2DSpatialTransformer()
        registTime = []
        logger.info('Begin Model Evaluation.')
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for istep,  test_data in enumerate(test_loader):
            idx += 1
            dataName = test_data['P'][0].split('/')[-1][:-4]
            print('Test Data: %s' % dataName)

            time1 = time.time()
            diffusion.feed_data(test_data)
            diffusion.test_registration(continuous=True)
            time2 = time.time()

            visuals = diffusion.get_current_registration()
            defm_frames = visuals['contD'].squeeze().numpy().transpose(0, 2, 3, 1)
            flow_frames = visuals['contF'].squeeze().numpy().transpose(0, 3, 4, 2, 1)
            flow_frames_ES = flow_frames[-1]
            sflow = torch.from_numpy(flow_frames_ES.transpose(3, 2, 0, 1).copy()).unsqueeze(0)
            sflow = Metrics.transform_grid(sflow[:, 0], sflow[:, 1], sflow[:, 2])
            nb, nc, nd, nh, nw = sflow.shape
            segflow = torch.FloatTensor(sflow.shape).zero_()
            segflow[:, 2] = (sflow[:, 0] / (nd - 1) - 0.5) * 2.0  # D[0 -> 2]
            segflow[:, 1] = (sflow[:, 1] / (nh - 1) - 0.5) * 2.0  # H[1 -> 1]
            segflow[:, 0] = (sflow[:, 2] / (nw - 1) - 0.5) * 2.0  # W[2 -> 0]
            origin_seg = test_data['MS'].squeeze()
            origin_seg = origin_seg.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            regist_seg = F.grid_sample(origin_seg.cuda().float(), (segflow.cuda().float().permute(0, 2, 3, 4, 1)),
                                       mode='nearest')
            regist_seg = regist_seg.squeeze().cpu().numpy().transpose(1, 2, 0)
            label_seg = test_data['FS'][0].cpu().numpy()
            origin_seg = test_data['MS'][0].cpu().numpy()

            vals_regist = Metrics.dice_ACDC(regist_seg, label_seg)[::3]
            vals_origin = Metrics.dice_ACDC(origin_seg, label_seg)[::3]
            registDice[istep] = vals_regist
            originDice[istep] = vals_origin
            print('---- Original Dice: %03f | Deformed Dice: %03f' % (np.mean(vals_origin), np.mean(vals_regist)))

            data_origin = test_data['M'].squeeze().cpu().numpy().transpose(1, 2, 0)
            data_fixed = test_data['F'].squeeze().cpu().numpy().transpose(1, 2, 0)
            label_origin = test_data['MS'].squeeze().cpu().numpy()
            label_fixed = test_data['FS'].squeeze().cpu().numpy()

            dpt = 12
            savePath = os.path.join(result_path, '%s_mov.png' % (dataName))
            save_image((data_origin[:, :, dpt] + 1) / 2. * 255, savePath)
            savePath = os.path.join(result_path, '%s_fix.png' % (dataName))
            save_image((data_fixed[:, :, dpt] + 1) / 2. * 255, savePath)

            nframe = defm_frames.shape[0]
            for iframe in range(nframe):
                savePath = os.path.join(result_path, '%s_regist_%.2f.png' % (dataName, (iframe + 1)/nframe))
                save_image(defm_frames[iframe][:, :, dpt] * 255, savePath)
            registTime.append(time2 - time1)

        omdice, osdice = np.mean(originDice), np.std(originDice)
        mdice, sdice = np.mean(registDice), np.std(registDice)
        mtime, stime = np.mean(registTime), np.std(registTime)

        print()
        print('---------------------------------------------')
        print('Total Dice and Time Metrics------------------')
        print('---------------------------------------------')
        print('origin Dice | mean = %.3f, std= %.3f' % (omdice, osdice))
        print('Deform Dice | mean = %.3f, std= %.3f' % (mdice, sdice))
        print('Deform Time | mean = %.3f, std= %.3f' % (mtime, stime))
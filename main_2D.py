import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
from math import *
import time
from util.visualizer import Visualizer
from PIL import Image
import numpy as np

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
            train_set = Data.create_dataset_2D(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            training_iters = int(ceil(train_set.data_len / float(batchSize)))
        elif opt['phase'] == 'test':
            test_set = Data.create_dataset_2D(dataset_opt, phase)
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
                if (istep+1) % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    t = (time.time() - iter_start_time) / batchSize
                    visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, t, 'Train')
                    visualizer.plot_current_errors(current_epoch, (istep+1) / float(training_iters), logs)
                    visuals = diffusion.get_current_visuals_train()
                    visualizer.display_current_results(visuals, current_epoch, True)

                # validation
                if (istep+1) % opt['train']['val_freq'] == 0:
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
                    diffusion.test_generation(continuous=False)
                    diffusion.test_registration(continuous=False)
                    visuals = diffusion.get_current_visuals()
                    visualizer.display_current_results(visuals, current_epoch, True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')

            if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

        # save model
        logger.info('End of training.')
    else:
        from model.deformation_net_2D import Dense2DSpatialTransformer
        stn = Dense2DSpatialTransformer()
        registTime = []
        logger.info('Begin Model Evaluation.')
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for istep,  test_data in enumerate(test_loader):
            idx += 1
            fileInfo = test_data['P']
            dataXinfo, dataYinfo = fileInfo[0][0][:-4], fileInfo[1][0][:-4]

            data_origin = test_data['M'].squeeze().cpu().numpy()
            data_fixed = test_data['F'].squeeze().cpu().numpy()
            data_originRGB = test_data['MC'].squeeze().cpu().numpy()
            data_fixedRGB = test_data['FC'].squeeze().cpu().numpy()
            time1 = time.time()
            diffusion.feed_data(test_data)

            print('Generation from %s to %s' % (dataXinfo, dataYinfo))
            diffusion.test_generation(continuous=True)
            print('Registration from %s to %s' % (dataXinfo, dataYinfo))
            diffusion.test_registration(continuous=True)
            time2 = time.time()

            data_origin = (data_origin+1)/2. * 255
            data_fixed = (data_fixed + 1) / 2. * 255
            savePath = os.path.join(result_path, '%s_TO_%s_mov.png' % (dataXinfo, dataYinfo))
            save_image(data_origin, savePath)
            savePath = os.path.join(result_path, '%s_TO_%s_fix.png' % (dataXinfo, dataYinfo))
            save_image(data_fixed, savePath)

            visuals = diffusion.get_current_generation()
            sample_data = visuals['MF'].squeeze().numpy()
            for isamp in range(0, sample_data.shape[0], 6):
                savePath = os.path.join(result_path, '%s_TO_%s_sample_%d.png' % (dataXinfo, dataYinfo, isamp))
                synthetic_data = sample_data[isamp]
                synthetic_data -= synthetic_data.min()
                synthetic_data /= synthetic_data.max()
                synthetic_data = synthetic_data * 255
                save_image(synthetic_data, savePath)
            savePath = os.path.join(result_path, '%s_TO_%s_sample_last.png' % (dataXinfo, dataYinfo))
            synthetic_data = sample_data[-1]
            synthetic_data -= synthetic_data.min()
            synthetic_data /= synthetic_data.max()
            synthetic_data = synthetic_data * 255
            save_image(synthetic_data, savePath)

            savePath = os.path.join(result_path, 'RGB_%s_TO_%s_mov.png' % (dataXinfo, dataYinfo))
            save_image(data_originRGB, savePath)
            savePath = os.path.join(result_path, 'RGB_%s_TO_%s_fix.png' % (dataXinfo, dataYinfo))
            save_image(data_fixedRGB, savePath)

            visuals = diffusion.get_current_registration()
            nsamp = visuals['contF'].shape[0]
            for isamp in range(0, nsamp):
                real_C = test_data['MC'].squeeze()
                real_C = real_C.permute(2, 0, 1).unsqueeze(0).cuda()
                out_y0 = stn(real_C[:, 0:1], visuals['contF'][isamp:isamp+1].cuda())
                out_y1 = stn(real_C[:, 1:2], visuals['contF'][isamp:isamp+1].cuda())
                out_y2 = stn(real_C[:, 2:3], visuals['contF'][isamp:isamp+1].cuda())
                regist_RGB = torch.cat([out_y0, out_y1, out_y2], dim=1)
                regist_dataRGB = regist_RGB.squeeze().cpu().numpy().transpose(1, 2, 0)
                savePath = os.path.join(result_path, 'RGB_%s_TO_%s_regist_%.2f.png' % (dataXinfo, dataYinfo, (isamp+1)/nsamp))
                save_image(regist_dataRGB, savePath)

            numer = np.sum((data_fixedRGB - regist_dataRGB) ** 2)
            denom = np.sum((data_fixedRGB) ** 2 )
            nmse = numer / denom
            print('NMSE_%s_TO_%s: %.4f' % (dataXinfo, dataYinfo, nmse))
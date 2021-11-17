import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from tensorboardX import SummaryWriter
import os
from math import *
import time
from util.visualizer import Visualizer
import scipy
import numpy as np

def save_image(image_numpy, image_path):
    image_pil = scipy.misc.toimage(image_numpy)
    image_pil.save(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
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
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    batchSize = opt['datasets']['train']['batch_size']
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase != opt['phase']: continue
        if opt['phase'] == 'train':
            train_set = Data.create_dataset_face(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            training_iters = int(ceil(train_set.data_len / float(batchSize)))
        elif opt['phase'] == 'test':
            val_set = Data.create_dataset_face(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    from model.deformation_net import Dense2DSpatialTransformer
    stn = Dense2DSpatialTransformer()
    registTime = []
    logger.info('Begin Model Evaluation.')
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    rgb_scale = False
    intrp_regist = True
    if intrp_regist:
        etaRange = np.linspace(0, 1, 11)
    else:
        etaRange = np.linspace(1, 1, 1)

    for istep,  val_data in enumerate(val_loader):
        idx += 1
        fileInfo = val_data['P']
        dataXinfo, dataYinfo = fileInfo[0], fileInfo[1]

        data_origin = val_data['M'].squeeze().cpu().numpy()
        data_fixed = val_data['F'].squeeze().cpu().numpy()
        data_originRGB = val_data['MC'].squeeze().cpu().numpy()
        data_fixedRGB = val_data['FC'].squeeze().cpu().numpy()
        time1 = time.time()
        diffusion.feed_data(val_data)

        print('Generation from %s to %s' % (dataXinfo[0], dataYinfo[0]))
        diffusion.test_generation(continous=True)
        print('Registration from %s to %s' % (dataXinfo[0], dataYinfo[0]))
        diffusion.test_registration(etaRange=etaRange)
        time2 = time.time()

        visuals = diffusion.get_current_data()
        regist_data = visuals['out_M'].squeeze().numpy()
        sample_data = visuals['MF'].squeeze().numpy()
        regist_flow = visuals['flow'].squeeze().numpy().transpose(1, 2, 0)

        if rgb_scale:
            real_C = val_data['MC'].squeeze()
            real_C = real_C.permute(2, 0, 1).unsqueeze(0).cuda()
            out_y0 = stn(real_C[:, 0:1], visuals['flow'].cuda())
            out_y1 = stn(real_C[:, 1:2], visuals['flow'].cuda())
            out_y2 = stn(real_C[:, 2:3], visuals['flow'].cuda())
            regist_RGB = torch.cat([out_y0, out_y1, out_y2], dim=1)
            regist_dataRGB = regist_RGB.squeeze().cpu().numpy().transpose(1, 2, 0)

        savePath = os.path.join(result_path, '%s_TO_%s_mov.png' % (dataXinfo[0], dataYinfo[0]))
        save_image(data_origin, savePath)
        savePath = os.path.join(result_path, '%s_TO_%s_fix.png' % (dataXinfo[0], dataYinfo[0]))
        save_image(data_fixed, savePath)
        for ieta, eta in enumerate(etaRange):
            savePath = os.path.join(result_path, '%s_TO_%s_regist_eta%.2f.png' % (dataXinfo[0], dataYinfo[0], eta))
            save_image(regist_data[ieta], savePath)
        for isamp in range(0, sample_data.shape[0], 6):
            savePath = os.path.join(result_path, '%s_TO_%s_sample%d.png' % (dataXinfo[0], dataYinfo[0], isamp))
            save_image(sample_data[isamp], savePath)
        savePath = os.path.join(result_path, '%s_TO_%s_sample.png' % (dataXinfo[0], dataYinfo[0]))
        save_image(sample_data[-1], savePath)

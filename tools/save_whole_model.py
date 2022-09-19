import argparse

import torch
from loguru import logger
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='../cfg/yolor_csp.cfg', help='model cfg file path')
    parser.add_argument('--weight', type=str, default=r'../runs/train/exp/weights/last.pt', help='weight path')
    parser.add_argument('--img_size', type=int, default=640, help='weight path')
    parser.add_argument('--save_whole_model', action='store_true', default=True, help='save whole model')
    parser.add_argument('--onnx', action='store_true', default=False, help='pytorch convert onnx')
    opt = parser.parse_args()
    logger.info(opt)
    if opt.save_whole_model:
        from models.models import *
        model = Darknet(opt.cfg_path, opt.img_size).cuda()
        ckpt = torch.load(opt.weight)
        model.load_state_dict(ckpt['model'])
        model.eval()
        logger.info(model)
        whole_model = {'epoch': ckpt['epoch'],
                                'best_fitness': ckpt['best_fitness'],
                                'best_fitness_p': ckpt['best_fitness_p'],
                                'best_fitness_r': ckpt['best_fitness_r'],
                                'best_fitness_ap50': ckpt['best_fitness_r'],
                                'best_fitness_ap': ckpt['best_fitness_ap'],
                                'best_fitness_f': ckpt['best_fitness_f'],
                                'training_results': ckpt['training_results'],
                                'model': model,
                                'optimizer': ckpt['optimizer'],
                                'wandb_id': ckpt['wandb_id']}
        torch.save(whole_model, '../whole_model.pt')
        logger.info("save model successful!")
    if opt.onnx:
        from models.models import *
        x = torch.ones(1, 3,opt.img_size,opt.img_size).cuda()
        model = Darknet(opt.cfg_path, opt.img_size).cuda()
        model.eval()
        torch.onnx.export(model, x, '../yolor_csp.onnx', verbose=True, opset_version=11)
        logger.info("pytorch convert onnx successful!")
import torch
import torch.nn as nn
import torch_pruning as tp
from loguru import  logger

"""
剪枝的时候根据模型结构去剪，不要盲目的猜
剪枝完需要进行一个微调训练
"""

@logger.catch
def Conv_pruning(whole_model_weights):
    logger.add('../logs/Conv_pruning.log', rotation='1 MB')
    ckpt = torch.load(whole_model_weights)
    model = ckpt['model']  # 模型的加载
    model_dict = model.state_dict()  # 获取模型的字典

    # -------------------特定卷积的剪枝--------------------
    for k, v in model_dict.items():
        """
        比如要剪 model.module_list[22].Conv2d
        """
        if k == 'module_list.22.Conv2d.weight':  # 筛选出该层 (根据自己需求)
            # 1. setup strategy (L1 Norm)
            strategy = tp.strategy.L1Strategy()  # or tp.strategy.RandomStrategy()

            # 2. build layer dependency
            DG = tp.DependencyGraph()
            DG.build_dependency(model, example_inputs=torch.randn(1, 3, 640, 640))
            num_params_before_pruning = tp.utils.count_params(model)
            # 3. get a pruning plan from the dependency graph.
            pruning_idxs = strategy(v, amount=0.4)  # or manually selected pruning_idxs=[2, 6, 9, ...]
            # 放入要剪枝的层
            pruning_plan = DG.get_pruning_plan(model.module_list[22].Conv2d, tp.prune_conv, idxs=pruning_idxs)
            logger.info(pruning_plan)

            # 4. execute this plan (prune the model)
            pruning_plan.exec()
            # 获得剪枝以后的参数量
            num_params_after_pruning = tp.utils.count_params(model)
            # 输出一下剪枝前后的参数量
            logger.info("  Params: %s => %s\n" % (num_params_before_pruning, num_params_after_pruning))
    model_ = {'epoch': ckpt['epoch'],
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
    torch.save(model_, '../model_data/Conv_pruning.pt')
    logger.info("剪枝完成\n")


@logger.catch
def layer_pruning(whole_model_weights):
    logger.add('../logs/layer_pruning.log', rotation='1 MB')
    ckpt = torch.load(whole_model_weights)
    model = ckpt['model']  # 模型的加载
    x = torch.randn(1, 3, 640, 640)
    # -----------------对整个模型的剪枝--------------------
    strategy = tp.strategy.L1Strategy()
    DG = tp.DependencyGraph()
    DG = DG.build_dependency(model, example_inputs=x)

    num_params_before_pruning = tp.utils.count_params(model)

    # 可以对照yolor结构进行剪枝
    """
    比如剪枝前30层：model.module_list[:31].Conv2d
    """
    included_layers = [layer.Conv2d for layer in model.module_list[:61] if
                      type(layer) is torch.nn.Sequential and layer.Conv2d]
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m in included_layers:
            pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=strategy(m.weight, amount=0.9))
            logger.info(pruning_plan)
            # 执行剪枝
            pruning_plan.exec()
    # 获得剪枝以后的参数量
    num_params_after_pruning = tp.utils.count_params(model)
    # 输出一下剪枝前后的参数量
    logger.info("  Params: %s => %s\n" % (num_params_before_pruning, num_params_after_pruning))
    # 剪枝完以后模型的保存(不要用torch.save(model.state_dict(),...))

    model_ = {'epoch': ckpt['epoch'],
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
    torch.save(model_, '../model_data/layer_pruning.pt')
    logger.info("剪枝完成\n")


layer_pruning('../whole_model.pt')
#Conv_pruning('../whole_model.pt')
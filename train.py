import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os, sys
import time
import tabulate
import data
import training_utils
import nets as models
import numpy as np
from parser_train import parser

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def cross_entropy(model, x, target, reduction="mean"):
    """standard cross-entropy loss function"""
    if model is not None:
        output = model(x)
    else:
        output = x

    loss = F.cross_entropy(output, target, reduction=reduction)

    if reduction is None or reduction == "none":
        loss = loss
    if reduction == 'mean':
        loss = torch.mean(loss)
    if reduction == 'sum':
        loss = torch.sum(loss)

    if model is not None:
        return loss, output

    return loss


def squared_loss(model, x, target, reduction="mean"):
    """
    В num_class / 2 (=5) меньше чем у Кати. Т.е. мой lr в num_class / 2 (=5) раз меньше чем Катин
    """
    if model is not None:
        output = model(x)
    else:
        output = x

    loss = (
                   torch.sum(torch.square(output), dim=1) -
                   2 * torch.gather(output, 1, target.view(-1, 1)).reshape(-1) + 1
           ) / output.shape[1]

    if reduction is None or reduction == "none":
        loss = loss
    if reduction == 'mean':
        loss = torch.mean(loss)
    if reduction == 'sum':
        loss = torch.sum(loss)

    if model is not None:
        return loss, output

    return loss

def check_si_name(n, model_name='ResNet18'):
    if model_name == 'ResNet18':
        return "conv1" in n or "1.bn1" in n or "1.0.bn1" in n or (("conv2" in n or "short" in n) and "4" not in n)
    elif model_name == 'ResNet18SI':
        return 'linear' not in n
    elif model_name == 'ResNet18SIAf':
        return ('linear' not in n and 'bn' not in n and 'shortcut.0' not in n)
    elif 'ConvNetSICI3WN' in model_name:
        return 'weight_v' in n
    elif 'ConvNet' in model_name:
        return 'conv_layers.0.' in n or 'conv_layers.3.' in n or 'conv_layers.7.' in n or 'conv_layers.11.' in n
    return False

def main():
    args = parser()
    args.device = None
    
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.cuda = True
    else:
        args.device = torch.device("cpu")
        args.cuda = False
        
    torch.backends.cudnn.benchmark = True
    set_random_seed(args.seed)

    # n_trials = 1
    
    print("Preparing base directory %s" % args.dir)
    os.makedirs(args.dir, exist_ok=True)

    # for trial in range(n_trials):
    trial = args.trial
    output_dir = args.dir + f"/trial_{trial}"
    
    ### resuming is modified!!!
    if args.resume_epoch > -1:
        resume_dir = output_dir
        output_dir = output_dir + f"/from_{args.resume_epoch}_for_{args.epochs}"
        if args.save_freq_int > 0:
            output_dir = output_dir + f"_save_int_{args.save_freq_int}"
        if args.noninvlr >= 0:
            output_dir = output_dir + f"_noninvlr_{args.noninvlr}"
        if args.fix_si_pnorm:
            output_dir = output_dir + f"_fix_si_pnorm"
        if args.seed > 1:
            output_dir = output_dir + '_seed{}'.format(args.seed)
         
    ### resuming is modified!!!
    print("Preparing directory %s" % output_dir)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "command.sh"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    print("Using model %s" % args.model)
    model_cfg = getattr(models, args.model)

    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    transform_train = model_cfg.transform_test if args.no_aug else model_cfg.transform_train
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        transform_train,
        model_cfg.transform_test,
        use_validation=not args.use_test,
        use_data_size = args.use_data_size,
        split_classes=args.split_classes,
        corrupt_train=args.corrupt_train
    )

    print("Preparing model")
    print(*model_cfg.args)

    # add extra args for varying names
    if 'ResNet18' in args.model:
        extra_args = {'init_channels':args.num_channels}
        if "SI" in args.model:
            extra_args.update({'linear_norm':args.init_scale})
    elif 'ConvNet' in args.model:
        extra_args = {'init_channels':args.num_channels, 'max_depth':args.depth,'init_scale':args.init_scale}
    elif args.model == 'LeNet':
        extra_args = {'scale':args.scale}
    else:
        extra_args = {}

    if args.same_init:
        set_random_seed(228)
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs,
                           **extra_args)
        set_random_seed(args.seed)
        
    else:
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs,
                           **extra_args)
    
    if args.same_last_layer and 'ConvNet' in args.model:
        set_random_seed(228)
        fin = nn.Linear(model.linear_layers[-1].in_features,model.linear_layers[-1].out_features) 
        alpha = args.init_scale
        W = fin.weight.data
        model.linear_layers[-1].weight.data = alpha * W / W.norm()
        model.linear_layers[-1].bias.data = fin.bias.data
        set_random_seed(args.seed)

    if args.same_last_layer and 'ResNet18' in args.model:
        set_random_seed(228)
        fin = nn.Linear(model.linear.in_features,model.linear.out_features,bias=False)
        alpha = args.init_scale
        W = fin.weight.data
        model.linear.weight.data = alpha * W / W.norm()
        set_random_seed(args.seed)
        
    model.to(args.device)

    if args.fix_elr:
        print("Training with fixed ELR")
        if args.momentum:
            print("WARNING: fixing ELR with momentum is ambiguous!")

        if args.model == 'ResNet18':
            params_dict = dict(model.named_parameters())
            param_groups = []

            # single pre-BN params first
            singles = training_utils.get_resnet_prebn_groups(g=1)
            param_groups.extend([{"params": [params_dict[n] for n in group]} for group in singles])

            # then pairs of pre-BN params
            pairs = training_utils.get_resnet_prebn_groups(g=2)
            param_groups.extend([{"params": [params_dict[n] for n in group]} for group in pairs])

            # then triples of pre-BN params
            triples = training_utils.get_resnet_prebn_groups(g=3)
            param_groups.extend([{"params": [params_dict[n] for n in group]} for group in triples])

            # finally others
            other_params = [p for n, p in params_dict.items() if all(n not in g for g in singles + pairs + triples)]
            param_groups.append({"params": other_params})

        elif 'ConvNetSI' in args.model or args.model == 'ResNet18SI':
            param_groups = [
                {'params': [p for n, p in model.named_parameters() if check_si_name(n, args.model)]},  # SI params are convolutions
                {'params': [p for n, p in model.named_parameters() if not check_si_name(n, args.model)]},  # other params
            ]

        else:
            raise ValueError("Fixing ELR currently is not allowed for this model!")

        # elr_coefs are coefs to multiply by lr * norm^2 for the fixed ELR, i.e.,
        # prebn_lr = elr_coef * lr * norm^2 => elr = prebn_lr / norm^2 = elr_coef * lr
        
        with torch.no_grad():
            si_pnorm_0 = np.sqrt(sum((p ** 2).sum().item() for p in param_groups[0]["params"]))
            lr = args.elr * si_pnorm_0 ** 2
    elif args.fix_all_elr:
        print("Training with all fixed ELRs")
        if args.momentum:
            print("WARNING: fixing ELR with momentum is ambiguous!")

        if 'ConvNetSI' in args.model or args.model == 'ResNet18SI':
            param_groups = [
                {'params': [p for n, p in model.named_parameters() if check_si_name(n, args.model)]},  # SI params are convolutions
                {'params': [p for n, p in model.named_parameters() if not check_si_name(n, args.model)]},  # other params
            ]
        else:
            raise ValueError("Fixing ELR currently is not allowed for this model!")

        with torch.no_grad():
            si_pnorm_0 = np.sqrt(sum((p ** 2).sum().item() for p in param_groups[0]["params"]) /
                                 sum(p.shape[0] for p in param_groups[0]["params"]))
            training_utils.fix_si_pnorms(model, si_pnorm_0, args.model)
            pnorm_0_sqr_total = sum((p ** 2).sum().item() for p in param_groups[0]["params"])
            lr = args.elr * pnorm_0_sqr_total
    else:
        param_groups = model.parameters()
        si_pnorm_0 = None
        lr = args.lr_init
        elr_coefs = None
        
    if args.noninvlr >= 0:
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if check_si_name(n, args.model)]},  
            {'params': [p for n, p in model.named_parameters() if not check_si_name(n, args.model)],'lr':args.noninvlr}, 
        ]

    optimizer = torch.optim.SGD(param_groups, 
                                lr=lr, 
                                momentum=args.momentum, 
                                weight_decay=args.wd)

    if args.cosan_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
    epoch_from = args.resume_epoch + 1
    epoch_to = epoch_from + args.epochs
    print(f"Training from {epoch_from} to {epoch_to - 1} epochs")

    if args.resume_epoch > -1:
        # Warning: due to specific lr schedule, resuming is generally not recommended!
        print(f"Loading checkpoint from the {args.resume_epoch} epoch")
        state = training_utils.load_checkpoint(resume_dir, args.resume_epoch)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        if args.noninvlr >= 0:
            optimizer.param_groups[1]["lr"] = args.noninvlr
            
    else:
        #save init
        train_res = {"loss": None, "accuracy": None}
        test_res = {"loss": None, "accuracy": None}
        
        def save_epoch(epoch):
            training_utils.save_checkpoint(
                output_dir,
                epoch,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
                train_res=train_res,
                test_res=test_res
            )

        save_epoch(0)
        epoch_from +=1

            
    for epoch in range(epoch_from, epoch_to+1):
        train_epoch(model, loaders, squared_loss if args.use_squared_loss else cross_entropy, optimizer,
                    epoch=epoch, 
                    end_epoch=epoch_to+1, 
                    eval_freq=args.eval_freq, 
                    save_freq=args.save_freq,
                    save_freq_int=args.save_freq_int,
                    fix_elr = args.fix_elr,
                    fix_all_elr=args.fix_all_elr,
                    si_pnorm_0=si_pnorm_0,
                    output_dir=output_dir,
                    lr_init=lr,
                    lr_schedule=not args.no_schedule,
                    noninvlr=args.noninvlr,
                    c_schedule=args.c_schedule,
                    d_schedule=args.d_schedule,
                    fbgd=args.fbgd,
                    cosan_schedule = args.cosan_schedule,
                    model_name = args.model)
        if args.cosan_schedule:
            scheduler.step()

    print("model ", trial, " done")


def train_epoch(model, loaders, criterion, optimizer, epoch, end_epoch,
                eval_freq=1, save_freq=10, save_freq_int=0, fix_elr=False, fix_all_elr = False,
                si_pnorm_0=None,output_dir='./',
                lr_init=0.01, lr_schedule=True, noninvlr = -1, c_schedule=None, d_schedule=None,
                fbgd=False, cosan_schedule = False, model_name = 'ResNet18'):

    time_ep = time.time()

    if not cosan_schedule:
        if not lr_schedule:
            lr = lr_init
        elif c_schedule > 0:
            lr = training_utils.c_schedule(epoch, lr_init, end_epoch, c_schedule)
        elif d_schedule > 0:
            lr = training_utils.d_schedule(epoch, lr_init, end_epoch, d_schedule)
        else:
            lr = training_utils.schedule(epoch, lr_init, end_epoch, swa=False)
        if noninvlr >= 0:
            training_utils.adjust_learning_rate_only_conv(optimizer, lr)
        else:
            training_utils.adjust_learning_rate(optimizer, lr)
    else:
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            break

    train_res = training_utils.train_epoch(loaders["train"], model, criterion, optimizer, fbgd=fbgd,
                                           save_freq_int=save_freq_int, epoch = epoch,
                                           output_dir=output_dir, fix_elr = fix_elr, fix_all_elr = fix_all_elr,
                                           si_pnorm_0=si_pnorm_0, model_name = model_name)
    if (
        epoch == 1
        or epoch % eval_freq == eval_freq - 1
        or epoch == end_epoch - 1
    ):
        test_res = training_utils.eval(loaders["test"], model, criterion)
    else:
        test_res = {"loss": None, "accuracy": None}
        
    def save_epoch(epoch):
        training_utils.save_checkpoint(
            output_dir,
            epoch,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            train_res=train_res,
            test_res=test_res
        )

    if save_freq is None:
        if training_utils.do_report(epoch):
            save_epoch(epoch)
    elif epoch % save_freq == 0:
        save_epoch(epoch)
        
    time_ep = time.time() - time_ep
    values = [
        epoch,
        lr,
        train_res["loss"],
        train_res["accuracy"],
        test_res["loss"],
        test_res["accuracy"],
        time_ep,
    ]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 1:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

if __name__ == '__main__':
    main()

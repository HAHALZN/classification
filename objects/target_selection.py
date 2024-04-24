import torch
from tqdm import tqdm
from objects.losses import Entropy


def target_selection(target_loader, tgt_net_f, tgt_net_b,
                     src_nets_f, src_nets_b, src_nets_c, checkpoints, args):
    min_ent = torch.inf
    min_source = ''
    criterion = Entropy()
    outputs = dict()
    with torch.no_grad():
        # 选择一个在目标域中，熵值最小的模型，作为最终的模型
        for i, (images, _, _) in tqdm(enumerate(target_loader)):
            for source in args.source:
                # compute and output
                output = src_nets_c[source](src_nets_b[source](src_nets_f[source](images)))
                if i == 0:
                    outputs[source] = output.cpu()
                else:
                    outputs[source] = torch.cat((outputs[source], output.cpu()), 0)
        for source in args.source:
            print("=> calculating condition entropy from source model {}".format(source))
            ent = criterion(outputs[source])
            print("   the condition entropy from source model {} is {:.4f}".format(source, ent.item()))
            if ent.item() < min_ent:
                min_ent = ent.item()
                min_source = source

    print("=> the minimum condition entropy is from source model {},"
          " the value is {:.4f}, selected".format(min_source, min_ent))
    tgt_net_f.load_state_dict(checkpoints[min_source]['state_dict_net_f'])
    tgt_net_b.load_state_dict(checkpoints[min_source]['state_dict_net_b'])

    return tgt_net_f, tgt_net_b


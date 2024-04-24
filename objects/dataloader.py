import os
from torch.utils.data import DataLoader
from torchvision import transforms

from objects.dataset import ImageList
from utils.imbalanced import ImbalancedDatasetSampler


def get_target_loader(args, transform=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    target_dir = os.path.join(args.data, args.target)
    target_list = open(os.path.join(target_dir,'new_labels', 'train.txt')).readlines()[1:] + \
                  open(os.path.join(target_dir,'new_labels', 'test.txt')).readlines()[1:]
    target_list = [os.path.join(target_dir,'imgs_cropped_enhanced') + '/' + line.strip() for line in target_list]
    target_dataset = ImageList(target_list, args, transform=transform, delimiter=args.delimiter)
    args.num_classes = target_dataset.num_classes

    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True, sampler=None,drop_last=True)

    return target_loader




def get_label_unlabel_loader(args, transform=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    target_loader={}
    target_dir = os.path.join(args.data, args.target)
    label_list = open(os.path.join(target_dir, 'train_label.txt')).readlines() 
    unlabel_list       =       open(os.path.join(target_dir, 'train_unlabel.txt')).readlines()
    all_list = open(os.path.join(target_dir, 'train.txt')).readlines() + \
                  open(os.path.join(target_dir, 'test.txt')).readlines()
    label_dataset = ImageList(label_list, args, transform=transform, delimiter=args.delimiter)
    unlabel_dataset = ImageList(unlabel_list, args, transform=transform, delimiter=args.delimiter)
    all_dataset = ImageList(all_list, args, transform=transform, delimiter=args.delimiter)


    args.num_classes = label_dataset.num_classes

    target_loader["train_label"] = DataLoader(label_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True, sampler=None,drop_last=True)
    target_loader['train_unlabel'] = DataLoader(unlabel_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True, sampler=None,drop_last=True)
    target_loader['all'] = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True, sampler=None,drop_last=True)

    return target_loader




def get_source_target_loaders(args, train_transform=None, test_transform=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    source_loaders, target_loaders = dict(), dict()
    source_dir = os.path.join(args.data, args.source)
    source_loaders["train"], source_loaders["test"] = get_train_test_loaders(source_dir, args,
                                                                             train_transform=train_transform)

    for target in args.target:
        if target != args.source:
            target_dir = os.path.join(args.data, target)
            target_list = open(os.path.join(target_dir, 'train.txt')).readlines() + \
                          open(os.path.join(target_dir, 'test.txt')).readlines()
            target_dataset = ImageList(target_list, args, transform=test_transform, delimiter=args.delimiter)

            if args.num_classes != target_dataset.num_classes:
                print("{} and {} datasets have the different number of classes, skipped.".format(args.source, target))
                continue

            target_loader = DataLoader(target_dataset, batch_size=512, shuffle=False,
                                       num_workers=args.workers, pin_memory=True, sampler=None,drop_last=True)
            target_loaders[target] = target_loader

    return source_loaders, target_loaders


def get_train_test_loaders(data_dir, args, train_transform=None, test_transform=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    train_list = open(os.path.join(data_dir, 'train.txt')).readlines()
    test_list = open(os.path.join(data_dir, 'test.txt')).readlines()
    train_dataset = ImageList(train_list, args, transform=train_transform, delimiter=args.delimiter)
    test_dataset = ImageList(test_list, args, transform=test_transform, delimiter=args.delimiter)

    args.num_classes = train_dataset.num_classes

    train_sampler = None
    if args.imbalanced_sampler:
        train_sampler = ImbalancedDatasetSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False, num_workers=args.workers, pin_memory=True,drop_last=True)

    return train_loader, test_loader





def get_label_unlabel_target_loaders(args, train_transform=None, test_transform=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    source_loaders, target_loaders = dict(), dict()
    source_dir = os.path.join(args.data, args.source)
    _, source_loaders["test"] = get_train_test_loaders(source_dir, args,
                                                                             train_transform=train_transform)

    source_label_list = open(os.path.join(source_dir, 'train_label.txt')).readlines() 
    source_unlabel_list = open(os.path.join(source_dir, 'train_unlabel.txt')).readlines()
    source_label_dataset = ImageList(source_label_list, args, transform=train_transform, delimiter=args.delimiter)
    source_unlabel_dataset = ImageList(source_unlabel_list, args, transform=train_transform, delimiter=args.delimiter)


    source_loaders["train_label"] = DataLoader(
        source_label_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)
    source_loaders["train_unlabel"]= DataLoader(
        source_unlabel_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)


    for target in args.target:
        if target != args.source:
            target_dir = os.path.join(args.data, target)
            target_list = open(os.path.join(target_dir, 'train.txt')).readlines() + \
                          open(os.path.join(target_dir, 'test.txt')).readlines()
            target_dataset = ImageList(target_list, args, transform=test_transform, delimiter=args.delimiter)

            if args.num_classes != target_dataset.num_classes:
                print("{} and {} datasets have the different number of classes, skipped.".format(args.source, target))
                continue

            target_loader = DataLoader(target_dataset, batch_size=512, shuffle=False,
                                       num_workers=args.workers, pin_memory=True, sampler=None,drop_last=True)
            target_loaders[target] = target_loader

    return source_loaders, target_loaders






def get_lup_transformer(aug_type='moco', mean_type='lup', height=256, width=128, two_crop=True):
    """ LUP dataset is only used for training. """
    if not mean_type.lower() in STAT_DICT:
        raise TypeError('Unknown mean value status type: {}'.format(mean_type))
    mean = STAT_DICT[mean_type.lower()]['mean']
    std = STAT_DICT[mean_type.lower()]['std']
    normalizer = T.Normalize(mean=mean, std=std)

    if aug_type == 'reid':
        augmentation = [RandomSizedRectCrop(height, width, interpolation=3)]
    else:
        augmentation = [T.RandomResizedCrop((height, width), scale=(0.2, 1.))]

    aug_dict = {'ori': {'cj': 0.8, 'cj-s': 0.4, 'gs': 0.2, 'gb': 0.5, },
                'ori-cj': {'gs': 0.2, 'gb': 0.5, },
                'ori-gs': {'cj': 0.8, 'cj-s': 0.4, 'gb': 0.5, },
                'ori-gb': {'cj': 0.8, 'cj-s': 0.4, 'gs': 0.2, },
                'ori+re': {'cj': 0.8, 'cj-s': 0.4, 'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.4},
                'ori-cj+re': {'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.4},
                'ori-gs+re': {'cj': 0.8, 'cj-s': 0.4, 'gb': 0.5, 're': 0.5, 're-s': 0.4},
                'ori-gb+re': {'cj': 0.8, 'cj-s': 0.4, 'gs': 0.2, 're': 0.5, 're-s': 0.4},
                'ori-cj-gb+re': {'gs': 0.2, 're': 0.5, 're-s': 0.4},
                'ori+wcj+re': {'cj': 0.8, 'cj-s': 0.2, 'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.4},
                'ori+wcj+sre': {'cj': 0.8, 'cj-s': 0.2, 'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.6},
                'ori+wwcj+re': {'cj': 0.8, 'cj-s': 0.1, 'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.4},

                'ori+sre': {'cj': 0.8, 'cj-s': 0.4, 'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.6},
                'ori-cj-gb+sre': {'gs': 0.2, 're': 0.5, 're-s': 0.6},
                'ori-cj+sre': {'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.6},
                'ori-cj+wre': {'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.2},
                'ori-cj+ssre': {'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.8},

                'moco': {'cj': 1.0, 'cj-s': 0.4, 'gs': 0.2, },
                'reid': {'re': 0.5, 're-s': 0.4},
                'reid-color': {'cj': 0.8, 'cj-s': 0.3, 're': 0.5, 're-s': 0.4},
                'reid-affine': {'rp': 0.5, 'rp-s': 0.4, 'ra': 0.5, 're': 0.5, 're-s': 0.4}
                }

    if not aug_type in aug_dict:
        raise TypeError('Unknown augmentation type: {}'.format(aug_type))
    aug_kwargs = aug_dict[aug_type]
    color_jitter_p = aug_kwargs.get('cj', 0.)
    color_jitter_s = aug_kwargs.get('cj-s', 0.4)
    if 0. < color_jitter_p < 1.:
        augmentation.append(T.RandomApply([T.ColorJitter(
            brightness=color_jitter_s, contrast=color_jitter_s)], color_jitter_p))
    # elif color_jitter_p == 1:
    #     augmentation.append(T.ColorJitter(color_jitter_s, color_jitter_s, color_jitter_s, color_jitter_s))

    random_affine_p = aug_kwargs.get('ra', 0.)
    if 0. < random_affine_p < 1.:
        augmentation.append(T.RandomApply([T.RandomAffine(
            degrees=(-10, 10), translate=(0.1, 0.2), scale=(0.9, 1.1), shear=(5))], random_affine_p))

    # gray_scale_p = aug_kwargs.get('gs', 0.)
    # if 0. < gray_scale_p <= 1.:
    #     augmentation.append(T.RandomGrayscale(p=gray_scale_p))

    gaussian_blur_p = aug_kwargs.get('gb', 0.)
    if 0. < gaussian_blur_p < 1.:
        augmentation.append(T.RandomApply([GaussianBlur([.1, 2.])], p=gaussian_blur_p))

    augmentation.append(T.RandomHorizontalFlip())
    augmentation.append(T.ToTensor())
    augmentation.append(normalizer)

    random_erase_p = aug_kwargs.get('re', 0.)
    random_erase_s = aug_kwargs.get('re-s', 0.4)
    if 0. < random_erase_p < 1.:
        augmentation.append(RandomErasing(probability=random_erase_p, sh=random_erase_s))

    transformer = T.Compose(augmentation)
    if two_crop:
        transformer = TwoCropsTransform(transformer)
    return transformer

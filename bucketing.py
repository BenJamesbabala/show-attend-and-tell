import numpy as np

def bucketing(captions, images, batch_size, only_train=True, valid_num=None):
    """
    args:
        valid_num is set only when only_train is False
    (in my case valid_num = 4100 when len(captions)=40460)
    """
    assert len(captions)==len(images)
    assert only_train or valid_num is not None

    if only_train:
        captions_list = [captions]
        images_list = [images]
        num_list = [len(captions)]
    else:
        valid_captions = captions[len(captions)-valid_num:]
        train_captions = captions[:len(captions)-valid_num]
        valid_images = images[len(images)-valid_num]
        train_images = images[len(images)-valid_num]
        
        captions_list = [train_captions, valid_captions]
        images_list = [train_images, valid_images]
        num_list = [len(captions)-valid_num, valid_num]

    new_captions_list, new_images_list, maxlen_list = [], [], []

    for captions, images, num in zip(captions_list, images_list, num_list):
        caption_lens = [len(i) for i in captions]
        order_shuffler = [i[0] for i in sorted(enumerate(caption_lens),
            key=lambda x:x[1])]

        new_captions = []
        new_images = []
        for i in order_shuffler:
            new_captions.append(captions[i])
            new_images.append(images[i])

        steps = num/batch_size
        maxlen = [np.max(map(lambda x : len(x), new_captions[i*batch_size:(i+1)*batch_size])) for i in range(steps)]

        new_captions_list.append(new_captions)
        new_images_list.append(new_images)
        maxlen_list.append(maxlen)

    if only_train:
        return new_captions_list[0], new_images_list[0], maxlen_list[0]
    else:
        return new_captions_list[0], new_images_list[0], maxlen_list[0], new_captions_list[1], new_images_list[1], maxlen_list[1]

        

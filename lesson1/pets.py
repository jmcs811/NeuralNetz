from fastai.vision import *
from fastai.metrics import error_rate

bs = 64

# get data
path = untar_data(URLs.PETS)

path_anno = path/'annotations'
path_img = path/'images'

fname = get_image_files(path_img)

np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fname, pat, ds_tfms=get_transforms(), size=224, bs=bs)


# train
learn = cnn_learner(data, models.resnet34, metrics=[error_rate, accuracy])

learn.fit_one_cycle(3)

learn.save('stage-1')
learn.export('trained_1.pkl')
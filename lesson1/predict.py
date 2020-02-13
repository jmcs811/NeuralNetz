from fastai.vision import *
from fastai.metrics import error_rate

img = open_image('2020-02-10.jpg')
img.resize(torch.Size([img.shape[0], 224, 224]))

learn = load_learner('./')
print(learn.predict(img)[0])

import numpy as np
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y

class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            if len(np.array(self.im).shape)==2:
                self.im = self.im.convert('RGB')
            self.draw = ImageDraw.Draw(self.im)
            self.font = ImageFont.load_default()
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        self.draw.rectangle(box, width=self.lw, outline=color)  # box
        if label:
            w, h = self.font.getsize(label)  # text width, height
            outside = box[1] - h >= 0  # label fits outside box
            self.draw.rectangle(
                (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                    box[1] + 1 if outside else box[1] + h + 1),
                fill=color,
            )
            # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
            self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

if __name__ == "__main__":
    # pic = np.array(Image.open('/home/yukai/large/UnifiedConfusionMatrix/datasets/coco/images/000000000009.jpg'))
    # # with open('/data/zhaowei/ConfusionMatrix/datasets/coco/predicts/000000074842.txt') as f:
    # with open('/home/yukai/large/UnifiedConfusionMatrix/datasets/coco/predicts/000000000009.txt') as f:
    #     gt = np.array([x.split() for x in f.read().strip().splitlines() if len(x)])
    # gt = gt.astype(np.float64)
    # anno = Annotator(pic, pil=True)
    # pic = Image.fromarray(pic)
    # amp = np.array([pic.width,pic.height,pic.width,pic.height])
    # for i in range(len(gt)):
    #     anno.box_label(xywh2xyxy(gt[i,2:6]*amp).tolist(), color=(255,0,0), label=''+str(gt[i,0]))
    # Image.fromarray(anno.result()).save('./123.jpg')
    # exit(0)

    from data.dataCtrler import DataCtrler
    dataCtrler = DataCtrler()
    dataCtrler.process("/data/zhaowei/ConfusionMatrix/datasets/coco/", "/data/zhaowei/ConfusionMatrix/backend/buffer/")
    samples = dataCtrler.filterSamples({
        "label": [28],
        "predict": [0],
        # "label_size": [0.96, 1]
    })[0]
    pairs = dataCtrler.predict_label_pairs[samples]
    print(len(samples))
    # print(dataCtrler.predict_label_ious[samples])

    for pr, gt in pairs:
        try:
            # print(pr,gt)
            image_id = dataCtrler.raw_label2imageid[gt]
            pr_name = np.array(dataCtrler.names)[int(dataCtrler.raw_predicts[pr, 0])]
            gt_name = np.array(dataCtrler.names)[int(dataCtrler.raw_labels[gt, 0])]
            pic_name = os.path.join(dataCtrler.images_path, os.listdir(dataCtrler.images_path)[image_id])
            dir_name = os.path.join('/home/yukai/large/UnifiedConfusionMatrix',gt_name+'-'+pr_name, pic_name.split('/')[-1].split('.')[0])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            pic = np.array(Image.open(pic_name))
            anno = Annotator(pic, pil=True)
            pic = Image.fromarray(pic)
            amp = np.array([pic.width,pic.height,pic.width,pic.height])
            anno.box_label(xywh2xyxy(dataCtrler.raw_predicts[pr, 1:5]*amp).tolist(), color=(255,0,0))
            anno.box_label(xywh2xyxy(dataCtrler.raw_labels[gt, 1:5]*amp).tolist(), color=(0,255,0))
            Image.fromarray(anno.result()).save(os.path.join(dir_name, '1.jpg'))
            for i in range(dataCtrler.imageid2raw_label[image_id][0], dataCtrler.imageid2raw_label[image_id][1]):
                anno.box_label(xywh2xyxy(dataCtrler.raw_labels[i, 1:5]*amp).tolist(), color=(0,255,0), label=''+np.array(dataCtrler.names)[int(dataCtrler.raw_labels[i, 0])])
            for i in range(dataCtrler.imageid2raw_predict[image_id][0], dataCtrler.imageid2raw_predict[image_id][1]):
                anno.box_label(xywh2xyxy(dataCtrler.raw_predicts[i, 1:5]*amp).tolist(), color=(255,0,0), label=''+np.array(dataCtrler.names)[int(dataCtrler.raw_predicts[i, 0])])
            Image.fromarray(anno.result()).save(os.path.join(dir_name, 'all.jpg'))
        except:
            print(os.listdir(dataCtrler.images_path)[dataCtrler.raw_label2imageid[gt]])
import torch
from itertools import product as product
import numpy as np
from math import ceil

"""
PriorBox computes coordinates of prior boxes by following:

    First calculates center_x and center_y of prior box:
        W≡WidthOfImage
        H≡HeightOfImage
            
    If step equals 0:
        centerx=(w+0.5)
        centery=(h+0.5)
            
    else:
        centerx=(w+offset)*step
        centery=(h+offset)*step
        w⊂(0,W)
        h⊂(0,H)

    Then, for each s⊂(0,minsizes) calculates coordinates of prior boxes:
        xmin=mean(centerx-s)/W
        ymin=mean(centery-s)/H
        xmax=mean(centerx+s)/W
        ymin=mean(centery+s)/H
"""

class PriorBox(object):
    def __init__(self,image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32] #distance between box centers
        self.clip = False #output tensor between [0,1]
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    #rel size:  min_size
                    s_kx = min_size / self.image_size[1] 
                    s_ky = min_size / self.image_size[0]
                    
                    #center x,y
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    
                    for cy, cx in product(dense_cy, dense_cx):
                        #define the anchors
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
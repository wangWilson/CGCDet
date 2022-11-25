import numpy as np
import cv2
from PIL import Image, ImageDraw
import math


def draw_a_rectangel_in_img(draw_obj, box, color, width, method):
    '''
    use draw lines to draw rectangle. since the draw_rectangle func can not modify the width of rectangle
    :param draw_obj:
    :param box: [x1, y1, x2, y2]
    :return:
    '''
    # color = (0, 255, 0)
    if method == 0:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        top_left, top_right = (x1, y1), (x2, y1)
        bottom_left, bottom_right = (x1, y2), (x2, y2)

        draw_obj.line(xy=[top_left, top_right],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[top_left, bottom_left],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[bottom_left, bottom_right],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[top_right, bottom_right],
                      fill=color,
                      width=width)
    else:
        box=list(map(int,box))
        x1, y1, x2, y2,x3, y3, x4, y4 = box
        print(box)
        cnt=np.array([[x1, y1],[x2, y2],[x3, y3],[x4, y4]])
        rect=cv2.minAreaRect(cnt)
        x_c, y_c, w, h, theta = rect[0][0],rect[0][1],rect[1][0],rect[1][1],rect[2]
        rect = ((x_c, y_c), (w, h), theta)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        draw_obj.line(xy=[(rect[0][0], rect[0][1]), (rect[1][0], rect[1][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[1][0], rect[1][1]), (rect[2][0], rect[2][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[2][0], rect[2][1]), (rect[3][0], rect[3][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[3][0], rect[3][1]), (rect[0][0], rect[0][1])],
                      fill=color,
                      width=width)


def trans_label2angle(bbox):

    x1,y1,x2,y2,x3,y3,x4,y4=bbox
    cnt=np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    rect=cv2.minAreaRect(cnt)
    return [rect[0][0],rect[0][1],rect[1][0],rect[1][1],rect[2]]


def draw_in_img(bboxs,rbboxs):
   image0 = Image.new('RGB', (1024, 1024), (255,255,255))
   image = ImageDraw.Draw(image0)

   print(2222)

   for k in range(bboxs.shape[0]):
      bbox_H=bboxs[k]
      rbbox=rbboxs[k]
      try:
         draw_a_rectangel_in_img(image,bbox_H,'AliceBlue',3,0)
         draw_a_rectangel_in_img(image,rbbox,'DarkOrange',3,1)
         print('true')
      except:
        print('error')

      image0.save('img_{}.png'.format(k))

   

if __name__ == '__main__':
   bbox=[890,450,989,506,974,532,875,475]
   bbox_a=trans_label2angle(bbox)
   print(bbox_a)
   
   bbox_H=[875,450,989,532]
   center=int((989-875)/2+875),int((532-450)/2+450)
   w,h=bbox_H[2]-bbox_H[0],bbox_H[3]-bbox_H[1]
   print(center,w,h)
   
   
   image0 = Image.new('RGB', (1024, 1024), (255,255,255))
   image = ImageDraw.Draw(image0)
   draw_a_rectangel_in_img(image,bbox_H,'AliceBlue',3,0)
   draw_a_rectangel_in_img(image,bbox,'DarkOrange',3,1)
   image0.save('img.png')
   
   print(math.cos(math.pi/180*60))
   
   cos=math.cos(math.pi/180*(1*bbox_a[-1]))
   sin=math.sin(math.pi/180*(1*bbox_a[-1]))
   # print(np.array([bbox_a[2],bbox_a[3]]).shape)
   print(np.array([[cos,-1*sin],[sin,cos]]))
   
   b=np.array([bbox_a[2],bbox_a[3]])@np.array([[cos,-1*sin],[sin,cos]])
   b=np.array([[cos,-1*sin],[sin,cos]])@np.array([30,114]).T
   print(b)



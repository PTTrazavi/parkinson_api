from PIL import Image
import sys, os
import requests, urllib.request
from io import BytesIO
import time, datetime
from django.core.files.base import ContentFile
from .models import Imageupload
from django.shortcuts import get_object_or_404

from keras.models import load_model
from keras import activations
from keras.preprocessing import image
from vis.visualization import overlay,visualize_cam
from vis.utils import utils
import numpy as np

from PIL import ImageDraw, ImageFont

#add frame to the image
def imgtool(pk_o): #pk of the original image
    photo_o = get_object_or_404(Imageupload, pk=pk_o)

    #get file name and extension
    f_n = photo_o.image_file.name.split("/")[-1].split(".")[0]
    f_e = photo_o.image_file.name.split('.')[-1]

    out_f_name = f_n + "_matting." + f_e
    #Load the input image

    if "https://storage" in photo_o.image_file.url: # for GCS
        response = requests.get(photo_o.image_file.url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(photo_o.image_file)
    #image process starts here
    width = 35
    # load pixels of pictures
    px = img.load()
    for x in range(0,img.size[0]):
        for y in range(0,img.size[1]):
            # add blue frame here
            if x < width or y < width or x > img.size[0] - width or y > img.size[1] - width :
                px[x,y] = 129, 216, 208 ,255

    img_io = BytesIO()
    img.save(img_io, format='JPEG')
    img_content = ContentFile(img_io.getvalue(), out_f_name)
    photo_o.result_file = img_content
    photo_o.readiness = "2"
    photo_o.save()

    return photo_o.result_file.url

#parkinson detection
def p_detection(pk_o):
    photo_o = get_object_or_404(Imageupload, pk=pk_o)
    img_name = photo_o.image_file.url
    #### SETTINGS ####
    # labels = {0:'ET',1:'NORMAL',2:'PD'}
    labels = {0:'NORMAL',1:'PD'}

    model = load_model('model/pd_normal_vgg16_88_bg.h5') #pd_normal_vgg16_88_bg.h5  pd_et_normal_vgg16.h5
    # model = load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model/pd_normal_vgg16_88_bg.h5'))

    # Modify the last layer of the model to make CAM model
    model_h = model
    print("Remove Activation from Last Layer")
    model_h.layers[-1].activation = activations.linear
    print("Done. Now Applying changes to the model ...")
    model_h = utils.apply_modifications(model_h)

    #get file name and extension
    f_n = img_name.split("/")[-1].split(".")[0]
    f_e = img_name.split(".")[-1]
    #remove special charactor
    tbd = ['!','@','#','$','%','^','&','*','(',')','-','+','=']
    for i in tbd:
        f_n = f_n.replace(i,'')
    #if the extension is too long make it .jpg
    if len(f_e) > 7:
        f_e = "jpg"
    out_f_name = f_n + "_out." + f_e
    #Load the input image
    if "http" in photo_o.image_file.url: # for GCS
        response = requests.get(photo_o.image_file.url)
        img = Image.open(BytesIO(response.content))
        img.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../media/images/temp.jpg'), format='JPEG')
        img_RGB = image.load_img(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../media/images/temp.jpg'), target_size=(256,256)) #RGB
    else:
        img_RGB = image.load_img(img_name[1:], target_size=(256,256)) #RGB

    #image process starts here
    img_RGB = image.img_to_array(img_RGB)
    print("resized image shape:", img_RGB.shape)

    x = np.expand_dims(img_RGB, axis=0)  / 255
    print("shape of the x for visualize_cam:", x.shape)

    # classifer 2類
    classes = model.predict(x)
    print("predicted class:", classes)
    if classes[0] <=0.5:
        class_name = labels[0]
    else:
        class_name = labels[1]
    print('the class name is:',class_name)

    # classifer 3類
    # classes = model.predict(x)
    # print('predicted class:',classes[0])
    # print("predicted class:", np.argmax(classes[0]))
    # class_name = labels[np.argmax(classes[0])]
    # print('the class name is:',class_name)

    # Generate heatmap of the predicted image
    heatmap = visualize_cam(model_h, -6, filter_indices=1, seed_input=x[0,:,:,:])
    print("shape of heatmap:",heatmap.shape)
    # combine two pic
    img = Image.fromarray(overlay(img_RGB, heatmap).astype('uint8'))

    ImageDraw.Draw(img).text(
        xy = (10, 10),  # Coordinates
        text = 'AI predicted result is: ' + class_name,  # Text
        color = (255, 255, 255)  # Color
    )
    # save the result image
    img_io = BytesIO()
    img.save(img_io, format='JPEG')
    img_content = ContentFile(img_io.getvalue(), out_f_name)

    photo_o.result_file = img_content
    photo_o.readiness = "2"
    photo_o.save()

    return photo_o.result_file.url

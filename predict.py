import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import json
from PIL import Image

image_size = 224

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size)).numpy()
    image /= 255
    return image

def predict(image_path, model, top_k,class_names):
    im = Image.open(image_path)
    test_image = np.asarray(im)

    processed_test_image = process_image(test_image)
    final_image=np.expand_dims(processed_test_image, axis=0)

    pred=model.predict(final_image)

    top_k_values, top_k_indices = tf.nn.top_k(pred, k=top_k)
    top_k_values = top_k_values.numpy()
    top_k_indices = top_k_indices.numpy()
    pred_names=[]
    pred_classes=top_k_indices[0]
    for i in pred_classes:
    #print(i)
        pred_names.append(class_names[str(i+1)])
    #print(pred_names)

    return top_k_values, top_k_indices ,pred_names,processed_test_image

if __name__ == '__main__':
    
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')# image path
    parser.add_argument('arg2')# model to be loaded
    parser.add_argument('--top_k',default=5)
    parser.add_argument('--category_names',default = "label_map.json") 
    

    
    args = parser.parse_args()
    print(args)
    
    print('arg1:', args.arg1)
    print('arg2:', args.arg2)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    image_path = args.arg1
    import re

    s=image_path
    n=re.findall('\w+',s)

    act_name=n[-2]

    model = tf.keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = args.top_k

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
   
    top_k_values, top_k_indices ,pred_names,processed_test_image = predict(image_path, model, top_k,class_names)    
    
    print('image actual name:', act_name)    
    print('Propabilties:', top_k_values)
    print('Classes Keys:', top_k_indices)
    print('Classes namesss:', pred_names)
#To check:
# python predict.py ./test_images/wild_pansy.jpg image_model.h5
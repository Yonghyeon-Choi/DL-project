"""
reference
https://medium.com/analytics-vidhya/visualizing-activation-heatmaps-using-tensorflow-5bdba018f759
https://blog.daum.net/geoscience/1263
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


model = InceptionV3(weights='imagenet')

bestDir = 'bestCaseCut'
worstDir = 'worstCaseCut'

bestCases = os.listdir(bestDir)
worstCases = os.listdir(worstDir)


def gradCAM(orig, Dir, filename, bestorworst, intensity=0.5, res=299):
    img = image.load_img(orig, target_size=(res, res))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pred = model.predict(x)
    label = decode_predictions(pred)[0][0][1]

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_93')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((8, 8))

    img = cv2.imread(orig)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    img = heatmap * intensity + img

    cv2.imwrite(os.path.join(Dir, filename[0:-4]+"_heatmap.png"), img)

    img1 = cv2.imread(orig)
    img2 = cv2.imread(os.path.join(Dir, filename[0:-4]+"_heatmap.png"))

    fig = plt.figure()
    rows = 1
    cols = 2

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original image')
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.set_title('Label : '+label)
    ax2.axis("off")
    if bestorworst:
        plt.savefig(os.path.join("result", "best_" + filename[0:-4] + "_result.png"))
    else:
        plt.savefig(os.path.join("result", "worst_" + filename[0:-4] + "_result.png"))


for file in bestCases:
    gradCAM(os.path.join(bestDir, file), bestDir, file, 1)

for file in worstCases:
    gradCAM(os.path.join(worstDir, file), worstDir, file, 0)

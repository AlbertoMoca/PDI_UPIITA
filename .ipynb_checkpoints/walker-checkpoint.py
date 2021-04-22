import numpy as np
from matplotlib import pyplot as plt
from skimage.util import random_noise
from tqdm import tqdm
from PIL import Image

to_gray = lambda img:np.sum(img.copy() * [.299,.587,.114], axis=-1)

#Segmentation function
def walker(img, initial_seed, center, tolerance, iterations):
    
    seeds = initial_seed
    print("Loading")
    for j in tqdm(range(iterations)):
        img_copy = img.copy()
        for i in seeds:
            mask = img[i[0]-1:i[0]+2,i[1]-1:i[1]+2].copy()
            img[i[0]-1:i[0]+2,i[1]-1:i[1]+2][np.abs(mask-center) <= tolerance] = - 256
        seeds = np.argwhere(img != img_copy)
    img[img == -256] = 255
    return img


#Load data 
img = Image.open('brain.jpg')
img_g = to_gray(np.array(img)).astype(int)


fig, ax = plt.subplots(1,2)
ax[0].imshow(img_g, cmap="gray")


# Event Handler
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

    x, y = int(event.xdata), int(event.ydata)
    
    tol = int( input("Tolerance:"))
    num_it = int( input("Num of iterations:"))

    segmented_img = walker(
	        img_g, 
            np.array([[y, x]]), 
            img_g[y, x], 
            tol, 
            num_it
        )
    ax[1].imshow(segmented_img, cmap="gray")

    plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
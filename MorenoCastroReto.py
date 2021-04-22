import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

images_names = []
for i in range(8):
	print(f"Nombre de la imagen {i}:")
	images_names.append(input())

images = [np.array(Image.open(i)) for i in images_names]


def extract_bit(x):
    bits = bin(x)[2:]
    bits = "0"*(8-len(bits))+bits
    bits = bits[::-1]
    return np.array([int(i) for i in bits])


vextract_bit = np.vectorize(extract_bit, signature='()->(n)')

final = np.zeros(images[0].shape)

fig, axs = plt.subplots(1, 8)
fig.set_size_inches(10, 4)

print("Extrayendo capas ocultas")

for i,j in enumerate(images[::-1]):
    hidden = vextract_bit(j)[:,:,0]
    final +=  hidden * 2**i
    axs[i].imshow(hidden, cmap="gray")
plt.show()

plt.imshow(final, cmap="gray")
plt.show()
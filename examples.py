"""Examples of DCT Steganography without encoding for string
and image as secret messages."""
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.metrics import structural_similarity as ssim
from skimage.transform import rescale
# from functions import bintostring, stringtobin, embedding, dembedding
from functions_refact import bintostring, stringtobin, embedding, dembedding

print('\nEXAMPLES OF DCT STEGANOGRAPHY WITHOUT ENCODING\n')
print('1. CHARACTERS STRING')
im = data.camera()  # Cover image
MESSAGE = 'Steganography among other rare disciplines is honored to be ' \
          'described as both an art and Science field.'     # Secret message
print('Num of characters in our message:', len(MESSAGE))
binarymessage = stringtobin(MESSAGE)

SQR_SIDE = 8    # Size of the blocks to be processed: Common info
BINMESLEN = len(binarymessage)    # Common info
steg = embedding(im, binarymessage, SQR_SIDE)    # Sender: stego image
# Receiver: secret message and recons image
secmes, recons = dembedding(steg, BINMESLEN, SQR_SIDE)

UNVEILED = bintostring(secmes)
print(f'The unveiled secret message is: {UNVEILED} \n')
# Plot
plt.figure(1)
plt.subplot(221)
plt.imshow(im, cmap='gray')
plt.title('Cover Im')
plt.subplot(222)
plt.imshow(steg, cmap='gray')
plt.title('Stego Im')
plt.subplot(223)
plt.imshow(recons, cmap='gray')
plt.title('Recons Im')
plt.show()


print('2. SECRET IMAGE')
im = data.astronaut()   # Cover image
im = 0.299 * im[:, :, 0] + 0.587 * \
    im[:, :, 1] + 0.114 * im[:, :, 2]  # Luminance
secim = rescale(data.camera(), 0.3, anti_aliasing=False)    # Secret image
secimflat = np.uint8(secim*255).flatten()   # Pre process to binarize
print('Num of pixels (bytes) in our message:', len(secimflat))
bitsec = np.unpackbits(secimflat)

SQR_SIDE = 8    # Size of the blocks to be processed: Common info
BINMESLEN = len(bitsec)    # Common info
steg = embedding(im, bitsec, SQR_SIDE)    # Sender: stego image
# Receiver: secret message and recons image
secmes, recons = dembedding(steg, BINMESLEN, SQR_SIDE)

revealsecflat = np.packbits(secmes)
revealsec = revealsecflat.reshape(np.shape(secim))
sec_ssim = ssim(np.double(secim*255), np.double(revealsec), data_range=255.0)

print(f'The SSIM comparing Original Sec Im and Revealed Sec Im is: '
      f'{sec_ssim} \n')
# Plot
plt.figure(2)
plt.subplot(231)
plt.imshow(im, cmap='gray')
plt.title('Cover (Lum) Im')
plt.subplot(232)
plt.imshow(secim, cmap='gray')
plt.title('Sec Im')
plt.subplot(233)
plt.imshow(steg, cmap='gray')
plt.title('Stego Im')
plt.subplot(234)
plt.imshow(recons, cmap='gray')
plt.title('Reconstructed Im')
plt.subplot(235)
plt.imshow(revealsec, cmap='gray')
plt.title('Reveal Im')
plt.show()

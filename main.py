# -*- coding: utf-8 -*-
"""
@author: Eric Vidal Marcos
@supervisor: Dr. Aurelien Coillet
@python version: 3.10.10 64-bits
@title: DCT Steganography
@description: This code is a Steganography method based on DCT transform. It
allows to hide a secret bitstring (e.g. message or image) into a cover image.
This is then encoded with Huffman encoding to reduce the size of the message.
Then, the secret is embedded using a hash function in the DCT coefficients of
the cover image. So we have a stego image. The receiver will use the same hash
function to extract the secret message from the stego image. Finally, the
secret message is decoded with Huffman decoding to recover the original secret.

In this code, we have 3 examples:
    1. Characters string
    2. Secret image
    3. Secret image with RGB cover image
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.metrics import structural_similarity as ssim
from skimage.transform import rescale
from functions import embedding, dembedding
# from functions_refact import embedding, dembedding
from huffman_tree import huffman_encoding, huffman_decoding


print('\nEXAMPLES OF DCT STEGANOGRAPHY WITH HUFFMAN ENCODING\n')
print('1. CHARACTERS STRING')
im = data.camera()  # Cover image
MESSAGE = 'Steganography among other rare disciplines is honored to be ' \
          'described as both an art and Science field.'
print('Secret message:', MESSAGE)
print('Num of characters in our message:', len(MESSAGE))
encod_text, huffmanhead = huffman_encoding(MESSAGE)
binarymessage = np.array([int(i) for i in encod_text], dtype='uint8')

SQR_SIDE = 8  # Size of the blocks to be processed: Common info
BINMESLEN = len(binarymessage)  # Common info

steg = embedding(im, binarymessage, SQR_SIDE)   # Sender: stego image
# Receiver: secret message and recons image
SECMES, recons = dembedding(steg, BINMESLEN, SQR_SIDE)

# create a string from secmes 1D-array to decode thanks to huffman tree
SECMES = ''.join(str(i) for i in SECMES)
decod_text = huffman_decoding(SECMES, huffmanhead)

UNVEILED = ''.join(decod_text)
print(f'The unveiled secret message is: {UNVEILED} \n')
# Plot
plt.figure(1)
plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.title('Cover Image')
plt.subplot(132)
plt.imshow(steg, cmap='gray')
plt.title('Stego Image')
plt.subplot(133)
plt.imshow(recons, cmap='gray')
plt.title('Reconstructed Image')
plt.tight_layout()
plt.show()

print('2. SECRET IMAGE')
im = data.astronaut()   # Cover image
im = 0.299 * im[:, :, 0] + 0.587 * \
    im[:, :, 1] + 0.114 * im[:, :, 2]  # Luminance
secim = rescale(data.camera(), 0.3, anti_aliasing=False)    # Secret image
# This is the maximum factor of scale that we can hide 0.34
# So the next step will be hiding a bit more in a RGB cover image

secimflat = list(np.uint64((secim*255).flatten()))
print('Num of pixels (bytes) in our message:', len(secimflat))
encod_text, huffmanhead = huffman_encoding(secimflat)
bitsec = np.array([int(i) for i in encod_text], dtype='uint8')

SQR_SIDE = 8    # Size of the blocks to be processed: Common info
BINMESLEN = len(bitsec)     # Common info

steg = embedding(im, bitsec, SQR_SIDE)

# Then we introduce it all to the dembedding PROCESS
SECMES, recons = dembedding(steg, BINMESLEN, SQR_SIDE)

# create a string from secmes 1D-array to decode thanks to huffman tree
SECMES = ''.join(str(i) for i in SECMES)

# ______________________________________________________________________________
# decod the huffman code of the image
textdecod = huffman_decoding(SECMES, huffmanhead)
# ______________________________________________________________________________
# from textdecod to a 1D-array
revealsecflat = np.array(textdecod)

# and reshape to obtain back again the secret image
revealsec = revealsecflat.reshape(np.shape(secim))

db_secim = np.double(secim*255)
db_revealsec = np.double(revealsec)
sec_ssim = ssim(db_secim, db_revealsec,
                data_range=255.0)

print(f'SSIM Original Sec Im and Revealed Sec Im is: {sec_ssim} \n')
# Plot
plt.figure(4)
plt.subplot(231)
plt.imshow(im, cmap='gray')
plt.title('Cover (Lum) Image')
plt.subplot(232)
plt.imshow(secim, cmap='gray')
plt.title('Secret Image')
plt.subplot(233)
plt.imshow(steg, cmap='gray')
plt.title('Stego Image')
plt.subplot(234)
plt.imshow(recons, cmap='gray')
plt.title('Reconstructed Image')
plt.subplot(235)
plt.imshow(revealsec, cmap='gray')
plt.title('Revealed Image')
plt.show()


print('3. SECRET IMAGE WITH RGB COVER')
im = data.astronaut()   # Cover image
secim = rescale(data.camera(), 0.5, anti_aliasing=False)    # Secret image
secimflat = list(np.uint64((secim*255).flatten()))
print('Num of pixels (bytes) in our message:', len(secimflat))
encod_text, huffmanhead = huffman_encoding(secimflat)
bitsec = np.array([int(i) for i in encod_text], dtype='uint8')

SQR_SIDE = 8    # Size of the blocks to be processed: Common info
BINMESLEN = len(bitsec)     # Common info

DIV = BINMESLEN // 3
bitsec_parts = [bitsec[:DIV], bitsec[DIV:(DIV*2)], bitsec[(DIV*2):]]

steg = np.zeros(np.shape(im))
recons = np.zeros(np.shape(im))
SECMES_FULL = []

_, _, layers = im.shape
for i in range(layers):
    print(f'Layer {i+1} of {layers}')
    im_part = im[:, :, i]
    bitsec_part = bitsec_parts[i]
    binmeslen_part = len(bitsec_part)

    # Then we introduce it all to the embedding PROCESS
    steg_part = embedding(im_part, bitsec_part, SQR_SIDE)

    # Then we introduce it all to the dembedding PROCESS
    SECMES_PART, recons_part = dembedding(steg_part, binmeslen_part, SQR_SIDE)
    # create a string from secmes 1D-array to decode thanks to huffman tree
    SECMES_JOINT = ''.join(str(i) for i in SECMES_PART)

    steg[:, :, i] = steg_part[:, :]
    recons[:, :, i] = recons_part[:, :]
    SECMES_FULL.append(SECMES_JOINT)

SECMES = ''.join(SECMES_FULL)  # Finally join the whole secret message str


textdecod = huffman_decoding(SECMES, huffmanhead)
revealsecflat = np.array(textdecod)  # put in a 1D-array
revealsec = revealsecflat.reshape(np.shape(secim))

# Due to there are negative values and values sligthly higher than 255,
# we rescale its values from 0 to 1 doubles values
steg = (steg-np.min(steg))/(np.max(steg)-np.min(steg))
recons = (recons-np.min(recons))/(np.max(recons)-np.min(recons))

db_normim = np.double(im)/np.max(im)
cover_ssim = ssim(db_normim, steg, data_range=255.0, multichannel=True,
                  channel_axis=2)
print('The SSIM comparing Cover Im and Stego Im is:', cover_ssim)

db_steg = np.double(steg)
db_recons = np.double(recons)
steg_ssim = ssim(db_steg, db_recons, data_range=255.0, multichannel=True,
                 channel_axis=2)
print('The SSIM comparing Stego Im and Reconstructed Im is:', steg_ssim)

db_secim = np.double(secim*255)
db_revealsec = np.double(revealsec)
sec_ssim = ssim(db_secim, db_revealsec,
                data_range=255.0)
print('The SSIM comparing Original Sec Im and Revealed Sec Im is:', sec_ssim)
# Plot
plt.figure(5)
plt.subplot(231)
plt.imshow(im)
plt.title('Cover (RGB) Image')
plt.subplot(232)
plt.imshow(secim, cmap='gray')
plt.title('Secret Image')
plt.subplot(233)
plt.imshow(steg)
plt.title('Stego Image')
plt.subplot(234)
plt.imshow(recons)
plt.title('Reconstructed Image')
plt.subplot(235)
plt.imshow(revealsec, cmap='gray')
plt.title('Revealed Image')
plt.show()

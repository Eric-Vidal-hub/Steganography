# -*- coding: utf-8 -*-
"""
@author: Eric Vidal Marcos
@supervisor: Dr. Aurelien Coillet
@python version: 3.10.10 64-bits
@title: DCT Steganography
@description: This code is a Steganography method based on DCT transform. It
allows to hide a secret message or image into a cover image. The secret message
or image is hidden by changing the least significant bits of the DCT
coefficients of the cover image. The secret message or image is recovered by
extracting the least significant bits of the DCT coefficients of the stego
image. The secret message or image is hidden by changing the least significant
bits of the DCT coefficients of the cover image. 
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.metrics import structural_similarity as ssim
from skimage.transform import rescale
from collections import Counter

from functions import bintostring, stringtobin, encoding, decoding
# from functions_refactored import bintostring, stringtobin, encoding, decoding

# %% 2. HUFFMAN TREE CODE
# Huffman static encoding: codifying and compressing binary data
class Node:
    '''__init__ parameters:
    prob: probability of the node
    symbol: character or number which characterizes the node.
    left: node at its left.
    right: node at its right.
    code: depending on the direction the tree goes, 0/1 is assigned to the
    node. Finally, we obtain a binary string that characterizes its symbol.
    '''
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob = prob
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ''


def calculate_frequency(data_in):
    '''
    Returns dictionary of characters (symbols) and its frequencies
    in the message or image.
    Input:
        data(list/str): flatten image (NO array) or message.
    Output:
        Counter(dict): keys -> symbols & values -> frequencies.
    '''
    return Counter(data_in)


# Definition outside the function to restart variable each time we call it
codes = {}


def calculate_codes(node, val=''):
    '''
    We will use a dictionary for the same reason as before, now each symbol
    will be associated with a 0/1 string that characterizes it, previously
    we have associated its frequency instead
    Input:
        node(class variable): node of the Huffman tree.
        val(str): binary code of the node.
            Default: ''. Empty str, because at the beggining the code values
            of each node are none
    Output:
        codes(dict): keys -> symbols & values -> binary code.
'''
    newval = val + str(node.code)
    # Explores all possible branches until reach all the tree leafs.
    if node.left:
        calculate_codes(node.left, newval)
    if node.right:
        calculate_codes(node.right, newval)
    # When it arrives to a leaf
    if (not node.left and not node.right):
        # Assign the leaf code to its corresponding symbol (dict key)
        codes[node.symbol] = newval
    return codes


def output_encoded(data_in, coding):
    '''
    Get the secret message or image written in huffman code
    and compare bits length with and without Huffman tree.
    Input:
        data_in(list/str): flatten image (NO array) or message.
        coding(dict): keys -> symbols & values -> binary code.
    Output:
        string(str): secret message or image written in huffman code.
        None. Print the bits used before and after compression.
    '''
    encoding_output = [coding[i] for i in data_in]
    encoded_string = ''.join([str(item) for item in encoding_output])

    # each text character or pixel image is 8 bits length
    before_compression = len(data_in)*8
    after_compression = 0
    symbols = coding.keys()
    for i in symbols:
        count = data_in.count(i)
        after_compression += count*len(coding[i])
    print('Bits used before compression:', before_compression)
    print('Bits used after compression:', after_compression)

    return encoded_string

# _______________________________________________________________________
#                               HUFFMAN TREE

# Function that compress the message using the  values of each character
# calculated by the huffman code


def Huffman_Encoding(data):
    '''Input:image or message (list or string). This function compresses the
    message by calculating the huffman tree and then encodes the message.
    Output: the encoded message in Huffman code 0/1 (str) and a class variable 
    node that contains all the huffman tree (node[0]) which is the treetop,
    respectively.'''

    # obtain the frequency of each character
    symbol_with_probs = calculate_frequency(data)

    # separate dictionary in characters (symbols) and frequencies
    symbols = symbol_with_probs.keys()
    # frequencies = symbol_with_probs.values()

    # empty list of nodes
    nodes = []

    # convert all the information of the symbol and frequency in nodes. Each
    # symbol will become a leaf of the tree.
    for symbol in symbols:
        nodes.append(Node(symbol_with_probs[symbol], symbol))

    # Create the huffman tree:
    while len(nodes) > 1:

        # to define the way of sorting we use the unname function lambda. It
        # extracts the frequency from the node and sort by least frequency to
        # most frequency one
        nodes = sorted(nodes, key=lambda x: x.prob)

        # we take the two least freq values of the list and assign them as
        # right node and left node.
        right = nodes[0]
        left = nodes[1]

        # Then we give them their associate code value (0 or 1)
        left.code = 0
        right.code = 1

        # we merge the two nodes to form a new one, which freq will be the sum of
        # the other two.
        # Here we are classifying the structure of the tree by introducing into
        # a new node which branch is located at its left and which at its right
        newNode = Node(left.prob+right.prob, left.symbol +
                       right.symbol, left, right)

        # now we remove from the node list the nodes that we already has used
        nodes.remove(left)
        nodes.remove(right)
        # and we introduce the new one
        nodes.append(newNode)

    # We start from the top and go down with the auxiliar function 3, so we get
    # the huffman code of each character
    huffman_coding = calculate_codes(nodes[0])

    # final string with encoded text
    coded_output = output_encoded(data, huffman_coding)

    return coded_output,    nodes[0]


# _______________________________________________________________________
# Decoding the Huffman tree:

# Huffman tree is the top node wich contains all the information of the others
# nodes and how they are related

def huffman_decoding(encod_data, huffmantree):
    '''Input:message codified (encod_data, str) and Huffman treetop (class node
    variable which contains all the information of the Huffman tree). This 
    function decode the message given the huffman tree used to encode the
    message. Output: decoded message, so corresponding symbols from the 
    original message (list).'''

    # value of the toptree
    tree_head = huffmantree

    # recovered message
    decoded_output = []

    # check the corresponding number of the message, if it is 1 goes left,
    # if it is 0 goes right
    for i in encod_data:
        if i == '1':
            # the new node is the right one
            huffmantree = huffmantree.right
        elif i == '0':
            # the new node is the left one
            huffmantree = huffmantree.left

        # prove each time, if we are not in a leaf pass if we are in a leaf
        # an error is expected and active the except AttributeError
        try:
            if (huffmantree.left.symbol == None) and (huffmantree.right.symbol == None):
                pass
        # when arrive at a leaf appends the character of the leaf into the list
        except AttributeError:
            decoded_output.append(huffmantree.symbol)
            huffmantree = tree_head

    return decoded_output

# IMPORTANT: the decoded output correspond to a list, this make possible to
# recover the message. BUT, you must take into account that if the original
# message is:
# String: convert the list to string
# Image: convert the list to array and reshape


# %% 3. EXAMPLES WITHOUT HUFFMAN TREE
'''We'll show the implementation of DCT Steganography in two examples:
character string and secret image.'''


# %%% 3.1. CHARACTERS STRING
print('EXAMPLE 3.1. CHARACTERS STRING')

im = data.camera()  # Cover image

# Secret message, this is the typical one:
message = 'Steganography among other rare disciplines is honored to be described\
 as both an art and Science field.'

# Firstly we wanna know how many characters has our message
print('Num of characters in our message:', len(message))

# We apply the proper function to get its corresponding binary values
binarymessage = stringtobin(message)

squarelength = 8  # square length to be used

# Then we introduce it all to the ENCODING PROCESS
steg = encoding(im, binarymessage, squarelength)

# Then we obtain the length in bits of the secret message
binarymessagelength = len(binarymessage)

# Then we introduce it all to the decoding PROCESS
secmes, recons = decoding(steg, binarymessagelength, squarelength)

# We use the 1D-array secmes, to finally unveil the secret message by turning
# from bin to string
unveiled = bintostring(secmes)

# Finally we print the unveiled message and prove that is the same that we
# introduced early in this example
print('The unveiled secret message is:', unveiled)
print('')  # This is to split examples
print('')

# Plot the most important images
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

# %%% 3.2. SECRET IMAGE
print('EXAMPLE 3.2. SECRET IMAGE')


im = data.astronaut()

# Cover image
im = 0.299 * im[:, :, 0] + 0.587 * \
    im[:, :, 1] + 0.114 * im[:, :, 2]  # Luminance

# Secret image
secim = rescale(data.camera(), 0.3, anti_aliasing=False)

# Previous process to further unpackbits
secimflat = np.uint8(secim*255).flatten()
print('Num of bytes in our message:', len(secimflat))

# We have to obtain our secret image in 1D-array of bits, so we unpackbits
bitsec = np.unpackbits(secimflat)

squarelength = 8  # square length to be used

# Then we introduce it all to the ENCODING PROCESS
steg = encoding(im, bitsec, squarelength)

# Then we obtain the length in bits of the secret message
binarymessagelength = len(bitsec)

# Then we introduce it all to the decoding PROCESS
secmes, recons = decoding(steg, binarymessagelength, squarelength)

# Once we obtain the 1D binary array we have to pack again to obtain integers
# (pixel values)
revealsecflat = np.packbits(secmes)
# and reshape to obtain back again the secret image
revealsec = revealsecflat.reshape(np.shape(secim))

db_secim = np.double(secim*255)
db_revealsec = np.double(revealsec)
sec_ssim = ssim(db_secim, db_revealsec,
                data_range=255.0)

print('The SSIM comparing Original Sec Im and Revealed Sec Im is:', sec_ssim)
print('')
print('')

# Plot the most important images
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

# %% 4. EXAMPLES WITH HUFFMAN TREE

'''We'll show the implementation of DCT Steganography in two examples:
character string and secret image which they are compressed by the Huffman
tree process.'''


# %%% 4.1. CHARACTERS STRING
print('EXAMPLE 4.1. CHARACTERS STRING WITH HUFFMAN TREE')

im = data.camera()  # Cover image

# Secret message, this is the typical one:
message = 'Steganography among other rare disciplines is honored to be described\
 as both an art and Science field.'

# Firstly we wanna know how many characters has our message
print('Num of characters in our message:', len(message))

# ______________________________________________________________________________
# now we use the huffman encoding
textcod, huffmantree = Huffman_Encoding(message)
# ______________________________________________________________________________
# Secret image in 1D-array of bits, from str textcod
binarymessage = np.zeros(len(textcod), dtype='uint8')

# Then we obtain the length in bits of the secret message
binarymessagelength = len(binarymessage)
# and put the textcod string in the array as uint8
for i in np.arange(binarymessagelength):
    binarymessage[i] = np.uint8(textcod[i])

squarelength = 8  # square length to be used

# Then we introduce it all to the ENCODING PROCESS
steg = encoding(im, binarymessage, squarelength)

# Then we obtain the length in bits of the secret message
binarymessagelength = len(binarymessage)

# Then we introduce it all to the decoding PROCESS
secmes, recons = decoding(steg, binarymessagelength, squarelength)

# create a string from secmes 1D-array to decode thanks to huffman tree
secmes1 = ''.join(str(i) for i in secmes)

# ______________________________________________________________________________
# decod the huffman code of the image
textdecod1 = huffman_decoding(secmes1, huffmantree)
# ______________________________________________________________________________

# We use the list secmes, to finally unveil the secret message by turning
# from list to string
unveiled1 = ''.join(i for i in textdecod1)

# Finally we print the unveiled message and prove that is the same that we
# introduced early in this example
print('The unveiled secret message is:', unveiled1)
print('')
print('')

# Plot the most important images
plt.figure(3)
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

# %%% 4.2. SECRET IMAGE
print('EXAMPLE 4.2. SECRET IMAGE WITH HUFFMAN TREE')

im = data.astronaut()

# Cover image
im = 0.299 * im[:, :, 0] + 0.587 * \
    im[:, :, 1] + 0.114 * im[:, :, 2]  # Luminance

# Secret image
secim = rescale(data.camera(), 0.3, anti_aliasing=False)
# This is the maximum factor of scale that we can hide 0.34
# So the next step will be hiding the whole image in a RGB cover image

# ______________________________________________________________________________
# flatten the image  to pas to a list
secimflat = np.uint64((secim*255).flatten())
secimflat = list(secimflat)
print('Num of bytes in our message:', len(secimflat))

# now we use the huffman encoding
textcod, huffmantree = Huffman_Encoding(secimflat)
# ______________________________________________________________________________
# Secret image in 1D-array of bits, from str textcod
bitsec = np.zeros(len(textcod), dtype='uint8')

# Then we obtain the length in bits of the secret message
binarymessagelength = len(bitsec)
# and put the textcod str in the array as uint8
for i in np.arange(binarymessagelength):
    bitsec[i] = np.uint8(textcod[i])


squarelength = 8  # square length to be used

# Then we introduce it all to the ENCODING PROCESS
steg = encoding(im, bitsec, squarelength)

# Then we introduce it all to the decoding PROCESS
secmes, recons = decoding(steg, binarymessagelength, squarelength)

# create a string from secmes 1D-array to decode thanks to huffman tree
secmes1 = ''.join(str(i) for i in secmes)

# ______________________________________________________________________________
# decod the huffman code of the image
textdecod = huffman_decoding(secmes1, huffmantree)
# ______________________________________________________________________________
# from textdecod to a 1D-array
revealsecflat = np.array(textdecod)

# and reshape to obtain back again the secret image
revealsec = revealsecflat.reshape(np.shape(secim))

db_secim = np.double(secim*255)
db_revealsec = np.double(revealsec)
sec_ssim = ssim(db_secim, db_revealsec,
                data_range=255.0)

print('The SSIM comparing Original Sec Im and Revealed Sec Im is:', sec_ssim)
print('')
print('')

# Plot the most important images
plt.figure(4)
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

# %%% 4.3. SECRET IMAGE WITH RGB COVER
print('EXAMPLE 4.3. SECRET IMAGE WITH HUFFMAN TREE and RGB COVER')

im = data.astronaut()
imR = im[:, :, 0]
imG = im[:, :, 1]
imB = im[:, :, 2]

# Secret image
secim = rescale(data.camera(), 0.5, anti_aliasing=False)
# This is the maximum factor of scale that we can hide 0.34
# So the next step will be hiding the whole image in a RGB cover image

# ______________________________________________________________________________
# flatten the image  to pass to a list
secimflat = np.uint64((secim*255).flatten())
secimflat = list(secimflat)
print('Num of bytes in our message:', len(secimflat))

# now we use the huffman encoding
textcod, huffmantree = Huffman_Encoding(secimflat)
# ______________________________________________________________________________
# Secret image in 1D-array of bits, from str textcod
bitsec = np.zeros(len(textcod), dtype='uint8')

# Then we obtain the length in bits of the secret message
binarymessagelength = len(bitsec)
# and put the textcod str in the array as uint8
for i in np.arange(binarymessagelength):
    bitsec[i] = np.uint8(textcod[i])

div = int(binarymessagelength/3)
bitsecR = bitsec[:div]
bitsecG = bitsec[div:(div*2)]
bitsecB = bitsec[(div*2):]

binmeslenR = len(bitsecR)
binmeslenG = len(bitsecG)
binmeslenB = len(bitsecB)

squarelength = 8  # square length to be used


# R
# Then we introduce it all to the ENCODING PROCESS
stegR = encoding(imR, bitsecR, squarelength)

# Then we introduce it all to the decoding PROCESS
secmes, reconsR = decoding(stegR, binmeslenR, squarelength)
# create a string from secmes 1D-array to decode thanks to huffman tree
secmesR = ''.join(str(i) for i in secmes)

# G
# Then we introduce it all to the ENCODING PROCESS
stegG = encoding(imG, bitsecG, squarelength)

# Then we introduce it all to the decoding PROCESS
secmes, reconsG = decoding(stegG, binmeslenG, squarelength)
# create a string from secmes 1D-array to decode thanks to huffman tree
secmesG = ''.join(str(i) for i in secmes)

# B
# Then we introduce it all to the ENCODING PROCESS
stegB = encoding(imB, bitsecB, squarelength)

# Then we introduce it all to the decoding PROCESS
secmes, reconsB = decoding(stegB, binmeslenB, squarelength)
# create a string from secmes 1D-array to decode thanks to huffman tree
secmesB = ''.join(str(i) for i in secmes)

secmes = secmesR+secmesG+secmesB  # Finally join the whole secret message str

# ______________________________________________________________________________
# decod the huffman code of the image
textdecod = huffman_decoding(secmes, huffmantree)
# ______________________________________________________________________________
revealsecflat = np.array(textdecod)  # put in a 1D-array

# Ensamble steg and recons layers to RGB images
steg = np.zeros(np.shape(im))
steg[:, :, 0] = stegR[:, :]
steg[:, :, 1] = stegG[:, :]
steg[:, :, 2] = stegB[:, :]

recons = np.zeros(np.shape(im))
recons[:, :, 0] = reconsR[:, :]
recons[:, :, 1] = reconsG[:, :]
recons[:, :, 2] = reconsB[:, :]

# Due to there are negative values and values sligthly higher than 255,
# we rescale its values from 0 to 1 doubles values
steg = (steg-np.min(steg))/(np.max(steg)-np.min(steg))
recons = (recons-np.min(recons))/(np.max(recons)-np.min(recons))

# and reshape to obtain back again the secret image
revealsec = revealsecflat.reshape(np.shape(secim))

db_normim = np.double(im)/np.max(im)
cover_ssim = ssim(db_normim, steg, data_range= 255.0, multichannel=True, channel_axis=2)
print('The SSIM comparing Cover Im and Stego Im is:', cover_ssim)

db_steg = np.double(steg)
db_recons = np.double(recons)
steg_ssim = ssim(db_steg, db_recons, data_range=255.0 , multichannel=True, channel_axis=2)
print('The SSIM comparing Stego Im and Reconstructed Im is:', steg_ssim)

db_secim = np.double(secim*255)
db_revealsec = np.double(revealsec)
sec_ssim = ssim(db_secim, db_revealsec,
                data_range=255.0)
print('The SSIM comparing Original Sec Im and Revealed Sec Im is:', sec_ssim)


# Plot the most important images
plt.figure(5)
plt.subplot(231)
plt.imshow(im)
plt.title('Cover (RGB) Im')
plt.subplot(232)
plt.imshow(secim, cmap='gray')
plt.title('Sec Im')
plt.subplot(233)
plt.imshow(steg)
plt.title('Stego Im')
plt.subplot(234)
plt.imshow(recons)
plt.title('Reconstructed Im')
plt.subplot(235)
plt.imshow(revealsec, cmap='gray')
plt.title('Reveal Im')
plt.show()

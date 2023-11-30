# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 11:34:09 2023
@author: Eric Vidal Marcos
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct
from skimage import data
from skimage.metrics import structural_similarity as ssim
from skimage.transform import rescale

from functions import bintostring, stringtobin, encoding


# %%% 1.3. MESSAGE DECODING PROCESS
'''We will implement a general method for every image to decode a message.
Mainly we have to take into account that sqlen used must be the same than
the one used to encode the message, and that we have to previously know
the secret message lenght, otherwise we could get in big trouble. It could
be encoded in the image, or sent via an encryted channel.'''


def decoding(im, binarymeslen, sqlen):
    '''In this function takes place the decoding process via dct. Inputs:
    stego image (2D-array) that must fit with the sqlen (integer) decided, and
    binarymes (1D- binary array) which is the one containg the secret message.
    Our output is the reconstructed image (2D-array) that appears to be exactly
    how the cover image looks like. It makes sense because the reconstructed
    image should be completely equal to the stego image. Output: the bin sec 
    mes (1D bin array), in order to aferwards do the convenient treatment to 
    get the secret image or string of characters.'''

    # Firstly we have to ensure that the sqlen fits the part of our cover image
    # that is going to encode the secret message
    # Actually it should be this way if every step is done correctly as we have
    # already checked this in the encoding function

    Nrow, Ncol = np.shape(im)  # Num of rows and columns of the im

    if Nrow % sqlen != 0:
        exit("Square length doesn't fit Nrow")
    if Ncol % sqlen != 0:
        exit("Square length doesn't fit Ncol")

    # We start the encoding process by defining the previous necessary variables

    # Temporary cover image square where we will apply the dct
    square = np.zeros((sqlen, sqlen))

    # Temporary recons image, where we'll have the dct coefficients with the
    # secret message embedded and recovered
    # but we will applyt the idct to obtain the final recons im
    decod = np.zeros((Nrow, Ncol))

    numsecbit = 0  # this is the position number of the bit message to be located
    # This acts as a counter

    # Array where we will put the embedded bits in order to recover the sec mes
    secmes = np.zeros(binarymeslen, dtype='uint8')

    for ii in np.arange(Nrow//sqlen):
        for jj in np.arange(Ncol//sqlen):
            # Take the squares from the cover image
            # We take every sqlenxsqlen square of the image
            square = im[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen]
            # Apply the normalized DCT to them
            square = dct(dct(square, axis=0, norm='ortho'),
                         axis=1, norm='ortho')  # DCT

            # The following condition is to stop embedding bits once all the
            # bits of the secret message are already embedded and we only
            # locate the corresponding dct square to the dctsteg
            if numsecbit == binarymeslen:
                decod[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)
                      * sqlen] = square[:, :]
                continue

            # finding secret process
            # We take the elements of the square
            for kk in np.arange(sqlen):
                for ll in np.arange(sqlen):

                    # Once we've extracted all the bits in the secret message
                    # we stop the process as we did before
                    if numsecbit == binarymeslen:
                        break

                    # The process consists in taking all the elements outside
                    # the square diagonal so we can recover a good idct.
                    if ll != kk:
                        # Take the corresponding element from the square
                        # as a temporary value
                        temp = square[kk, ll]

                        # Absolute and rounded value to be able to binarize
                        temp = bin(round(np.abs(temp)))

                        # We've obtained a string and convert to a list
                        temp2 = list(temp)

                        # where first 2 values are to identify a binary number
                        # by python: 0b, we omit them
                        temp2 = temp2[2:]

                        # Calculate the hash function to deceive the
                        # eavesdroppers: (Row+Column)%3= 0,1,2 position to embed
                        # our bit
                        # Which we treat it to be the bit position of the
                        # temporary value
                        # For binary numbers with length<4 we dont use the hash
                        # function to determine where we will place the bit,
                        # because it could mean a huge impact in its value,
                        # so we have decided to put this bit in the LSB
                        if len(temp2) < 4:
                            hashfun = -1
                        else:
                            hashfun = -((ii*sqlen+jj*sqlen+kk+ll) % 3+1)

                        # THIS IS THE PART THAT DIFFERS FROM ENCODING PROCESS
                        # Then we take the corresponding embedded bit from the
                        # 1D-array, secret message, according to numsecbit,
                        # and save to secmes array as an integer
                        secmes[numsecbit] = int(temp2[hashfun])

                        # We sum 1 to numsecbit to shift to the following bit
                        # until we run out of bits (we decode the whole sec mes)
                        numsecbit += 1

            # Once we have the whole square with the embedded bits we introduce
            # this in decod
            decod[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen] = square[:, :]

    # Now we have to apply the idct to every square of the decod to obtain
    # the reconstructed im which must be equal to the stego im
    recons = np.zeros((Nrow, Ncol))
    for ii in np.arange(Nrow//sqlen):
        for jj in np.arange(Ncol//sqlen):
            # Take the squares from the cover image
            # We take every sqlenxsqlen square of the image
            square = decod[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen]

            # Apply the normalized IDCT to them
            square = idct(idct(square, axis=0, norm='ortho'),
                          axis=1, norm='ortho')  # IDCT

            # Constuct the stego im by filling the squares
            recons[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen] = square[:, :]

    # Finally we print the number of bits decoded to ensure that is equal to the
    # secret message length
    print('Num of bits decoded:', numsecbit)
    # and the structural similar image to show how equal, or im and stego im are
    db_im = np.double(im)
    db_recons = np.double(recons)
    dec_ssim = ssim(db_im, db_recons,
                    data_range=db_recons.max() - db_recons.min())
    print('The SSIM comparing Cover Im and Stego Im is:', dec_ssim)
    return secmes, recons


# %% 2. HUFFMAN TREE CODE

# Code to programme the Huffman encoding, which enables us to codify and compress
# information in binary

# Auxiliar functions which we'll use in static Huffman code:
# _______________________________________________________________________

# 1 First we have to create a Node which is a class. It will let us to develop
# a list of nodes but keeping intact all other parameters

# Arguments: left=None and right=None means that if we don't especify the values
# their default values are None

# Self:We use that command when programe a class variable to refer to the
# class object itself. It is a way that the program understand that the same
# node is used and called in the functions


class Node:
    '''__init__: whenever we call a Node class, the node will apply this
    function, and we can access to its variables by writing Node.variable'''

    def __init__(self, prob, symbol, left=None, right=None):
        # It relates node with character frequencies
        self.prob = prob

        # character or number which characterizes the node
        self.symbol = symbol

        # Saves which node has at its left
        self.left = left

        # Saves which node has at its right
        self.right = right

        # Depending on the direction where we go, we'll introduce a value 0/1
        # to variable code, at first we give an empty value
        # Finally we'll obtain the 0/1 chain that characterizes its symbol
        self.code = ''


# _______________________________________________________________________
# 2 Function which will calculate frequencies of each character (symbol)
# in the message

# We want to save each character frecuency from our message, so
# the most natural way to do it is using dictionaries
def calculate_frequency(data):
    '''Input: Flatten image as a list (no array) or message (str). This 
    function return the frequency of appearence of each symbol in the data. 
    Output: dictionary with symbols as a keys and freq as a values'''

    symbols = {}  # dictonary to be returned with values and its frequencies
    for element in data:
        # if the element is not in the dictionary yet, it is introduced: freq 1
        if symbols.get(element) is None:
            symbols[element] = 1
        else:
            symbols[element] += 1  # if it already is sum 1
    return symbols


# _______________________________________________________________________
# 3 Function that calculate Huffman code of each character

# We will use a dictionary for the same reason as before, now each symbol
# will be associated with a 0/1 string that characterizes it, previously
# we have associated its frequency instead
codes = dict()
# Every time we run the program, codes dictionary must restart, so that's why we
# define codes outside the function

# Reminding: we put val=empty by default, because at the beggining the
# code values of each node are none


def calculate_codes(node, val=''):
    '''Input: class variable node (node) and value given to that note (val)
    which by default is empty. This function calculate the new binary
    code given to each symbol of the text. Output: dictionary (codes) with 
    symbols as keys and the binary code as values'''

    # Value of the Huffman code in the current node. We call the node class
    # and take its code value and then sum it to the previous ones.
    newval = val+str(node.code)

    # Iterative function. It explores all possible branches until reach all the
    # tree leafs.
    if (node.left):
        calculate_codes(node.left, newval)
    if (node.right):
        calculate_codes(node.right, newval)

    # Finally arrive to a leaf
    if (not node.left and not node.right):
        # put the code of the leaf to its corresponding symbol (key of the dict)
        codes[node.symbol] = newval
    return codes


# _______________________________________________________________________
# 4 Get the secret message or image write with huffman code

def output_encoded(data, coding):
    '''Input: secret message or image (str/list) and the binary values of each
    symbol from coding (dict). This function gets the secret message encoded
    with the huffman code. Output: Huffman encoded message (string)'''

    encoding_output = []
    # these will be the keys of coding (which are the characters)
    for i in data:
        # then we gather the message in Huffman code
        encoding_output.append(coding[i])

    # take each element of the list and joining in an empty string
    string = ''.join([str(item) for item in encoding_output])
    return string


# _______________________________________________________________________
# 5 Function to compare how the secret text or image  has been reduced

def total_gain(data, coding):
    '''Input:secret message or image (str/list) and the binary values of each
    symbol (coding, dict).This function calculate the length of the message
    without compressing and with Huffman tree method. Output: it doesnt return
    any variable, it directly prints the both lengths to compare'''
    # each character of the text or image are 8 bits, so the length is the
    # number of charachters multiplied by 8
    before_compression = len(data)*8
    after_compression = 0

    # count how many times appears the carachter in the message and multiply
    # by the length of huffman code
    symbols = coding.keys()
    for i in symbols:
        count = data.count(i)
        after_compression += count*len(coding[i])
    print('Bits used before compression:', before_compression)
    print('Bits used after compression:', after_compression)

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

    # calculate the total gain
    total_gain(data, huffman_coding)

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
                data_range=db_revealsec.max() - db_revealsec.min())

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
                data_range=db_revealsec.max() - db_revealsec.min())

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
cover_ssim = ssim(db_normim, steg, data_range=steg.max() -
                  steg.min(), multichannel=True, channel_axis=2)
print('The SSIM comparing Cover Im and Stego Im is:', cover_ssim)

db_steg = np.double(steg)
db_recons = np.double(recons)
steg_ssim = ssim(db_steg, db_recons, data_range=db_recons.max() -
                 db_recons.min(), multichannel=True, channel_axis=2)
print('The SSIM comparing Stego Im and Reconstructed Im is:', steg_ssim)

db_secim = np.double(secim*255)
db_revealsec = np.double(revealsec)
sec_ssim = ssim(db_secim, db_revealsec,
                data_range=db_revealsec.max() - db_revealsec.min())
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

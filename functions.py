import numpy as np


def stringtobin(mes):
    '''Input: message (str). This is changed to binary by using ord() to get
    its ASCII values (int), then we unpackbits and get an array containg the
    message in bits. Output: binary message (1D 0/1 uint8 array) .'''

    binmes = np.zeros(len(mes), dtype='uint8')  # Array of mes length

    # Then we introduce every ASCII value of the mes to binmes
    for i in range(len(mes)):
        binmes[i] = ord(mes[i])

    # Finally we unpack the integer ASCII values
    binmes = np.unpackbits(binmes)
    # to its corresponding bits
    return binmes


def bintostring(binmes):
    '''This is the inverse function to stringtobin. Input: message in bin num
    (array). Then we packbits, get integer which correspond to ASCII values.
    Finally, transform this integers to characters by means of chr() and join
    them, which means that we recovered the original message. Output: original
    message (str).'''

    # bin to integer which must be the ASCII values
    integs = np.packbits(binmes)

    # Finally we return the original message by turning integers to characters
    # with chr() and joining this characters
    return ''.join(chr(integs[i]) for i in range(len(integs)))

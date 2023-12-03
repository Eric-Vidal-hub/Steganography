from collections import Counter
import numpy as np
from skimage import data


class Node:
    '''
    Represents a node in a Huffman tree.

    Args:
        prob: The probability of the node.
        symbol: The character or number that characterizes the node.
        left: The node at its left. Default: None.
        right: The node at its right. Default: None.
        code: The binary code assigned to the node. Default: ''.
    '''
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob = prob
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ''


# Definition outside the function to restart variable each time we call it
codes = {}


def calculate_codes(node, val=''):
    '''
    Calculate the Huffman code for each symbol in the Huffman tree.

    Args:
        node(class variable): node of the Huffman tree.
        val(str): binary code of the node.
            Default: ''. Empty str, because at the beggining the code values
            of each node are none.
    
    Returns:
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

    Args:
        data_in(list/str): flatten image (NO array) or message.
        coding(dict): keys -> symbols & values -> binary code.

    Retruns(tuple):
        string(str): secret message or image written in huffman code.
        None. Print the bits used before and after compression.
    '''
    # Directly join the strings for encoding_output
    encoded_string = ''.join(coding[i] for i in data_in)
    # each text character or pixel image is 8 bits length
    before_compression = len(data_in) * 8
    after_compression = sum(data_in.count(i)
                            * len(code) for i, code in coding.items())
    print('Bits used before compression:', before_compression)
    print('Bits used after compression:', after_compression)
    return encoded_string


# HUFFMAN TREE
def Huffman_Encoding(data_in):
    '''
    Message compression using the values of each character calculated by the
    Huffman code.

    Args:
        data_in(list/str): flatten image (NO array) or message.

    Retruns(tuple):
        coded_output(str): secret message or image written in huffman code.
        nodes[0](class variable): Huffman tree top.
    '''
    symbol_with_probs = Counter(data_in)    # Character's frequency (dict)
    symbols = symbol_with_probs.keys()
    frequencies = symbol_with_probs.values()

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
    coded_output = output_encoded(data_in, huffman_coding)

    return coded_output,    nodes[0]


# _______________________________________________________________________
# Decoding the Huffman tree:

# Huffman tree is the top node wich contains all the information of the others
# nodes and how they are related

def huffman_decoding(encod_data, huffmantree):
    """
    Decode the message given the Huffman tree used to encode the message.

    Args:
        encod_data (str): The encoded message to be decoded.
        huffmantree (Node): The top node of the Huffman tree used for encoding.

    Returns:
        list: The decoded message as a list of symbols.
    """

    tree_head = huffmantree
    decoded_output = []

    for i in encod_data:
        if i == '1':
            huffmantree = huffmantree.right
        elif i == '0':
            huffmantree = huffmantree.left

        if huffmantree.symbol is not None:  # Directly check if it's a leaf node
            decoded_output.append(huffmantree.symbol)
            huffmantree = tree_head

    return decoded_output

# IMPORTANT: the decoded output correspond to a list, this make possible to
# recover the message. BUT, you must take into account that if the original
# message is:
# String: convert the list to string
# Image: convert the list to array and reshape


# EXAMPLE
# create the secret image
im = data.camera()

# flatten the image  to pas to a list
im = np.uint64(im.flatten())
im1 = list(im)

# now we use the huffman encoding
textcod2, huffmantree2 = Huffman_Encoding(im1)

# decod the huffman code of the image
textdecod2 = huffman_decoding(textcod2, huffmantree2)

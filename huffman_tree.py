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
    after_compression = 0
    symbols = coding.keys()
    for i in symbols:
        count = data_in.count(i)
        after_compression += count*len(coding[i])
    print('Bits used before compression:', before_compression)
    print('Bits used after compression:', after_compression)
    return encoded_string


def huffman_encoding(data_in):
    '''
    Message compression using the values of each character calculated by the
    Huffman code.

    Args:
        data_in(list/str): flatten image (NO array) or message.

    Retruns(tuple):
        coded_output(str): secret message or image written in huffman code.
        nodes[0](class variable): Huffman tree top. It contains all the
        information of the others nodes and how they are related.
    '''
    symbol_with_probs = Counter(data_in)    # Character's frequency (dict)
    symbols = symbol_with_probs.keys()
    nodes = [Node(symbol_with_probs[symbol], symbol) for symbol in symbols]
    # Huffman tree creation
    while len(nodes) > 1:
        # Sorting definition with the unnamed function lambda. It extracts the
        # node frequency and sort it from the least to the most frequent
        nodes = sorted(nodes, key=lambda x: x.prob)
        # Least frequent and assign them as right node and left node.
        right = nodes[0]
        left = nodes[1]
        # Then we associate them a code value (0 or 1)
        left.code = 0
        right.code = 1
        # Merge the two nodes with frequency summed up
        newnode = Node(left.prob+right.prob, left.symbol +
                       right.symbol, left, right)
        # Remove the nodes that we have already merged
        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newnode)   # and introduce the new one

    # Start from the top and go down toe get the code of each symbol
    huffman_coding = calculate_codes(nodes[0])
    # final string with encoded text
    coded_output = output_encoded(data_in, huffman_coding)
    return coded_output, nodes[0]


def huffman_decoding(encod_data, huffmantree):
    # sourcery skip: remove-empty-nested-block, remove-redundant-if
    """
    Decode the message given the Huffman tree used to encode the message.

    Args:
        encod_data (str): The encoded message to be decoded.
        huffmantree (Node): The top node of the Huffman tree used for encoding.

    Returns:
        list: The decoded message as a list of symbols.
    If the original message is:
    String: convert the list to string.
    Image: convert the list to array and reshape.
    """
    tree_head = huffmantree
    decoded_output = []
    for i in encod_data:
        if i == '0':
            huffmantree = huffmantree.left
        elif i == '1':
            huffmantree = huffmantree.right
        # prove each time, if we are not in a leaf pass if we are in a leaf
        # an error is expected and active the except AttributeError
        try:
            if (
                huffmantree.left.symbol is None
                and huffmantree.right.symbol is None
            ):
                pass
        except AttributeError:
            decoded_output.append(huffmantree.symbol)
            huffmantree = tree_head
    return decoded_output


# EXAMPLE
im = data.camera()  # Secret image
im = list(np.uint64(im.flatten()))  # Pre process to binarize
textehcod, hufftree = huffman_encoding(im)

# decod the huffman code of the image
textdecod = huffman_decoding(textehcod, hufftree)

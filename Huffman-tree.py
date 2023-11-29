# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 19:04:51 2023
@author: Eric Vidal Marcos
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data

#Code to programme the Huffman encoding, which enables us to codify and compress
#information in binary

#Auxiliar functions which we'll use in static Huffman code:
# _______________________________________________________________________

#1 First we have to create a Node which is a class. It will let us to develop
#a list of nodes but keeping intact all other parameters 

#Arguments: left=None and right=None means that if we don't especify the values
# their default values are None

#Self:We use that command when programe a class variable to refer to the 
# class object itself. It is a way that the program understand that the same
#node is used and called in the functions 


class Node:
    '''__init__: whenever we call a Node class, the node will apply this
    function, and we can access to its variables by writing Node.variable'''
    def __init__(self, prob, symbol, left=None,right=None):
        #It relates node with character frequencies
        self.prob=prob
        
        #character or number which characterizes the node
        self.symbol=symbol
        
        #Saves which node has at its left
        self.left=left
        
        #Saves which node has at its right
        self.right=right
        
        #Depending on the direction where we go, we'll introduce a value 0/1 
        #to variable code, at first we give an empty value
        #Finally we'll obtain the 0/1 chain that characterizes its symbol
        self.code=''


# _______________________________________________________________________
#2 Function which will calculate frequencies of each character (symbol)
#in the message

#We want to save each character frecuency from our message, so 
#the most natural way to do it is using dictionaries 
def calculate_frequency(data):
    '''Input: Flatten image as a list (no array) or message (str). This 
    function return the frequency of appearence of each symbol in the data. 
    Output: dictionary with symbols as a keys and freq as a values'''
    
    symbols=dict()  #dictonary to be returned with values and its frequencies
    for element in data:
        #if the element is not in the dictionary yet, it is introduced: freq 1
        if symbols.get(element)==None:
            symbols[element]=1
        else:
            symbols[element]+=1     #if it already is sum 1
    return symbols


# _______________________________________________________________________
#3 Function that calculate Huffman code of each character 

#We will use a dictionary for the same reason as before, now each symbol 
#will be associated with a 0/1 string that characterizes it, previously
#we have associated its frequency instead
codes=dict()
#Every time we run the program, codes dictionary must restart, so that's why we
#define codes outside the function

#Reminding: we put val=empty by default, because at the beggining the
#code values of each node are none


def calculate_codes(node,val=''):
    '''Input: class variable node (node) and value given to that note (val)
    which by default is empty. This function calculate the new binary
    code given to each symbol of the text. Output: dictionary (codes) with 
    symbols as keys and the binary code as values'''
    
    #Value of the Huffman code in the current node. We call the node class
    #and take its code value and then sum it to the previous ones.
    newval=val+str(node.code)
    
    #Iterative function. It explores all possible branches until reach all the
    #tree leafs.
    if (node.left):
        calculate_codes(node.left,newval)
    if (node.right):
        calculate_codes(node.right,newval)
    
    #Finally arrive to a leaf
    if (not node.left and not node.right):
        #put the code of the leaf to its corresponding symbol (key of the dict)
        codes[node.symbol]=newval
    return codes



# _______________________________________________________________________
#4 Get the secret message or image write with huffman code

def output_encoded(data,coding):
    '''Input: secret message or image (str/list) and the binary values of each
    symbol from coding (dict). This function gets the secret message encoded
    with the huffman code. Output: Huffman encoded message (string)'''
    
    encoding_output=[]
    for i in data: #these will be the keys of coding (which are the characters)
    #then we gather the message in Huffman code
        encoding_output.append(coding[i])   
   
    #take each element of the list and joining in an empty string
    string=''.join([str(item) for item in encoding_output])
    return string


# _______________________________________________________________________
#5 Function to compare how the secret text or image  has been reduced 

def total_gain(data,coding):
    '''Input:secret message or image (str/list) and the binary values of each
    symbol (coding, dict).This function calculate the length of the message
    without compressing and with Huffman tree method. Output: it doesnt return
    any variable, it directly prints the both lengths to compare'''
    #each character of the text or image are 8 bits, so the length is the
    #number of charachters multiplied by 8
    before_compression=len(data)*8
    after_compression=0
    
    #count how many times appears the carachter in the message and multiply
    #by the length of huffman code
    symbols=coding.keys()
    for i in symbols:
        count=data.count(i)
        after_compression+=count*len(coding[i])
    print('Bits used before compression:',before_compression)
    print('Bits used after compression:',after_compression)

# _______________________________________________________________________
#                               HUFFMAN TREE

#Function that compress the message using the  values of each character
#calculated by the huffman code

def Huffman_Encoding(data):
    '''Input:image or message (list or string). This function compresses the
    message by calculating the huffman tree and then encodes the message.
    Output: the encoded message in Huffman code 0/1 (str) and aclass variable 
    node that contains all the huffman tree (node[0]) which is the treetop,
    respectively.'''
    
    #obtain the frequency of each character
    symbol_with_probs= calculate_frequency(data)
    
    #separate dictionary in characters (symbols) and frequencies
    symbols=symbol_with_probs.keys()
    frequencies=symbol_with_probs.values()
    
    #empty list of nodes
    nodes=[]
    
    #convert all the information of the symbol and frequency in nodes. Each
    #symbol will become a leaf of the tree.
    for symbol in symbols:        
        nodes.append(Node(symbol_with_probs[symbol],symbol))
        
    #Create the huffman tree:
    while len(nodes)>1:

        #to define the way of sorting we use the unname function lambda. It 
        #extracts the frequency from the node and sort by least frequency to
        #most frequency one
        nodes=sorted(nodes,key=lambda x: x.prob)
        
        #we take the two least freq values of the list and assign them as 
        #right node and left node. 
        right=nodes[0]
        left=nodes[1]
        
        #Then we give them their associate code value (0 or 1)
        left.code=0
        right.code=1
        
        #we merge the two nodes to form a new one, which freq will be the sum of
        #the other two. 
        #Here we are classifying the structure of the tree by introducing into
        #a new node which branch is located at its left and which at its right
        newNode=Node(left.prob+right.prob,left.symbol+right.symbol,left,right)
        
        #now we remove from the node list the nodes that we already has used 
        nodes.remove(left)
        nodes.remove(right)
        #and we introduce the new one
        nodes.append(newNode)
        

    #We start from the top and go down with the auxiliar function 3, so we get 
    #the huffman code of each character
    huffman_coding=calculate_codes(nodes[0])
    
    #calculate the total gain
    total_gain(data, huffman_coding)
    
    #final string with encoded text
    coded_output=output_encoded(data,huffman_coding )
    
    return coded_output,    nodes[0]


# _______________________________________________________________________
#Decoding the Huffman tree:

#Huffman tree is the top node wich contains all the information of the others
#nodes and how they are related

def huffman_decoding(encod_data, huffmantree):
    '''Input:message codified (encod_data, str) and Huffman treetop (class node
    variable which contains all the information of the Huffman tree). This 
    function decode the message given the huffman tree used to encode the
    message. Output: decoded message, so corresponding symbols from the 
    original message (list).'''
    
    #value of the toptree
    tree_head= huffmantree
    
    #recovered message
    decoded_output=[]
    
    #check the corresponding number of the message, if it is 1 goes left,
    #if it is 0 goes right
    for i in encod_data:
        if i=='1':
            #the new node is the right one
            huffmantree=huffmantree.right
        elif i=='0':
            #the new node is the left one
            huffmantree=huffmantree.left
        
        #prove each time, if we are not in a leaf pass if we are in a leaf 
        #an error is expected and active the except AttributeError
        try:
            
            
            if (huffmantree.left.symbol==None 
            and huffmantree.right.symbol==None):
                pass
            
        #when arrive at a leaf appends the character of the leaf into the list
        except AttributeError:
            decoded_output.append(huffmantree.symbol)
            huffmantree=tree_head
            
    return decoded_output

#IMPORTANT: the decoded output correspond to a list, this make possible to 
#recover the message. BUT, you must take into account that if the original 
#message is:
#String: convert the list to string
#Image: convert the list to array and reshape


#EXAMPLE
#create the secret image
im=data.camera()

#flatten the image  to pas to a list
im=np.uint64(im.flatten())
im1=list(im)

#now we use the huffman encoding
textcod2,huffmantree2=Huffman_Encoding(im1)

#decod the huffman code of the image
textdecod2=huffman_decoding(textcod2,huffmantree2)
import numpy as np
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim


def stringtobin(mes):
    '''
    Converts a string to a binary array.
    Convert the string to its ASCII values (int),
    then we unpackbits and get an array containg the
    message in bits (0/1).
    Input(str): message
    Output(1DArray[uint8]): binary message
    '''
    length = len(mes)
    binmes = np.zeros(length, dtype='uint8')
    for i in range(length):
        binmes[i] = ord(mes[i])     # String to its ASCII values
    return np.unpackbits(binmes)    # ASCII values to bits


def bintostring(binmes):
    '''
    Inverse function of stringtobin.
    Packbits, thus get the ASCII values (int),
    then transform this into characters and join them.
    Input(1DArray[uint8]): binary message
    Output(str): message
    '''
    ints = np.packbits(binmes)    # bin to int which are the mes ASCII values
    return ''.join(chr(ints[i]) for i in range(len(ints)))  # ASCII to mes(str)


def encoding(im, binmes, sqlen):
    '''
    Message embedding process via DCT coefficients into 1 layer image.
    First, it ensures that cover image fits the decided square length
    and the message length. Then, secret bin message is introduced in the
    stego image using its DCT coefficients, so the original and stego images
    look the same.
    Inputs:
    im(2D-array): cover image
    binmes(1D-array): binary message
    sqlen(int): square length
    Output:
    stego(2D-array): stego image
    '''
    nrow, ncol = np.shape(im)   # Size of the cover image
    if nrow % sqlen != 0:
        exit("Square length doesn't fit nrow")
    if ncol % sqlen != 0:
        exit("Square length doesn't fit ncol")

    binmes_len = len(binmes)    # Length of the secret message

    # The diagonal of each square DCT coefficients contain vital information
    # about the original image, so we do not embed bits in these coefficients
    # in order to obtain a good approximation of the original image when
    # we perform the inverse dct (idct)

    # So, max number of bits that can be embedded in our cover im
    maxbits = np.size(im) * (1. - 1./sqlen)
    maxbytes = maxbits / 8.  # A byte can either be a pixel or a character
    if maxbits < binmes_len:
        exit('Number of bits to be embedded from your secret message exceeds \
              the capacity to embed bits that your cover image has.')
    else:
        print('Number of bits to be embedded:', binmes_len)
        print('Max num of bits that can be embedded:', maxbits)
        print('Max num of bytes that can be encoded in this cover image \
          and square length:', maxbytes)

    # Orientative value of max side length of largest square image embedable
    maxsqrimside = np.uint8(np.sqrt(maxbytes))
    # It could have been perfectly be a rectangle, it means a reshape process
    print('Max side length that could have an square image encoded in this \
          cover image and square length:', maxsqrimside)
    print('')  # To separate different information

    # ENCODING PROCESS
    # Initial variables
    square = np.zeros((sqlen, sqlen))   # Temporary square to apply the dct

    # Temporary stego image with dct coefficients and secret message embedded
    dctsteg = np.zeros((nrow, ncol))

    numsecbit = 0  # Position index of the message bit to be located

    # Take the squares from the cover image
    for ii in np.arange(nrow//sqlen):
        for jj in np.arange(ncol//sqlen):
            square = im[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen]
            # Apply the normalized DCT to them
            square = dct(dct(square, axis=0, norm='ortho'),
                         axis=1, norm='ortho')  # DCT

            # The following condition is to stop embedding bits once all the
            # bits of the secret message are already embedded and we only
            # locate the corresponding dct square to the dctsteg
            if (numsecbit == (len(binmes))):
                dctsteg[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)
                        * sqlen] = square[:, :]
                continue

            # hiding secret process
            # We take the elements of the square
            for kk in np.arange(sqlen):
                for ll in np.arange(sqlen):

                    # Once we embed all the bits in the secret message
                    # we stop the process as we did before
                    if (numsecbit == (len(binmes))):
                        break

                    # The process consists in taking all the elements outside
                    # the square diagonal so we can recover a good idct.
                    if (ll != kk):
                        # Take the corresponding element from the square
                        # as a temporary value
                        temp = square[kk, ll]

                        # First we save its sign in order to approximately
                        # recover our initial value
                        sign = 1 if (temp >= 0) else -1
                        # Absolute and rounded value to be able to binarize
                        temp = bin(round(np.abs(temp)))

                        # We've obtained a string and convert to a list
                        temp2 = list(temp)

                        # where first 2 values are to identify a binary number
                        # by python: 0b, we omit them
                        temp2 = temp2[2:]

                        # Calculate the hash function to deceive the eavesdroppers:
                        # (Row+Column)%3= 0,1,2 position to embed our bit
                        # Which we treat it to be the bit position of the temporary value
                        # For binary numbers with length<4 we dont use the hash function
                        # to determine where we will place the bit, because it could mean
                        # a huge impact in its value, so we have decided to put this bit
                        # in the LSB
                        if (len(temp2) < 4):
                            hashfun = -1
                        else:
                            hashfun = -((ii*sqlen+jj*sqlen+kk+ll) % 3+1)

                        # Then we embed the corresponding bit from the 1D-array,
                        # secret message, according to numsecbit, as a string
                        # into the list of temp
                        temp2[hashfun] = str(binmes[numsecbit])

                        # we recover a string from the list created as temp2
                        fbit = ''.join(temp2)

                        # and compute the final value
                        value = int(fbit, 2)

                        # Put it back again in the square with its corresponding sign
                        # and the secret bit embeded
                        square[kk, ll] = value*sign

                        # We sum 1 to numsecbit to shift to the following bit
                        # until we run out of bits (we embed the whole sec mes)
                        numsecbit += 1

            # Once we have the whole square with the embedded bits we introduce
            # this in the dctsteg
            dctsteg[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)
                    * sqlen] = square[:, :]

    # Now we have to apply the idct to every square of the dctsteg to obtain
    # the actual stego im
    stego = np.zeros((nrow, ncol))
    for ii in np.arange(nrow//sqlen):
        for jj in np.arange(ncol//sqlen):
            # Take the squares from the cover image
            # We take every sqlenxsqlen square of the image
            square = dctsteg[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen]

            # Apply the normalized IDCT to them
            square = idct(idct(square, axis=0, norm='ortho'),
                          axis=1, norm='ortho')  # IDCT

            # Constuct the stego im by filling the squares
            stego[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen] = square[:, :]

    # Finally we print the number of bits encoded to ensure that is equal to the
    # secret message length
    print('Num of bits encoded:', numsecbit)
    # and the structural similar image to show how equal, or im and stego im are
    db_im = np.double(im)
    db_steg = np.double(stego)
    enc_ssim = ssim(db_im, db_steg, data_range=db_steg.max() - db_steg.min())
    print('The SSIM comparing Cover Im and Stego Im is:', enc_ssim)
    return stego
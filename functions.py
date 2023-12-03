import numpy as np
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim


def stringtobin(mes):
    '''
    Converts a string to its ASCII values (int), then unpackbits to obtain an
    array containg the message in bits (0/1).

    Args:
        mes(str): message

    Retrurns:
        binmes(1DArray[uint8]): binary message
    '''
    length = len(mes)
    binmes = np.zeros(length, dtype='uint8')
    for i in range(length):
        binmes[i] = ord(mes[i])     # String to its ASCII values
    return np.unpackbits(binmes)    # ASCII values to bits


def bintostring(binmes):
    '''
    Check if the square length fits the image dimensions.

    Args:
        nrow (int): The number of rows in the image.
        ncol (int): The number of columns in the image.
        sqlen (int): The square length.

    Raises:
        ValueError: If the square length doesn't fit the image dimensions.

    Examples:
        # Check if the square length fits the image dimensions
        num_rows = ...
        num_cols = ...
        square_length = ...
        check_image_dimensions(num_rows, num_cols, square_length)
    '''
    ints = np.packbits(binmes)    # bin to int which are the mes ASCII values
    return ''.join(chr(ints[i]) for i in range(len(ints)))  # ASCII to mes(str)


def check_image_dimensions(nrow, ncol, sqlen):
    if nrow % sqlen != 0 or ncol % sqlen != 0:
        raise ValueError("Square length doesn't fit image dimensions")


def check_message_length(maxbits, binmes_len):
    """
    Check if the number of bits to be embedded exceeds the image capacity.

    Args:
        maxbits (int): The maximum number of bits that can be embedded.
        binmes_len (int): The length of the binary message.

    Raises:
        ValueError: If message exceeds the image capacity.

    Examples:
        # Check if the message length exceeds the image capacity
        maxbits = 1000
        binmes_len = 1200
        check_message_length(maxbits, binmes_len)
    """
    if maxbits < binmes_len:
        raise ValueError('Number of bits to be embedded exceeds capacity of\
                         image')


def calculate_hashfun(temp, ii, sqlen, jj, kk, ll):
    return -1 if len(temp) < 4 else -((ii*sqlen+jj*sqlen+kk+ll) % 3+1)


def encoding(im, binmes, sqlen):
    '''
    Perform message embedding process via DCT coefficients into 1-layer image.

    First, it ensures that the cover image fits the decided square length and
    message length. Then, the secret binary message is introduced in the stego
    image using its DCT coefficients, so the original and stego images look
    the same.

    Args:
        im (2D-array): The cover image.
        binmes (1D-array): The binary message.
        sqlen (int): The square length.

    Returns:
        2D-array: The stego image.
    '''
    nrow, ncol = np.shape(im)   # Size of the cover image
    check_image_dimensions(nrow, ncol, sqlen)
    binmes_len = len(binmes)    # Length of the secret message
    # The diagonal of each square DCT coefficients contain vital information
    # about the original image, so we do not embed bits in these coefficients
    # in order to obtain a good approximation of the original image when
    # we perform the inverse dct (idct)
    # So, max number of bits that can be embedded in our cover im
    maxbits = np.size(im) * (1. - 1./sqlen)
    maxbytes = maxbits / 8.  # A byte can either be a pixel or a character
    check_message_length(maxbits, binmes_len)
    print('Number of bits to be embedded:', binmes_len)
    print('Max num of bits that can be embedded:', maxbits)
    print('Max num of bytes that can be encoded in this cover image and square\
          length:', maxbytes)
    # Orientative value of max side length of largest square image embedable
    maxside = np.uint8(np.sqrt(maxbytes))
    # One can also embed a rectangle im, but it means a reshape process
    print(f'Max side length that could have an square image encoded in this\
          cover image and square length: {maxside}\n')
    # ENCODING PROCESS
    # Initial variables
    square = np.zeros((sqlen, sqlen))   # Temporary square to apply the dct
    stego = np.zeros((nrow, ncol))      # Final stego image
    numsecbit = 0  # Position index of the message bit to be located
    # Take the squares from the cover image
    for ii in np.arange(nrow//sqlen):
        for jj in np.arange(ncol//sqlen):
            square = im[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen]
            square = dct(dct(square, axis=0, norm='ortho'),
                         axis=1, norm='ortho')  # DCT
            # Hiding secret, take elements of the square
            for kk in np.arange(sqlen):
                for ll in np.arange(sqlen):
                    if (numsecbit == (len(binmes))):
                        break
                    if (ll == kk):  # Elements outside the square diagonal
                        continue
                    temp = square[kk, ll]
                    sign = 1 if (temp >= 0) else -1     # Different np.sign
                    temp = list(bin(round(np.abs(temp)))[2:])   # Binarize
                    # Hash function to deceive the eavesdroppers:
                    hashval = calculate_hashfun(temp, ii, sqlen, jj, kk, ll)
                    # Embed the secret bit in the corresponding position
                    temp[hashval] = str(binmes[numsecbit])
                    value = int(''.join(temp), 2)
                    # Put it back again in the square
                    square[kk, ll] = value * sign

                    numsecbit += 1
            # Apply the normalized IDCT to the square
            square = idct(idct(square, axis=0, norm='ortho'), axis=1,
                          norm='ortho')  # IDCT
            # Construct the stego image by filling the squares
            stego[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen] = square[:, :]
    # Sanity checks
    print('Num of bits encoded:', numsecbit)
    enc_ssim = ssim(np.double(im), np.double(stego), data_range=255.0)
    print('The SSIM comparing Cover Im and Stego Im is: %.8f' % (enc_ssim))
    return stego


def decoding(im, binmes_len, sqlen):
    '''
    Perform message dembedding process via DCT coefficients into 1-layer image.

    First, it ensures that the cover image fits the received square length and
    message length. Then, the secret binary message is extracted from the
    stego image using its DCT coefficients. Finally, it reconstructs the
    original image and compares it with the stego one.

    Args:
        im (2D-array): The stego image.
        binmes_len (int): The length of the binary message.
        sqlen (int): The square length.

    Returns (tuple):
        secmes(1D-array): The secret message.
        recons(2D-array): The reconstructed image.
    '''
    nrow, ncol = np.shape(im)
    check_image_dimensions(nrow, ncol, sqlen)
    maxbits = np.size(im) * (1. - 1./sqlen)
    check_message_length(maxbits, binmes_len)
    print('Number of bits to be dembedded:', binmes_len)
    # DECODING PROCESS
    # Initial variables
    square = np.zeros((sqlen, sqlen))
    recons = np.zeros((nrow, ncol))
    secmes = np.zeros(binmes_len, dtype='uint8')
    numsecbit = 0
    for ii in np.arange(nrow//sqlen):
        for jj in np.arange(ncol//sqlen):
            square = im[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen]
            square = dct(dct(square, axis=0, norm='ortho'), axis=1,
                         norm='ortho')
            for kk in np.arange(sqlen):
                for ll in np.arange(sqlen):
                    if numsecbit == binmes_len:
                        break
                    if ll == kk:
                        continue
                    temp = bin(round(np.abs(square[kk, ll])))[2:]
                    hashval = calculate_hashfun(temp, ii, sqlen, jj, kk, ll)
                    secmes[numsecbit] = int(temp[hashval])
                    numsecbit += 1
            square = idct(idct(square, axis=0, norm='ortho'), axis=1,
                          norm='ortho')
            recons[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen] = square[:, :]
    print('Num of bits decoded:', numsecbit)
    dec_ssim = ssim(np.double(im), np.double(recons), data_range=255.0)
    print('The SSIM comparing Cover Im and Stego Im is: %.8f' % dec_ssim)
    return secmes, recons

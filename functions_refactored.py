import numpy as np
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim


def stringtobin(mes):
    ascii_values = np.frombuffer(mes.encode(), dtype='uint8')
    return np.unpackbits(ascii_values)    # ASCII values to bits


def bintostring(binmes):
    ints = np.packbits(binmes)  # bin to int which are the mes ASCII values
    chars = np.vectorize(chr)(ints)  # Vectorized conversion of ints to chars
    return ''.join(chars)


def encoding(im: np.ndarray, binmes: np.ndarray, sqlen: int) -> np.ndarray:
    nrow, ncol = np.shape(im)
    square = np.zeros((sqlen, sqlen))
    stego = np.zeros((nrow, ncol))
    numsecbit = 0
    for ii in np.arange(nrow//sqlen):
        for jj in np.arange(ncol//sqlen):
            square = im[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen]
            square = dct(dct(square, axis=0, norm='ortho'), axis=1,
                         norm='ortho')
            for kk in np.arange(sqlen):
                for ll in np.arange(sqlen):
                    if numsecbit == len(binmes):
                        break
                    if ll == kk:
                        continue
                    temp = square[kk, ll]
                    sign = 1 if temp >= 0 else -1
                    temp = list(bin(round(np.abs(temp)))[2:])
                    hash_val = -1 if len(temp) < 4 else -((ii*sqlen+jj*sqlen +
                                                           kk+ll) % 3+1)
                    temp[hash_val] = str(binmes[numsecbit])
                    value = int(''.join(temp), 2)
                    square[kk, ll] = value * sign
                    numsecbit += 1
            square = idct(idct(square, axis=0, norm='ortho'), axis=1,
                          norm='ortho')
            stego[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen] = square[:, :]
    return stego


def decoding(im, binarymeslen, sqlen):
    '''
    We will implement a general method for every image to decode a message.        
    Mainly we have to take into account that sqlen used must be the same than
    the one used to encode the message, and that we have to previously know
    the secret message lenght, otherwise we could get in big trouble. It could
    be encoded in the image, or sent via an encryted channel.
    In this function takes place the decoding process via dct. Inputs:
    stego image (2D-array) that must fit with the sqlen (integer) decided, and
    binarymes (1D- binary array) which is the one containg the secret message.
    Our output is the reconstructed image (2D-array) that appears to be exactly
    how the cover image looks like. It makes sense because the reconstructed
    image should be completely equal to the stego image. Output: the bin sec
    mes (1D bin array), in order to aferwards do the convenient treatment to
    get the secret image or string of characters.
    '''

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
                    data_range=255.0)
    print('The SSIM comparing Cover Im and Stego Im is: %.8f' % (dec_ssim))
    return secmes, recons

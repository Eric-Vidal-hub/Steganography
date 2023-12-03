import numpy as np
from scipy.fftpack import dct, idct


def stringtobin(mes):
    ascii_values = np.frombuffer(mes.encode(), dtype='uint8')
    return np.unpackbits(ascii_values)    # ASCII values to bits


def bintostring(binmes):
    ints = np.packbits(binmes)  # bin to int which are the mes ASCII values
    chars = np.vectorize(chr)(ints)  # Vectorized conversion of ints to chars
    return ''.join(chars)


def embedding(im: np.ndarray, binmes: np.ndarray, sqlen: int) -> np.ndarray:
    nrow, ncol = np.shape(im)
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


def dembedding(im, binarymeslen, sqlen):
    nrow, ncol = np.shape(im)
    recons = np.zeros((nrow, ncol))
    secmes = np.zeros(binarymeslen, dtype='uint8')
    numsecbit = 0
    for ii in np.arange(nrow//sqlen):
        for jj in np.arange(ncol//sqlen):
            square = im[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen]
            square = dct(dct(square, axis=0, norm='ortho'), axis=1,
                         norm='ortho')
            for kk in np.arange(sqlen):
                for ll in np.arange(sqlen):
                    if numsecbit == binarymeslen:
                        break
                    if ll == kk:
                        continue
                    temp = bin(round(np.abs(square[kk, ll])))[2:]
                    hash_val = -1 if len(temp) < 4 else -((ii*sqlen+jj*sqlen +
                                                           kk+ll) % 3+1)
                    secmes[numsecbit] = int(temp[hash_val])
                    numsecbit += 1
            square = idct(idct(square, axis=0, norm='ortho'), axis=1,
                          norm='ortho')
            recons[ii*sqlen:(ii+1)*sqlen, jj*sqlen:(jj+1)*sqlen] = square[:, :]
    return secmes, recons

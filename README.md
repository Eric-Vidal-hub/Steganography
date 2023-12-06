# Steganography

This code is a steganography method based on DCT transform. It
allows to hide a secret bitstring, e.g. message or image, into a cover image. The secret message is encoded with the Huffman encoding algorithm to reduce the size of the message and make more difficult to eavesdroppers its contain.
Then, the secret is embedded using a hash function in the DCT coefficients of the cover image. So we have a stego image. The receiver will use the same hash function to extract the secret message from the stego image. Finally, the secret message is decoded with Huffman decoding to recover the original secret.

In this code, we have 3 examples:
1. Characters string
2. Secret image
3. Secret image with RGB cover image

## Libraries used
- numpy
- scipy
- matplotlib
- skimage

## Additional information
Apart from the Python code, there is a pdf where one can find a review of how all the processes work.
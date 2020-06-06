import numpy as np
import matplotlib.pyplot as plt


def decode(imprefix,start,threshold):
    """
    Given a sequence of 20 images of a scene showing projected 10 bit gray code, 
    decode the binary sequence into a decimal value in (0,1023) for each pixel.
    Mark those pixels whose code is likely to be incorrect based on the user 
    provided threshold.  Images are assumed to be named "imageprefixN.png" where
    N is a 2 digit index (e.g., "img00.png,img01.png,img02.png...")
 
    Parameters
    ----------
    imprefix : str
       Image name prefix
      
    start : int
       Starting index
       
    threshold : float
       Threshold to determine if a bit is decodeable
       
    Returns
    -------
    code : 2D numpy.array (dtype=float)
        Array the same size as input images with entries in (0..1023)
        
    mask : 2D numpy.array (dtype=logical)
        Array indicating which pixels were correctly decoded based on the threshold
    
    """
        
    # we will assume a 10 bit code
    nbits = 10
    grey_code = []
    mask = []
    
    # don't forget to convert images to grayscale / float after loading them in
    for num in range(start, start + 20, 2):
        zero = ""
        if num < 10:
            zero += "0"
            
        image1 = plt.imread(imprefix + zero + str(num) + ".png")
        if (image1.dtype == np.uint8):
            image1 = image1.astype(float) / 256
        if len(image1.shape) == 4:
            image1 = image1[:,:,:3]
        if len(image1.shape)==3:
            image1 = np.mean(image1, axis=-1)

        image2 = plt.imread(imprefix + zero + str(num + 1) + ".png")
        if (image2.dtype == np.uint8):
            image2 = image2.astype(float) / 256
        if len(image2.shape) == 4:
            image2 = image2[:,:,:3]
        if len(image2.shape)==3:
            image2 = np.mean(image2, axis=-1)
        
        if num == start:
            mask = np.ones(image1.shape)
        
        grey_code.append(image1 > image2)
        mask = mask * (abs(image1 - image2) > threshold)
    

    binary_code = [grey_code[0]]
    for x in range(nbits - 1):
        binary_code.append(np.logical_xor(binary_code[x], grey_code[x + 1]))
    
    code=np.zeros((image1.shape))
    for n in range(nbits):
        code += binary_code[9-n]*(2**n)
    mask = np.array(mask)
    
    assert mask.shape == image1.shape
    
    return code,mask

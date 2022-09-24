import numpy as np

def my_imfilter(image, imfilter):
    """function which imitates the default behavior of the build in scipy.misc.imfilter function.

    Input:
        image: A 3d array represent the input image.
        imfilter: The gaussian filter.
    Output:
        output: The filtered image.
    """
    # =================================================================================
    # TODO:                                                                           
    # This function is intended to behave like the scipy.ndimage.filters.correlate    
    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         
    # of the filter matrix.)                                                          
    # Your function should work for color images. Simply filter each color            
    # channel independently.                                                          
    # Your function should work for filters of any width and height                   
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This
    # restriction makes it unambigious which pixel in the filter is the center        
    # pixel.                                                                          
    # Boundary handling can be tricky. The filter can't be centered on pixels         
    # at the image boundary without parts of the filter being out of bounds. You      
    # should simply recreate the default behavior of scipy.signal.convolve2d --       
    # pad the input image with zeros, and return a filtered image which matches the   
    # input resolution. A better approach is to mirror the image content over the     
    # boundaries for padding.                                                         
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can 
    # see the desired behavior.                                                       
    # When you write your actual solution, you can't use the convolution functions    
    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   
    # Simply loop over all the pixels and do the actual computation.                  
    # It might be slow.                        
    
    # NOTE:                                                                           
    # Some useful functions:                                                        
    #     numpy.pad (https://numpy.org/doc/stable/reference/generated/numpy.pad.html)      
    #     numpy.sum (https://numpy.org/doc/stable/reference/generated/numpy.sum.html)                                     
    # =================================================================================

    # ============================== Start OF YOUR CODE ===============================
    padding = 0
    strides = 1
    # imfilter = np.flipud(np.fliplr(imfilter))

    # Calculate shape of output img
    xImfilterShape = imfilter.shape[0] 
    yImfilterShape = imfilter.shape[1] 
    xImgShape = image.shape[0] 
    yImgShape = image.shape[1]
    zImgShape = image.shape[2]

    xOutput = int(((xImgShape - xImfilterShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yImfilterShape + 2 * padding) / strides) + 1)
    zOutput = zImgShape
    # output = np.zeros((xOutput, yOutput, zOutput))
    output = np.zeros_like(image)

    imagePadded = image
    

    # if padding != 0:
    #     imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
    #     imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    #     print(imagePadded)
    # else:
    #     imagePadded = image

    # Iterate through image
    for z in range(image.shape[2]):
        for y in range(image.shape[1]):
            # Exit Convolution
            if y > image.shape[1] - yImfilterShape:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(image.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > image.shape[0] - xImfilterShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y, z] = (imfilter * imagePadded[x: x + xImfilterShape, y: y + yImfilterShape, z]).sum()
                    except:
                        break

    return output
   
    # =============================== END OF YOUR CODE ================================

    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can
    # see the desired behavior.
    # import scipy.ndimage as ndimage
    # output = np.zeros_like(image)
    # for ch in range(image.shape[2]):
    #    output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')

    return output
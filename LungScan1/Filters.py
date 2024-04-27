import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import os

def SaltAndPepper(image, density):
    # create an empty array with same size as input image
    output = np.zeros(image.shape, np.uint8)

    # parameter for controlling how much salt and paper are added
    threshhold = 1 - density

    # loop every each pixel and decide add the noise or not base on threshhold (density)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            possibility = random.random()
            if possibility < density:
                output[i][j] = 0
            elif possibility > threshhold:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def MeanFilter(image, filter_size):
    # create an empty array with same size as input image
    output = np.zeros(image.shape, np.uint8)

    # creat an empty variable
    result = 0

    # deal with filter size = 3x3
    if filter_size == 9:
        for j in range(1, image.shape[0]-1):
            for i in range(1, image.shape[1]-1):
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        result = result + image[j+y, i+x]
                output[j][i] = int(result / filter_size)
                result = 0

    # deal with filter size = 5x5
    elif filter_size == 25:
        for j in range(2, image.shape[0]-2):
            for i in range(2, image.shape[1]-2):
                for y in range(-2, 3):
                    for x in range(-2, 3):
                        result = result + image[j+y, i+x]
                output[j][i] = int(result / filter_size)
                result = 0

    return output

def MedianFilter(image, filter_size):
    # create an empty array with same size as input image
    output = np.zeros(image.shape, np.uint8)

    # create the kernel array of filter as same size as filter_size
    filter_array = [image[0][0]] * filter_size

    # deal with filter size = 3x3
    if filter_size == 9:
        for j in range(1, image.shape[0]-1):
            for i in range(1, image.shape[1]-1):
                filter_array[0] = image[j-1, i-1]
                filter_array[1] = image[j, i-1]
                filter_array[2] = image[j+1, i-1]
                filter_array[3] = image[j-1, i]
                filter_array[4] = image[j, i]
                filter_array[5] = image[j+1, i]
                filter_array[6] = image[j-1, i+1]
                filter_array[7] = image[j, i+1]
                filter_array[8] = image[j+1, i+1]

                # sort the array
                filter_array.sort()

                # put the median number into output array
                output[j][i] = filter_array[4]

    # deal with filter size = 5x5
    elif filter_size == 25:
        for j in range(2, image.shape[0]-2):
            for i in range(2, image.shape[1]-2):
                filter_array[0] = image[j-2, i-2]
                filter_array[1] = image[j-1, i-2]
                filter_array[2] = image[j, i-2]
                filter_array[3] = image[j+1, i-2]
                filter_array[4] = image[j+2, i-2]
                filter_array[5] = image[j-2, i-1]
                filter_array[6] = image[j-1, i-1]
                filter_array[7] = image[j, i-1]
                filter_array[8] = image[j+1, i-1]
                filter_array[9] = image[j+2, i-1]
                filter_array[10] = image[j-2, i]
                filter_array[11] = image[j-1, i]
                filter_array[12] = image[j, i]
                filter_array[13] = image[j+1, i]
                filter_array[14] = image[j+2, i]
                filter_array[15] = image[j-2, i+1]
                filter_array[16] = image[j-1, i+1]
                filter_array[17] = image[j, i+1]
                filter_array[18] = image[j+1, i+1]
                filter_array[19] = image[j+2, i+1]
                filter_array[20] = image[j-2, i+2]
                filter_array[21] = image[j-1, i+2]
                filter_array[22] = image[j, i+2]
                filter_array[23] = image[j+1, i+2]
                filter_array[24] = image[j+2, i+2]

                # sort the array
                filter_array.sort()

                # put the median number into output array
                output[j][i] = filter_array[12]
    return output



def main():
    filepathmain =str(input("Enter the path of data folder: "))
    destinationPath = str(input("Enter the destination path: "))
    mainfoldernames=["test","train","val"]
    subfoldernames =["COVID19","NORMAL","PNEUMONIA","TURBERCULOSIS"]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for j in [0,1,2]:
        for k in [0,1,2,3]:
            filepath = "%s/%s/%s"%(filepathmain,mainfoldernames[j],subfoldernames[k])
            
            i = 1
            
            for filename in os.listdir(filepath):
                f =os.path.join(filepath,filename)
                if os.path.isfile(f):
                    print("Image taken from ",f)
                
                    # read image
                    gray_lena = cv2.imread(f, 0)

            # add salt and paper (0.01 is a proper parameter)
                    noise_lena = SaltAndPepper(gray_lena, 0.01)

            # use 3x3 mean filter
                    mean_3x3_lena = MeanFilter(noise_lena, 9)

            

            # use 5x5 mean filter
                    mean_5x5_lena = MeanFilter(noise_lena, 25)

            

           
                    
                    cl1 = clahe.apply(noise_lena)
                    print("Image is saved to ",'%s/saltNpepper/%s/%s/sample%s.jpg' %(destinationPath,mainfoldernames[j],subfoldernames[k],i))
                    cv2.imwrite('%s/saltNpepper/%s/%s/claheSP%s.jpg' %(destinationPath,mainfoldernames[j],subfoldernames[k],i), cl1)



                
                    cl2 = clahe.apply(mean_3x3_lena)
                    print("Image is saved to ",'%s/3x3Mean/%s/%s/clahe3x3avg%s.jpg' %(destinationPath,mainfoldernames[j],subfoldernames[k],i))
                    cv2.imwrite('%s/3x3Mean/%s/%s/clahe3x3avg%s.jpg' %(destinationPath,mainfoldernames[j],subfoldernames[k],i), cl2)

                
                    cl3 = clahe.apply(mean_5x5_lena)
                    print("Image is saved to ",'%s/5x5Mean/%s/%s/clahe5x5avg%s.jpg' %(destinationPath,mainfoldernames[j],subfoldernames[k],i))
                    cv2.imwrite('%s/5x5Mean/%s/%s/clahe5x5avg%s.jpg' %(destinationPath,mainfoldernames[j],subfoldernames[k],i), cl3)

          
                    print("Image %s done for %s %s %s"%(i,destinationPath,mainfoldernames[j],subfoldernames[k]))
                    i=i+1       
    
    
    
    
    


#if __name__ == "__main__":
main()

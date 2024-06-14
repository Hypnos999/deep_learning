from PIL import Image
import numpy as np
import cv2

# Load the image
with Image.open(r'pooling\preprocessed.jpg') as img:
    # Convert the image to a NumPy array
    img_array = np.array(img)
    img_preprocessed = Image.fromarray(img_array)
    # img_preprocessed.show()

    # Define the pooling window size and stride
    pool_size = 2
    
    # Calculate the output shape
    output_height = int(img_array.shape[0])
    output_width = int(img_array.shape[1])

    ## Quadratare la dimensione dell'immagine
    lato_maggiore = max(img_array.shape)
    
    ## prof:
    # zero = np.zeros((lato_maggiore, lato_maggiore, img_array.shape[2]), dtype=img_array.dtype)
    # zero[:] = (200, 30, 5)
    # altezza, base, canali = img_array.shape
    # zero[:altezza, :base] = img_array
    # img_quadrata_prof = Image.fromarray(zero)
    # img_quadrata_prof.show()
    # img_quadrata_prof.save(r'pooling\squared_prof.jpg')
    
    ## mio metodo:
    ## migliore perchè centro l'immmagine, invece che metterla all'angolo
    diff = int(abs(output_height - output_width)/2)
    if output_width < output_height:
        ## immagine è rettangolare in verticale
        zero = np.zeros((img_array.shape[0], diff, img_array.shape[2]), dtype=img_array.dtype)
    else:
        ## immagine è rettangolare in orizzontale
        zero = np.zeros((diff, img_array.shape[1], img_array.shape[2]), dtype=img_array.dtype)

    img_quadrata_array = np.concatenate((zero, img_array, zero), axis=0)
    img_quadrata = Image.fromarray(img_quadrata_array)
    # img_quadrata.show()
    img_quadrata.save(r'pooling\squared.jpg')
    

    ## Max pooling
    ## mio metodo:
    ## questo metodo ritorna una immagine in scala di grigi, anche se non so come possa farlo in autonomia

    
    gray_arrray = cv2.cvtColor(img_quadrata_array, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Scala di grigi', gray_arr)
    # cv2.waitKey(0)
    
    output_height, output_width = gray_arrray.shape[0] // pool_size, gray_arrray.shape[1] // pool_size
    pooled_array = np.zeros((output_height, output_width), dtype=img_array.dtype)
    for i in range(output_height):
        for j in range(output_width):
            window = gray_arrray[
                i*pool_size : i*pool_size+pool_size, 
                j*pool_size : j*pool_size+pool_size
            ]

            pooled_array[i, j] = np.max(window)
    
    # Convert the pooled array back to an image
    # print(pooled_array.shape)
    pooled_img = Image.fromarray(pooled_array)
    pooled_img.save(r'pooling\processed.jpg')
    pooled_img.show()
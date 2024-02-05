from PIL import Image
import numpy as np
import time
import argparse
#time

def Kmeans(cents, pixm, fl):
    ncents = []
    ml = []
    for cent in cents:
        pixnorm = pixm - np.array(cent)
        pixnorm = np.apply_along_axis(np.linalg.norm, 2, pixnorm)
        #pixnorm = np.linalg.norm(pixnorm, axis = 2)
        ml.append(pixnorm)
    ml = np.array(ml)
    result = np.argmin(ml, axis=0)
    pixm_r = pixm.reshape(pixm.shape[0]*pixm.shape[1], pixm.shape[2])
    for i in range(int(ml.shape[0])):
        c = (result == i)
        d = c.reshape(pixm.shape[0]*pixm.shape[1], 1)
        e = np.repeat(d, 3, axis = 1)
        e = np.invert(e)
        f = np.ma.array(pixm_r, mask= e)
        ncents.append(np.mean(f, axis = 0).data)
        if fl == True:
            e = np.invert(e)
            pixm_r[:, 0][e[:, 0]] = int(cents[i][0])
            pixm_r[:, 1][e[:, 1]] = int(cents[i][1])
            pixm_r[:, 2][e[:, 2]] = int(cents[i][2])

    if fl == True:
        pixm_c = pixm_r.reshape([pixm.shape[0], pixm.shape[1], pixm.shape[2]])
        return pixm_c
    else:
        return [list(x) for x in ncents]

def main():

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-image', '--img', type=str)
    parser.add_argument('-K', '--k', type=str)

    arg = parser.parse_args()
    img = arg.img
    K = int(arg.k)
    #opening the image
    image = Image.open(img)
    #get no. of pixels in length and width of the image
    width, height = image.size
    image.load()
    #load image as a numpy array
    pixel_mat = np.asarray(image, dtype= "int32")

    flag = False
    converged = False
    #list of previous centroids
    prev_centroids = []
    #choosing random centroids
    for i in range(K):
        prev_centroids.append([np.random.randint(256), np.random.randint(256), np.random.randint(256)])

    print("Random centroids : ", prev_centroids)

    #run the program for 500 epochs and break if the algorithm converges
    for i in range(500):
        if (i+1) % 50 == 0:
            print("{}th epoch".format(i+1))
        new_centroids = Kmeans(prev_centroids, pixel_mat, flag)
        if prev_centroids == new_centroids:
            converged = True
            break
        else:
            prev_centroids = new_centroids
    #retun the clustering matrix and convert to image
    flag = True
    new_mat = Kmeans(prev_centroids, pixel_mat, flag)
    new_mat = new_mat.astype(np.uint8)
    image_new = Image.fromarray(new_mat, "RGB")
    image_new.save("Koala K = {}, E = {}.png".format(K, i + 1))
    if converged:
        print("Kmeans converged at {} epoch ".format(i + 1))
    else:
        print("Kmeans not converged, algo ran for {} epochs".format(i + 1))
    end = time.time() - start
    print("Time taken : {}".format(end))

main()
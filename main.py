import cv2
import os
import glob
import numpy as np
import math
from tqdm import trange 

# utils
def run_alignment(images):
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)
    return images

def init_weight_function():
    Zmin = 0 
    Zmax = 255 
    Zmid = (Zmax+Zmin)//2
    w = np.zeros((Zmax-Zmin+1))

    for z in range(Zmin,Zmax+1): 
        if z <= Zmid:
            w[z] = z - Zmin + 1
        else: 
            w[z] = Zmax - z + 1
    return w

# utils
def sampling(images, n_channels=3):
    # To solve the ODE, we need N(P-1) > (255-0) => P>255/N - 1
    N = len(images)
    P = math.ceil(255*2/(N-1))*2 # ?

    # choose random images 
    # images[0].shape = (1424, 2144, 3)
    width, height = images[0].shape[0], images[0].shape[1]

    sample_idx = (np.random.choice(width, P), np.random.choice(height, P))

    sample = np.zeros((N, P, n_channels), dtype=np.uint8)
    for i in range(N):
        sample[i] = images[i][sample_idx]
        
    return sample

# load_data
def load_data(data_name = "exposures_2"):
    root = os.getcwd()
    img_path = os.path.join(root, data_name,"*.JPG")

    filenames = sorted(glob.glob(img_path))
    images = [cv2.imread(fn) for fn in filenames]
    # print(len(images))
    
    exposure_times = 1/np.array([2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0], dtype=np.float32)
    return images, exposure_times

# solve_response_curve
def solve_response_curve(sample, exposure_times, l=100):
    # print(sample.shape)
    # print(exposure_times.shape)
    Zmin = 0 
    Zmax = 255 
    Z_range = Zmax-Zmin # 255
    # A: (NP+255) by (256+N)
    A = np.zeros((sample.shape[0]*sample.shape[1]+Z_range, Z_range+1+sample.shape[0]), dtype=np.float64)
    # b: (NP+255) by (1)
    b = np.zeros((A.shape[0], 1), dtype=np.float64)
    # weight function w(z)
    w = init_weight_function()
    
    k = 0
    # NP equations
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            I_ij = sample[i,j]
            w_ij = w[I_ij]
            A[k, I_ij] = w_ij
            A[k, 255 + 1 + i] = -w_ij
            b[k, 0] = w_ij * exposure_times[i]
            k += 1
    # Color normalize equation: g(127)=1
    A[k, int((Zmax - Zmin) // 2)] = 1
    k += 1 
    # 254 equations
    for I_k in range(Zmin+1, Zmax):
        w_k = w[I_k]
        A[k, I_k-1] = w_k * l
        A[k, I_k] = -2 * w_k * l
        A[k, I_k+1] = w_k * l
        k += 1    

    # Solve
    inv_A = np.linalg.pinv(A)
    x = np.dot(inv_A, b)

    # Finally, g is obtained.
    g = x[0:Z_range+1]
    # print(g)

    return g[:,0]

# compute_radiance(
def compute_radiance(images, exposure_times, g):
    n_channels = g.shape[-1]
    num_images = len(images)
    width, height = images[0].shape[0], images[0].shape[1]
    img_rad = np.zeros((width, height, n_channels), dtype=np.float64)
    
    # weight function w(z)
    w = init_weight_function()  
    for c in range(n_channels):
        g_channel = g[:, c]

        for i in trange(width):
            for j in range(height):

                g_value = np.array([g_channel[ int(images[k][i, j, c]) ] for k in range(num_images) ])
                w_value = np.array([w[ int(images[k][i, j, c]) ] for k in range(num_images) ])
                # print(g_value, w_value)
                
                sumW = np.sum(w_value)
                if sumW > 0:
                    img_rad[i,j,c] = np.sum(w_value * (g_value - exposure_times) / sumW)
                else:
                    img_rad[i,j,c] = g_value[num_images // 2] - exposure_times[num_images //2]

    return img_rad

def globalTonemap(img, l):
    return cv2.pow(img/255., 1.0/l)



def main():
    print("main")
    
    images, exposure_times = load_data()
    # print(exposure_times)

    n_channels = images[0].shape[-1]
    hdr_img = np.zeros(images[0].shape, dtype=np.float64)

    # Run MTB
    images = run_alignment(images)
    
    # Sample some indices from image.
    sample = sampling(images, n_channels)
    # print(sample[:,:,0].shape)
    
    # For each individual channel:
    Zmin = 0 
    Zmax = 255 
    Z_range = Zmax-Zmin+1 # 256
    g = np.zeros((Z_range, n_channels), dtype=np.float64)
    for c in range(n_channels):
        # Solve ODE.
        g[:, c]= solve_response_curve(sample[:,:,c], exposure_times, l=100)
    
    img_rad = compute_radiance(images, exposure_times, g)

    hdr_img = cv2.normalize(img_rad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    cv2.imwrite('./new_original.hdr', hdr_img)
    # Tone mapping
    if True:
        output = np.uint8(globalTonemap(hdr_img, 1.3) * 255.)

if __name__ == "__main__":
    main()
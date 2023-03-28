import cv2
import os
import glob
import numpy as np
import math
from tqdm import trange 
from matplotlib.pylab import cm
import matplotlib.pylab as plt



# utils
def run_alignment(images):
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)
    return images

def init_weight_function():
    Zmin = 0 
    Zmax = 255 
    Zmid = (Zmax+Zmin+1)//2
    w = np.zeros((Zmax-Zmin+1))

    for z in range(Zmin,Zmax+1): 
        if z <= Zmid:
            w[z] = z - Zmin
        else: 
            w[z] = Zmax - z
    return w

# utils
def sampling(images, n_channels=3):
    # To solve the ODE, we need N(P-1) > (255-0) => P>255/N - 1
    N = len(images)
    P = math.ceil(255*2/(N-1))*2 # ?
    Zmin = 0 
    Zmax = 255 
    # P = Zmax-Zmin

    # choose random images 
    # images[0].shape = (1424, 2144, 3)
    width, height = images[0].shape[0], images[0].shape[1]

    sample_idx = (np.random.choice(width, P), np.random.choice(height, P))

    sample = np.zeros((N, P, n_channels), dtype=np.uint8)
    for i in range(N):
        sample[i] = images[i][sample_idx]
        
    return sample

# load_data
def load_data(data_name):
    root = os.getcwd()
    img_path = os.path.join(root, data_name,"*.JPG")

    filenames = sorted(glob.glob(img_path))
    # print(filenames)
    images = [cv2.imread(fn) for fn in filenames]
    # print(len(images))
    
    exposure_times = 1/np.array([4.0, 8.0, 15.0, 30.0, 60.0, 125.0, 250.0, 500.0, 1000.0, 2000.0], dtype=np.float32)
    
    # important: 取log
    ln_exposure_times = np.log(exposure_times)
    return images, ln_exposure_times

# solve_response_curve
def solve_response_curve(sample, ln_exposure_times, l=100):
    # print(sample.shape)
    # print(ln_exposure_times.shape)
    Zmin = 0 
    Zmax = 255 
    Z_range = Zmax-Zmin+1 # 256
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
            w_ij = w[int(I_ij)]
            A[k, I_ij] = w_ij
            A[k, Z_range + i] = -w_ij
            b[k, 0] = w_ij * ln_exposure_times[i]
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
    g = x[0:Z_range]
    # print(g)

    return g[:,0]

# compute_radiance(
def compute_radiance(images, ln_exposure_times, g):
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
                    img_rad[i,j,c] = np.sum(w_value * (g_value - ln_exposure_times) / sumW)
                else:
                    img_rad[i,j,c] = g_value[num_images // 2] - ln_exposure_times[num_images //2]

    return img_rad

def globalTonemap(img, l):
    return cv2.pow(img/255., 1.0/l)



def run_Debevec(data_name):
    print("Run Debevec")
    print(data_name)
    prefix = f'./result_Debevec_{data_name}/'
    os.makedirs(prefix,exist_ok=True)
    images, ln_exposure_times = load_data(data_name)
    # print(exposure_times)

    n_channels = images[0].shape[-1]
    hdr_img = np.zeros(images[0].shape, dtype=np.float64)

    # Run MTB
    if True:
        images = run_alignment(images)
    
    # Sample some indices from image.
    sample = sampling(images, n_channels)
    # print(sample[:,:,0].shape)
    
    # For each individual channel:
    Zmin = 0 
    Zmax = 255 
    Z_range = Zmax-Zmin+1 # 256
    g = np.zeros((Z_range, n_channels), dtype=np.float64)


    x = np.arange(0, Z_range, 1)
    color = ['r', 'g', 'b']
    for c in range(n_channels):

        # Solve ODE.
        g[:, c]= solve_response_curve(sample[:,:,c], ln_exposure_times, l=100)
        
        # Show curve
        plt.figure()
        plt.plot( x, g[:, c], color=color[c])
        plt.savefig(f'{prefix}/curve_{color[c]}.jpg') 

    img_rad = compute_radiance(images, ln_exposure_times, g)
    hdr_img = cv2.normalize(img_rad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite(f'{prefix}/{data_name}.hdr', hdr_img)
    # plt.figure()
    # plt.imsave(f'./result_Debevec/{data_name}.hdr', hdr_img)
    
    # Save cmap
    colorize = cm.jet
    cmap = np.float32(cv2.cvtColor(np.uint8(hdr_img), cv2.COLOR_BGR2GRAY)/255.)
    cmap = colorize(cmap)
    # cv2.imwrite(f'./result_Debevec/{data_name}_cmap.jpg', np.uint8(cmap*255.))
    plt.figure()
    plt.imsave(f'{prefix}/{data_name}_cmap.jpg', np.uint8(cmap*255.), cmap="jet")
    
    # Gamma tone mapping
    if True:
        Gamma = np.uint8(globalTonemap(hdr_img, 1.5) * 255.)
        # output = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #plt.figure()
        cv2.imwrite(f'{prefix}/{data_name}_gamma_tomemapping.jpg', Gamma)

    # Mantiuk tone mapping
    if True:
        tm = cv2.createTonemapMantiuk()
        ldrMantiuk = np.uint8(255. * tm.process((hdr_img/255.).astype(np.float32)))
        #plt.figure()
        cv2.imwrite(f'{prefix}/{data_name}_Mantiuk_tomemapping.jpg', ldrMantiuk)

    # # Drago tone mapping
    # if True:
    #     tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    #     ldrDrago = tonemapDrago.process(hdr_img/255.)
    #     ldrDrago = 3 * ldrDrago
    #     cv2.imwrite(f'./result_Debevec_{data_name}/{data_name}_Drago_tomemapping.jpg', ldrDrago * 255)

    # if True: has no attribute
    #     tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
    #     ldrDurand = tonemapDurand.process(hdr_img/255.)
    #     ldrDurand = 3 * ldrDurand
    #     cv2.imwrite(f'./result_Debevec_{data_name}/{data_name}_Durand_tomemapping.jpg', ldrDurand * 255)

    # Tonemap using Reinhard's method to obtain 24-bit color image
    if True:
        # hdr_img is CV_32FC3 ?
        tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        ldrReinhard = tonemapReinhard.process(np.float32(hdr_img))
        cv2.imwrite(f'{prefix}/{data_name}_Reinhard_tomemapping.jpg', ldrReinhard * 255)

    if True:
        plt.figure()
        x = np.arange(0, Z_range, 1)
        color = ['r', 'g', 'b']
        for c in range(n_channels):
            # Show curve
            plt.plot( x, g[:, c], color=color[c])
        plt.savefig(f'{prefix}/recovery_curve.jpg') 
        
if __name__ == "__main__":
    run_Debevec(data_name="exposures_2")
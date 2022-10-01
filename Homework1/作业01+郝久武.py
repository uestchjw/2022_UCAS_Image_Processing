

''' Author: UCAS-Hao Jiuwu '''
''' Date: 2022-10-1 '''

# %%
''' Problem 1 '''
import matplotlib.pyplot as plt
import numpy as np
import cv2,math
# 扫描函数
def scanLine4e(f,I,loc):
    if loc == 'row':
        return f[I,:]
    elif loc == 'column':
        return f[:,I]
    else:
        raise Exception('Wrong loc!')
# 找到中心行和中心列，扫描后可视化输出
def extract(path):
    img = cv2.imread(path,flags=0)
    rows,columns = img.shape[0],img.shape[1]
    if rows%2 == 1:
        crow_value = scanLine4e(img,math.floor(rows/2),'row')
    else:
        crow_value = (scanLine4e(img,int(rows/2)-1,'row') + scanLine4e(img,int(rows/2),'row'))/2

    if columns%2 == 1:
        ccolumn_value = scanLine4e(img,math.floor(columns/2),'column')
    else:
        ccolumn_value = (scanLine4e(img,int(columns/2)-1,'column') + scanLine4e(img,int(columns/2),'column'))/2

    # 可视化部分
    ax1 = plt.subplot(1,2,1)
    font1 = {'family':'Times New Roman','weight':'normal','size':20}
    plt.plot(np.arange(1,len(crow_value)+1),crow_value)
    plt.title('Central row',font1)

    x_kedu = ax1.get_xticklabels()
    [i.set_fontname('Times New Roman') for i in x_kedu]
    y_kedu = ax1.get_yticklabels()
    [i.set_fontname('Times New Roman') for i in y_kedu]
    plt.tick_params(labelsize = 20)

    ax2 = plt.subplot(1,2,2)
    font1 = {'family':'Times New Roman','weight':'normal','size':20}
    plt.plot(np.arange(1,len(ccolumn_value)+1),ccolumn_value)
    plt.title('Central column',font1)

    x_kedu = ax2.get_xticklabels()
    [i.set_fontname('Times New Roman') for i in x_kedu]
    y_kedu = ax2.get_yticklabels()
    [i.set_fontname('Times New Roman') for i in y_kedu]
    plt.tick_params(labelsize = 20)
    plt.show()
    return crow_value,ccolumn_value
path1 = r'C:\Users\Lenovo\Desktop\Images\cameraman.tif'
path2 = r'C:\Users\Lenovo\Desktop\Images\einstein.tif'
r1,c1 = extract(path1)
r2,c2 = extract(path2)


# %%
''' Problem 2 '''
import cv2
def rgb1gray(f,method='NTSC'):
    if method == 'average':
        gray = (f[:,:,0]+f[:,:,1]+f[:,:,2])/3
        return gray.astype(np.uint8)
    elif method == 'NTSC':
        gray = 0.1140*f[:,:,0] + 0.5870*f[:,:,1] + 0.2989*f[:,:,2]
        return gray.astype(np.uint8)
    else:
        raise Exception('Wrong method!')
path1 = r'C:\Users\Lenovo\Desktop\Images\mandril_color.tif'
path2 = r'C:\Users\Lenovo\Desktop\Images\lena512color.tiff'
img1,img2 = cv2.imread(path1),cv2.imread(path2)
gray11,gray12 = rgb1gray(img1,'average'),rgb1gray(img1)
gray21,gray22 = rgb1gray(img2,'average'),rgb1gray(img2)
img1gray,img2gray = cv2.imread(path1,flags=0),cv2.imread(path2,flags=0)
gray = [gray11,gray12,img1gray,gray21,gray22,img2gray]
kk = 1
for i in gray:
    cv2.imwrite(f'C:/Users/Lenovo/Desktop/Images/{kk}.png',i)
    kk += 1
a = (gray12 == img1gray).all()
print(a)


# %%
''' Problem 3 '''
def twodConv(f,w,o='zero'): 
    # 若是卷积，则w需要进行旋转
    w = np.fliplr(np.flipud(w))
    kerner_size = w.shape[0]
    r,c = f.shape[0],f.shape[1]
    # default: stride=1, kerner_size is odd
    p = int((kerner_size-1)/2)
    # F是padding后的矩阵
    F = np.zeros((r+2*p,c+2*p))
    F[p:p+r,p:p+c] = f
    if o == 'replicate':
        F[0:p,0:p] = np.ones((p,p))*f[0,0]
        F[0:p,-p:] = np.ones((p,p))*f[0,-1]
        F[-p:,0:p] = np.ones((p,p))*f[-1,0]
        F[-p:,-p:] = np.ones((p,p))*f[-1,-1]
        for i in range(p):
            F[i,p:p+c] = f[0,:]
            F[p:p+r,i] = f[:,0]
            F[p+r+i,p:p+c] = f[-1,:]
            F[p:p+r,p+c+i] = f[:,-1]
    g = np.zeros_like(f,dtype=float)
    for i in range(r):
        for j in range(c):
            area = F[i:i+kerner_size,j:j+kerner_size]
            g[i,j] = round(np.sum(np.multiply(area,w)))   # !!!!! 这里不能是int，因为int直接舍弃小数；也不能不用round，下面指定unint8，因为同样会舍弃小数
            if g[i,j] > 255:
                g[i,j] = 255
    return g.astype(np.uint8)

# %%
''' Problem 4 '''
def gaussKernel(sig,m=0):
    M = 1 + 2*math.ceil(3*sig)
    # 没有提供m时
    if m == 0:
        m = M
    elif m < M:
        raise Warning('The giving size of gaussKernel is too small!')
    else:
        m = m 
    l = int((m-1)/2)
    kernel = []
    for i in range(-l,l+1):
        for j in range(-l,l+1):
            kernel.append(math.exp( -(i**2+j**2)/(2*sig**2)))
    kernel /= np.sum(kernel)
    kernel = np.reshape(np.array(kernel),(m,m))
    return kernel


# %%
''' Problem 5 '''
path1 = r'C:\Users\Lenovo\Desktop\Images\cameraman.tif'
path2 = r'C:\Users\Lenovo\Desktop\Images\einstein.tif'
img1,img2 = cv2.imread(path1,flags=0),cv2.imread(path2,flags=0)
images = [img1,img2,gray12,gray22]
sigma = [1,2,3,5]
for img in images:
    for sig in sigma:
        kernel = gaussKernel(sig)
        result = twodConv(img,kernel)
        cv2.imshow('a',result)
        cv2.waitKey(0)

# sig = 1时
# M为高斯滤波器大小
M = 1 + 2*math.ceil(3*1)
i = 1
for img in images:
    kernel = gaussKernel(1)
    result1 = twodConv(img,kernel)
    result2 = cv2.GaussianBlur(img,(M,M),1,borderType=cv2.BORDER_CONSTANT)
    result3 = cv2.GaussianBlur(img,(M,M),1,borderType=cv2.BORDER_REPLICATE)
    
    # dif = abs(result1 - result2)  # 不能这样：因为uint8相减，结果只有0和255，如灰度17-16=1，但是显示出来就是255
    dif = abs(result1.astype(int) - result2.astype(int))
    dif255 = result1-result2
    # print(np.max(dif))
    cv2.imwrite(f'C:/Users/Lenovo/Desktop/Images/Problem_5/{i}_me.png',result1)
    cv2.imwrite(f'C:/Users/Lenovo/Desktop/Images/Problem_5/{i}_offical.png',result2)
    cv2.imwrite(f'C:/Users/Lenovo/Desktop/Images/Problem_5/{i}_offical_replicate.png',result3)
    cv2.imwrite(f'C:/Users/Lenovo/Desktop/Images/Problem_5/{i}_dif.png',dif)
    cv2.imwrite(f'C:/Users/Lenovo/Desktop/Images/Problem_5/{i}_dif255.png',dif255)
    i += 1

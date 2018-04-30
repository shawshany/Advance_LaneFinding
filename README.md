本篇博客整个项目源码：[github](https://github.com/shawshany/Advance_LaneFinding)
# 引言
前面我们介绍车道线检测的处理方法：[车道线检测之lanelines-detection](https://blog.csdn.net/u010665216/article/details/80057921)
在文章末尾，我们分析了该算法的鲁棒性，当时我们提出了一些解决方法，比如说：
>* 角度滤波器：滤除极小锐角或极大钝角的线段
>* 选取黄色的色调，并用白色代替
>* 在边缘检测前，放大一些特征

但是上述算法还存在一个问题：在弯道处无法检测车道线，因此本篇博客将分享一种高级的车道线检测技术，同时也是Udacity无人驾驶的第四个项目。
本项目主要实现如下几个功能：
>* 摄像头校正
>* 对原始图像做校正
>* 使用图像处理技术来获得一个二值化的图像
>* 应用角度变换来获得二值化图像的birds-eye view(鸟瞰图)
>* 检测车道像素（histogram）并获得车道边界
>* 计算车道弯曲率及基于车道中心的车辆位置
>* 在原始图像上包裹住车道边界，用绿色色块表示
>* 将车道线进行可视化显示，并实时计算车道曲率及车辆位置

**NOTE**:关于视频流的处理请参考[Advance-Lanefinding](https://github.com/shawshany/Advance_LaneFinding)
为了讲解方便，本部分代码，我选取了一帧来说明
# 获取车道线的算法流程图
本项目中，我们检测车道线的算法流程图如下所示：
![这里写图片描述](https://img-blog.csdn.net/20180430215559153?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
# 项目实现及代码注解



```python
# import some libs we need
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from skimage import morphology
from collections import deque
%matplotlib inline
```

为了接下来的代码中，可视化输出的结果，这里我们实现一个图像显示算法


```python
#Display
showMe = 0
def display(img,title,color=1):
    '''
    func:display image
    img: rgb or grayscale
    title:figure title
    color:show image in color(1) or grayscale(0)
    '''
    if color:
        plt.imshow(img)
    else:
        plt.imshow(img,cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
```

# Calibration Camera
接下来我们来实现摄像头校准的函数,这里我们参考[Opencv-python官方教程](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html?highlight=findchessboardcorners)


```python
def camera_calibration(folder,nx,ny,showMe=0):
    '''
    使用opencv的findChessBoardCorners函数来找到所有角落的(x,y)坐标
    folder:校准图像的目录
    nx:x轴方向期望的角落个数
    ny:y轴方向期望的角落个数
    返回一个字典：
    ret:校正的RMS误差
    mtx:摄像头矩阵
    dist:畸变系数
    rvecs:旋转向量
    tvecs：平移向量
    '''
    
    #存储物体实际三维空间坐标及图像中图像的坐标
    objpoints = []#3D
    imgpoints = []
    objp = np.zeros((nx*ny,3),np.float32)
    #print(objp.shape)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)   
    assert len(folder)!=0
    #print(len(folder))
    for fname in folder:
        img = cv2.imread(fname)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,corners = cv2.findChessboardCorners(gray,(nx,ny))
        img_sz = gray.shape[::-1]#倒序输出
        #print(corners)
        #if ret is True , find corners
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            if showMe:
                draw_corners = cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
                display(draw_corners,'Found all corners:{}'.format(ret))
    if len(objpoints)==len(imgpoints) and len(objpoints)!=0:
        ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_sz,None,None)
        return {'ret':ret,'cameraMatrix':mtx,'distorsionCoeff':dist,\
               'rotationVec':rvecs,'translationVec':tvecs}
    else:
        raise  Error('Camera Calibration failed')
def correction(image,calib_params,showMe=0):
    '''
    失真矫正
    calib_params:由摄像头矫正返回的矫正参数
    '''
    corrected = cv2.undistort(image,calib_params['cameraMatrix'],calib_params['distorsionCoeff'],\
                              None,calib_params['cameraMatrix'])
    if showMe:
        display(image,'Original',color=1)
        display(corrected,'After correction',color=1)
    return corrected
```

接下来我们来看看，寻找棋盘角点及实际场景下校正的结果


```python
nx = 9
ny = 6
folder_calibration = glob.glob("camera_cal/calibration[1-3].jpg")
#print(folder_calibration)
calib_params = camera_calibration(folder_calibration,nx,ny,showMe=1)
print('RMS Error of Camera calibration:{:.3f}'.format(calib_params['ret']))
print('This number must be between 0.1 and 1.0')
imgs_tests = glob.glob("test_images/*.jpg")
original_img = np.random.choice(imgs_tests)
original_img = cv2.imread("test_images/straight_lines2.jpg")
original_img = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
corr_img = correction(original_img,calib_params,showMe=1)
```


![png](https://img-blog.csdn.net/20180430214456131?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/20180430214504218?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


    RMS Error of Camera calibration:1.208
    This number must be between 0.1 and 1.0



![png](https://img-blog.csdn.net/20180430214515892?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/20180430214523479?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


如上图所示，摄像头首先通过标准黑白棋盘来获得校正系数，然后再通过校正系数来对车辆前置摄像头拍摄的图片来应用校正系数。
接下来，我们获取图像的灰度图


```python
gray_ex = cv2.cvtColor(corr_img,cv2.COLOR_RGB2GRAY)
display(gray_ex,'Apply Camera Correction',color=0)
```


![png](https://img-blog.csdn.net/20180430214533240?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


## Threshold Binary
获得了图像的灰度图后，我们再沿着X轴方向求梯度


```python
def directional_gradient(img,direction='x',thresh=[0,255]):
    '''
    使用Opencv Sobel算子来求方向梯度
    img:Grayscale
    direction:x or y axis
    thresh:apply threshold on pixel intensity of gradient image
    output is binary image
    '''
    if direction=='x':
        sobel = cv2.Sobel(img,cv2.CV_64F,1,0)
    elif direction=='y':
        sobel = cv2.Sobel(img,cv2.CV_64F,0,1)
    sobel_abs = np.absolute(sobel)#absolute value
    scaled_sobel = np.uint8(sobel_abs*255/np.max(sobel_abs))
    binary_output = np.zeros_like(sobel)
    binary_output[(scaled_sobel>=thresh[0])&(scaled_sobel<=thresh[1])] = 1
    return binary_output
gradx_thresh = [25,255]
gradx = directional_gradient(gray_ex,direction='x',thresh = gradx_thresh)
display(gradx,'Gradient x',color=0)
```


![png](https://img-blog.csdn.net/20180430214542889?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


## Color Transform
然后我们将undistorted image 转换成HLS，并只保留S通道，二值化


```python
def color_binary(img,dst_format='HLS',ch=2,ch_thresh=[0,255]):
    '''
    Color thresholding on channel ch
    img:RGB
    dst_format:destination format(HLS or HSV)
    ch_thresh:pixel intensity threshold on channel ch
    output is binary image
    '''
    if dst_format =='HSV':
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        ch_binary = np.zeros_like(img[:,:,int(ch-1)])
        ch_binary[(img[:,:,int(ch-1)]>=ch_thresh[0])&(img[:,:,int(ch-1)]<=ch_thresh[1])] = 1
    else:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
        ch_binary = np.zeros_like(img[:,:,int(ch-1)])
        ch_binary[(img[:,:,int(ch-1)]>=ch_thresh[0])&(img[:,:,int(ch-1)]<=ch_thresh[1])] = 1
    return ch_binary
ch_thresh = [50,255]
ch3_hls_binary = color_binary(corr_img,dst_format='HLS',ch=3,ch_thresh=ch_thresh)
display(ch3_hls_binary,'HLS to Binary S',color=0)
```


![png](https://img-blog.csdn.net/20180430214553970?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


接下来我们将Gradx与Binary S进行结合


```python
combined_output = np.zeros_like(gradx)
combined_output[((gradx==1)|(ch3_hls_binary==1))] = 1
display(combined_output,'Combined output',color=0)
```


![png](https://img-blog.csdn.net/2018043021461013?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


接下来我们再来应用ROI mask来去除无关的背景信息


```python
mask = np.zeros_like(combined_output)
vertices = np.array([[(100,720),(545,470),(755,470),(1290,720)]],dtype=np.int32)
cv2.fillPoly(mask,vertices,1)
masked_image = cv2.bitwise_and(combined_output,mask)
display(masked_image,'Masked',color=0)
```


![png](https://img-blog.csdn.net/20180430214618319?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


从上图我们可以看到有些散乱的小点，这些点很明显是噪声，我们可以利用morphology.remove_small_objects()函数来去除这些杂乱点


```python
min_sz = 50
cleaned = morphology.remove_small_objects(masked_image.astype('bool'),min_size=min_sz,connectivity=2)
display(cleaned,'cleaned',color=0)
```


![png](https://img-blog.csdn.net/20180430214632232?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


## Perspective Transform
这一步我将利用前面的ImageProcess类中的birds_eye()来实现将undistorted image to a 'birds eye view' of the road.
这样操作后将有利于后面我们来拟合直线及测量曲率


```python
def birdView(img,M):
    '''
    Transform image to birdeye view
    img:binary image
    M:transformation matrix
    return a wraped image
    '''
    img_sz = (img.shape[1],img.shape[0])
    img_warped = cv2.warpPerspective(img,M,img_sz,flags = cv2.INTER_LINEAR)
    return img_warped
def perspective_transform(src_pts,dst_pts):
    '''
    perspective transform
    args:source and destiantion points
    return M and Minv
    '''
    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts,src_pts)
    return {'M':M,'Minv':Minv}
# original image to bird view (transformation)
src_pts = np.float32([[240,720],[575,470],[735,470],[1200,720]])
dst_pts = np.float32([[240,720],[240,0],[1200,0],[1200,720]])
transform_matrix = perspective_transform(src_pts,dst_pts)
warped_image = birdView(cleaned*1.0,transform_matrix['M'])
display(cleaned,'undistorted',color=0)
display(warped_image,'BirdViews',color=0)
```


![png](https://img-blog.csdn.net/20180430214641211?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/20180430214651421?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


从上述图像，我们可以看到图片底部有一部分白色，这部分是前置摄像头拍摄的汽车front-end
我们可以将底部的这部分白色截取调


```python
bottom_crop = -40
warped_image = warped_image[0:bottom_crop,:]
```

## 车道线像素检测
这里车道线像素检测，我们主要采用Histogram法，一种基于像素直方图的算法
这里分别计算一张图片底部的左半部分及右半部分的histogram


```python
def find_centroid(image,peak_thresh,window,showMe):
    '''
    find centroid in a window using histogram of hotpixels
    img:binary image
    window with specs {'x0','y0','width','height'}
    (x0,y0) coordinates of bottom-left corner of window
    return x-position of centroid ,peak intensity and hotpixels_cnt in window
    '''
    #crop image to window dimension 
    mask_window = image[round(window['y0']-window['height']):round(window['y0']),
                        round(window['x0']):round(window['x0']+window['width'])]
    histogram = np.sum(mask_window,axis=0)
    centroid = np.argmax(histogram)
    hotpixels_cnt = np.sum(histogram)
    peak_intensity = histogram[centroid]
    if peak_intensity<=peak_thresh:
        centroid = int(round(window['x0']+window['width']/2))
        peak_intensity = 0
    else:
        centroid = int(round(centroid+window['x0']))
    if showMe:
        plt.plot(histogram)
        plt.title('Histogram')
        plt.xlabel('horzontal position')
        plt.ylabel('hot pixels count')
        plt.show()
    return (centroid,peak_intensity,hotpixels_cnt)
def find_starter_centroids(image,x0,peak_thresh,showMe):
    '''
    find starter centroids using histogram
    peak_thresh:if peak intensity is below a threshold use histogram on the full height of the image
    returns x-position of centroid and peak intensity
    '''
    window = {'x0':x0,'y0':image.shape[0],'width':image.shape[1]/2,'height':image.shape[0]/2}  
    # get centroid
    centroid , peak_intensity,_ = find_centroid(image,peak_thresh,window,showMe)
    if peak_intensity<peak_thresh:
        window['height'] = image.shape[0]
        centroid,peak_intensity,_ = find_centroid(image,peak_thresh,window,showMe)
    return {'centroid':centroid,'intensity':peak_intensity}
# if number of histogram pixels in window is below 10,condisder them as noise and does not attempt to get centroid 
peak_thresh = 10
showMe = 1
centroid_starter_right = find_starter_centroids(warped_image,x0=warped_image.shape[1]/2,
                                               peak_thresh=peak_thresh,showMe=showMe)
centroid_starter_left = find_starter_centroids(warped_image,x0=0,peak_thresh=peak_thresh,
                                              showMe=showMe)
```


![png](https://img-blog.csdn.net/20180430214710205?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/20180430214720939?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


如上图所示，warped_image右半边的车道线处对应的histogram有个峰值
接下来，我们需要通过运行滑动窗口，并记录hotpixels($\neq$0)的坐标。
该窗口的width=120px,height=68px。该窗口从底部开始向上扫描，并记录hotpixels的坐标。


```python
def run_sliding_window(image,centroid_starter,sliding_window_specs,showMe = showMe):
    '''
    Run sliding window from bottom to top of the image and return indexes of the hotpixels associated with lane
    image:binary image
    centroid_starter:centroid starting location sliding window
    sliding_window_specs:['width','n_steps']
        width of sliding window
        number of steps of sliding window alog vertical axis
    return {'x':[],'y':[]}
        coordiantes of all hotpixels detected by sliding window
        coordinates of alll centroids recorded but not used yet!        
    '''
    #Initialize sliding window
    window = {'x0':centroid_starter-int(sliding_window_specs['width']/2),
              'y0':image.shape[0],'width':sliding_window_specs['width'],
              'height':round(image.shape[0]/sliding_window_specs['n_steps'])}
    hotpixels_log = {'x':[],'y':[]}
    centroids_log = []
    if showMe:
        out_img = (np.dstack((image,image,image))*255).astype('uint8')
    for step in range(sliding_window_specs['n_steps']):
        if window['x0']<0: window['x0'] = 0
        if (window['x0']+sliding_window_specs['width'])>image.shape[1]:
            window['x0'] = image.shape[1] - sliding_window_specs['width']
        centroid,peak_intensity,hotpixels_cnt = find_centroid(image,peak_thresh,window,showMe=0)
        if step==0:
            starter_centroid = centroid
        if hotpixels_cnt/(window['width']*window['height'])>0.6:
            window['width'] = window['width']*2
            window['x0']  = round(window['x0']-window['width']/2)
            if (window['x0']<0):window['x0']=0
            if (window['x0']+window['width'])>image.shape[1]:
                window['x0'] = image.shape[1]-window['width']
            centroid,peak_intensity,hotpixels_cnt = find_centroid(image,peak_thresh,window,showMe=0)          
            
        #if showMe:
            #print('peak intensity{}'.format(peak_intensity))
            #print('This is centroid:{}'.format(centroid))
        mask_window = np.zeros_like(image)
        mask_window[window['y0']-window['height']:window['y0'],
                    window['x0']:window['x0']+window['width']]\
            = image[window['y0']-window['height']:window['y0'],
                window['x0']:window['x0']+window['width']]
        
        hotpixels = np.nonzero(mask_window)
        #print(hotpixels_log['x'])
        
        hotpixels_log['x'].extend(hotpixels[0].tolist())
        hotpixels_log['y'].extend(hotpixels[1].tolist())
        # update record of centroid
        centroids_log.append(centroid)
        
        if showMe:
            cv2.rectangle(out_img,
                            (window['x0'],window['y0']-window['height']),
                            (window['x0']+window['width'],window['y0']),(0,255,0),2)
            
            if int(window['y0'])==68:
                plt.imshow(out_img)
                plt.show()
            '''
            print(window['y0'])
            plt.imshow(out_img)
            '''
        # set next position of window and use standard sliding window width
        window['width'] = sliding_window_specs['width']
        window['x0'] = round(centroid-window['width']/2)
        window['y0'] = window['y0'] - window['height']
    return hotpixels_log
sliding_window_specs = {'width':120,'n_steps':10}
log_lineLeft = run_sliding_window(warped_image,centroid_starter_left['centroid'],sliding_window_specs,showMe=showMe)
log_lineRight = run_sliding_window(warped_image,centroid_starter_right['centroid'],sliding_window_specs,showMe=showMe)
```


![png](https://img-blog.csdn.net/20180430214735401?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![png](https://img-blog.csdn.net/2018043021474351?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


接下来我们需要利用[马氏距离](https://en.wikipedia.org/wiki/Mahalanobis_distance)来去除一些二变量的异常值
$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x-\mu)}$$


```python
def MahalanobisDist(x,y):
    '''
    Mahalanobis Distance for bi-variate distribution
    
    '''
    covariance_xy = np.cov(x,y,rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i-xy_mean[0] for x_i in x])
    y_diff = np.array([y_i-xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff,y_diff])
    
    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
    return md
    
def MD_removeOutliers(x,y,MD_thresh):
    '''
    remove pixels outliers using Mahalonobis distance
    '''
    MD = MahalanobisDist(x,y)
    threshold = np.mean(MD)*MD_thresh
    nx,ny,outliers = [],[],[]
    for i in range(len(MD)):
        if MD[i]<=threshold:
            nx.append(x[i])
            ny.append(y[i])
        else:
            outliers.append(i)
    return (nx,ny)
MD_thresh = 1.8
log_lineLeft['x'],log_lineLeft['y'] = \
MD_removeOutliers(log_lineLeft['x'],log_lineLeft['y'],MD_thresh)
log_lineRight['x'],log_lineRight['y'] = \
MD_removeOutliers(log_lineRight['x'],log_lineRight['y'],MD_thresh)
```

接下来我们用hotpixels来拟合二次多项式，这里我们采用当前
帧的hotpixels来拟合曲线。如果想更好地拟合直线，我们可以利用过去连续5帧地图像


```python
def update_tracker(tracker,new_value):
    '''
    update tracker(self.bestfit or self.bestfit_real or radO Curv or hotpixels) with new coeffs
    new_coeffs is of the form {'a2':[val2,...],'a1':[va'1,...],'a0':[val0,...]}
    tracker is of the form {'a2':[val2,...]}
    update tracker of radius of curvature
    update allx and ally with hotpixels coordinates from last sliding window
    '''
    if tracker =='bestfit':
        bestfit['a0'].append(new_value['a0'])
        bestfit['a1'].append(new_value['a1'])
        bestfit['a2'].append(new_value['a2'])
    elif tracker == 'bestfit_real':
        bestfit_real['a0'].append(new_value['a0'])
        bestfit_real['a1'].append(new_value['a1'])
        bestfit_real['a2'].append(new_value['a2'])
    elif tracker == 'radOfCurvature':
        radOfCurv_tracker.append(new_value)
    elif tracker == 'hotpixels':
        allx.append(new_value['x'])
        ally.append(new_value['y'])
buffer_sz = 5
allx = deque([],maxlen=buffer_sz)
ally = deque([],maxlen=buffer_sz)        
bestfit = {'a0':deque([],maxlen=buffer_sz),
                       'a1':deque([],maxlen = buffer_sz),
                       'a2':deque([],maxlen=buffer_sz)}
bestfit_real = {'a0':deque([],maxlen=buffer_sz),
                            'a1':deque([],maxlen=buffer_sz),
                            'a2':deque([],maxlen=buffer_sz)}
radOfCurv_tracker = deque([],maxlen=buffer_sz)


update_tracker('hotpixels',log_lineRight)
update_tracker('hotpixels',log_lineLeft)
multiframe_r = {'x':[val for sublist in allx for val in sublist],
                   'y':[val for sublist in ally for val in sublist]}
multiframe_l = {'x':[val for sublist in allx for val in sublist],
                   'y':[val for sublist in ally for val in sublist]}
#fit to polynomial in pixel space
def polynomial_fit(data):
    '''
    多项式拟合
    a0+a1 x+a2 x**2
    data:dictionary with x and y values{'x':[],'y':[]}
    '''
    a2,a1,a0 = np.polyfit(data['x'],data['y'],2)
    return {'a0':a0,'a1':a1,'a2':a2}
#merters per pixel in y or x dimension
ym_per_pix = 12/450
xm_per_pix = 3.7/911
fit_lineLeft = polynomial_fit(multiframe_l)
fit_lineLeft_real = polynomial_fit({'x':[i*ym_per_pix for i in multiframe_l['x']],
                                    'y':[i*xm_per_pix for i in multiframe_l['y']]})
fit_lineRight = polynomial_fit(multiframe_r)
fit_lineRight_real = polynomial_fit({'x':[i*ym_per_pix for i in multiframe_r['x']],
                                               'y':[i*xm_per_pix for i in multiframe_r['y']]})

```


```python
def predict_line(x0,xmax,coeffs):
    '''
    predict road line using polyfit cofficient
    x vaues are in range (x0,xmax)
    polyfit coeffs:{'a2':,'a1':,'a2':}
    returns array of [x,y] predicted points ,x along image vertical / y along image horizontal direction
    '''
    x_pts = np.linspace(x0,xmax-1,num=xmax)
    pred = coeffs['a2']*x_pts**2+coeffs['a1']*x_pts+coeffs['a0']
    return np.column_stack((x_pts,pred))
fit_lineRight_singleframe = polynomial_fit(log_lineRight)
fit_lineLeft_singleframe = polynomial_fit(log_lineLeft)
var_pts = np.linspace(0,corr_img.shape[0]-1,num=corr_img.shape[0])
pred_lineLeft_singleframe = predict_line(0,corr_img.shape[0],fit_lineLeft_singleframe)
pred_lineRight_sigleframe = predict_line(0,corr_img.shape[0],fit_lineRight_singleframe)
plt.plot(pred_lineLeft_singleframe[:,1],pred_lineLeft_singleframe[:,0],'b-',label='singleframe',linewidth=1)
plt.plot(pred_lineRight_sigleframe[:,1],pred_lineRight_sigleframe[:,0],'b-',label='singleframe',linewidth=1)
plt.show()
```


![png](https://img-blog.csdn.net/20180430214756461?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


## 曲率及车辆位置估计
这里我们通过下面的公式来计算车道的[曲率](https://www.intmath.com/applications-differentiation/8-radius-curvature.php)
$$ R = \frac{(1 + (2 a_2 y + a_1)^2 )^{3/2}}{|2 a_2|}$$
多项式满足以下形式：$$a_2 y^2 + a_1 y + a_0$$


```python
def compute_radOfCurvature(coeffs,pt):
    return ((1+(2*coeffs['a2']*pt+coeffs['a1'])**2)**1.5)/np.absolute(2*coeffs['a2'])
pt_curvature = corr_img.shape[0]
radOfCurv_r = compute_radOfCurvature(fit_lineRight_real,pt_curvature*ym_per_pix)
radOfCurv_l = compute_radOfCurvature(fit_lineLeft_real,pt_curvature*ym_per_pix)
average_radCurv = (radOfCurv_r+radOfCurv_l)/2

center_of_lane = (pred_lineLeft_singleframe[:,1][-1]+pred_lineRight_sigleframe[:,1][-1])/2
offset = (corr_img.shape[1]/2 - center_of_lane)*xm_per_pix

side_pos = 'right'
if offset <0:
    side_pos = 'left'
wrap_zero = np.zeros_like(gray_ex).astype(np.uint8)
color_wrap = np.dstack((wrap_zero,wrap_zero,wrap_zero))
left_fitx = fit_lineLeft_singleframe['a2']*var_pts**2 + fit_lineLeft_singleframe['a1']*var_pts + fit_lineLeft_singleframe['a0']
right_fitx = fit_lineRight_singleframe['a2']*var_pts**2 + fit_lineRight_singleframe['a1']*var_pts+fit_lineRight_singleframe['a0']
pts_left = np.array([np.transpose(np.vstack([left_fitx,var_pts]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,var_pts])))])    
pts = np.hstack((pts_left,pts_right))
cv2.fillPoly(color_wrap,np.int_([pts]),(0,255,0))
cv2.putText(color_wrap,'|',(int(corr_img.shape[1]/2),corr_img.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),8)
cv2.putText(color_wrap,'|',(int(center_of_lane),corr_img.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),8)  
newwrap = cv2.warpPerspective(color_wrap,transform_matrix['Minv'],(corr_img.shape[1],corr_img.shape[0]))
result = cv2.addWeighted(corr_img,1,newwrap,0.3,0) 
cv2.putText(result,'Vehicle is' + str(round(offset,3))+'m'+side_pos+'of center',
            (50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),thickness=2)
cv2.putText(result,'Radius of curvature:'+str(round(average_radCurv))+'m',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),thickness=2) 
if showMe:
    plt.title('Final Result')
    plt.imshow(result)
    plt.axis('off')
    plt.show()
```


![png](https://img-blog.csdn.net/20180430214807546?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA2NjUyMTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



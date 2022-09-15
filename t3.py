from pylab import *
from numpy import *
from PIL import Image

# If you have PCV installed, these imports should work
from PCV.geometry import homography, warp
from PCV.localdescriptors import sift

"""
This is the panorama example from section 3.3.
"""

# set paths to data folder
# imname使我们要拼接的原图
# featname是sift文件，这个文件是需要根据原图进行生成的
# 需要根据自己的图像地址和图像数量修改地址和循环次数
# featname = ['./images5/'+str(i+1)+'.sift' for i in range(2)] 
# imname = ['./images5/'+str(i+1)+'.jpg' for i in range(2)]
featname = ['./test/1027-'+str(i+1)+'.sift' for i in range(2)] 
imname = ['./test/1027-'+str(i+1)+'.jpg' for i in range(2)]

# extract features and match
l = {}
d = {}
for i in range(2): 
    sift.process_image(imname[i],featname[i])
    l[i],d[i] = sift.read_features_from_file(featname[i])

matches = {}
for i in range(1):
    matches[i] = sift.match(d[i+1],d[i])

# visualize the matches (Figure 3-11 in the book)
for i in range(1):
    im1 = array(Image.open(imname[i]))
    im2 = array(Image.open(imname[i+1]))
    figure()
    sift.plot_matches(im2,im1,l[i+1],l[i],matches[i],show_below=True)


# function to convert the matches to hom. points
def convert_points(j):
    ndx = matches[j].nonzero()[0]
    fp = homography.make_homog(l[j+1][ndx,:2].T) 
    ndx2 = [int(matches[j][i]) for i in ndx]
    tp = homography.make_homog(l[j][ndx2,:2].T) 
    
    # switch x and y - TODO this should move elsewhere
    fp = vstack([fp[1],fp[0],fp[2]])
    tp = vstack([tp[1],tp[0],tp[2]])
    return fp,tp


# estimate the homographies
model = homography.RansacModel() 
# 此代码段为2图图像拼接，若需要多幅图，只需将其中的注释部分取消即可，图像顺序为自右向左。
fp,tp = convert_points(0)
H_01 = homography.H_from_ransac(fp,tp,model)[0] #im 0 to 1

#fp,tp = convert_points(1)
#H_12 = homography.H_from_ransac(fp,tp,model)[0] #im 1 to 2 

#tp,fp = convert_points(2) #NB: reverse order
#H_32 = homography.H_from_ransac(fp,tp,model)[0] #im 3 to 2 

#tp,fp = convert_points(3) #NB: reverse order
#H_43 = homography.H_from_ransac(fp,tp,model)[0] #im 4 to 3    


# warp the images
delta = 2000 # for padding and translation

im1 = array(Image.open(imname[0]), "uint8")
im2 = array(Image.open(imname[1]), "uint8")
im_12 = warp.panorama(H_01,im1,im2,delta,delta)
#im1 = array(Image.open(imname[0]), "f")
#im_02 = warp.panorama(dot(H_12,H_01),im1,im_12,delta,delta)

#im1 = array(Image.open(imname[3]), "f")
#im_32 = warp.panorama(H_32,im1,im_02,delta,delta)

#im1 = array(Image.open(imname[4]), "f")
#im_42 = warp.panorama(dot(H_32,H_43),im1,im_32,delta,2*delta)


figure()
imshow(array(im_12, "uint8"))
axis('off')
savefig("example1.png",dpi=300)
show()


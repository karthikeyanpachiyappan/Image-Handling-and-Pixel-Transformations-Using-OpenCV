# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
## Name: KARTHIKEYAN P
## Register Number: 212223230102

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
img=cv2.imread('Eagle_in_Flight.jpg',cv2.IMREAD_GRAYSCALE)
```

#### 2. Print the image width, height & Channel.
```python
img.shape
```

#### 3. Display the image using matplotlib imshow().
```python
plt.imshow(img,cmap='gray')
plt.title('GRAYSCALE IMAGE')
plt.axis('on')
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite('Eagle_in_Flight.png',img)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
img=cv2.imread('Eagle_in_Flight.png',1)
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(img_rgb)
print(img_rgb.shape)
plt.title('Color Image from saved PNG image')
plt.show()
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
crop_img=img[20:410,200:545]
print(crop_img.shape)
plt.imshow(crop_img)
plt.title('EAGLE')
plt.show()
```

#### 8. Resize the image up by a factor of 2x.
```python
resize_crop_img=cv2.resize(crop_img,None,fx=2,fy=2)
plt.imshow(resize_crop_img)
resize_crop_img.shape
```

#### 9. Flip the cropped/resized image horizontally.
```python
flip_img=cv2.flip(resize_crop_img,1)
plt.imshow(flip_img)
plt.show()
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
appollo=cv2.imread('Apollo-11-launch.jpg')
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
appollo_text=appollo.copy()
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
color=(255,255,255)
(text_width, text_height), baseline = cv2.getTextSize(text, font_face, 1, 2)
x=(img.shape[1]-text_width)//2
y=img.shape[0] + 80
cv2.putText(appollo_text,text,(x,y),font_face,3,color,2,cv2.LINE_AA)
plt.imshow(appollo_text)
plt.show()
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rect_color = (255,0,255)
appollo_rect=cv2.rectangle(appollo,(500,100),(700,630),rect_color,2)
```

#### 13. Display the final annotated image.
```python
plt.imshow(appollo_rect)
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```python
boy=cv2.imread('boy.jpg',cv2.IMREAD_COLOR)
boy_rgb = cv2.cvtColor(boy, cv2.COLOR_BGR2RGB)
```

#### 15. Adjust the brightness of the image.
```python
brightness_value=100
matrix = np.ones(boy_rgb.shape, dtype=np.uint8) * abs(brightness_value)
```

#### 16. Create brighter and darker images.
```python
boy_bright=cv2.add(boy_rgb,matrix)
boy_dark=cv2.subtract(boy_rgb,matrix)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
fig,axes=plt.subplots(1,3,figsize=(12,4))
axes[0].imshow(boy_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(boy_dark)
axes[1].set_title("Darker Image")
axes[1].axis("off")

axes[2].imshow(boy_bright)
axes[2].set_title("Brighter Image")
axes[2].axis("off")
plt.show()
```

#### 18. Modify the image contrast.
```python
# Create two higher contrast images using the 'scale' option with factors of 1.1 and 1.2 (without overflow fix)
matrix1 = 1.1
matrix2 = 1.2
boy_higher1 = cv2.convertScaleAbs(boy_rgb, alpha=matrix1, beta=0)
boy_higher2 = cv2.convertScaleAbs(boy_rgb, alpha=matrix2, beta=0)
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
fig,axes=plt.subplots(1,3,figsize=(12,4))
axes[0].imshow(boy_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(boy_higher1)
axes[1].set_title("With 1.1 contrast")
axes[1].axis("off")

axes[2].imshow(boy_higher2)
axes[2].set_title("With 1.2 contrast")
axes[2].axis("off")
plt.show()
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
b,g,r=cv2.split(boy_rgb)
fig,axes=plt.subplots(1,3,figsize=(12,4))
axes[0].imshow(b)
axes[0].set_title("B channel")
axes[0].axis("off")

axes[1].imshow(g)
axes[1].set_title("G channel")
axes[1].axis("off")

axes[2].imshow(r)
axes[2].set_title("R channel")
axes[2].axis("off")
plt.show()
```

#### 21. Merged the R, G, B , displays along with the original image
```python
merge_boy=cv2.merge((b,g,r))
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].imshow(boy_rgb)
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(merge_boy)
axes[1].set_title("Merged RGB image")
axes[1].axis("off")
plt.show()
```

#### 22. Split the image into the H, S, V components & Display the channels.
```python
hsv = cv2.cvtColor(boy_rgb, cv2.COLOR_RGB2HSV)
h,s,v=cv2.split(hsv)
fig,axes=plt.subplots(1,3,figsize=(12,4))
axes[0].imshow(h)
axes[0].set_title("H channel")
axes[0].axis("off")

axes[1].imshow(s)
axes[1].set_title("S channel")
axes[1].axis("off")

axes[2].imshow(v)
axes[2].set_title("V channel")
axes[2].axis("off")
plt.show()
```
#### 23. Merged the H, S, V, displays along with original image.
```python
merge_hsv=cv2.merge((h,s,v))
rgb_merged = cv2.cvtColor(merge_hsv, cv2.COLOR_HSV2RGB)
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].imshow(boy_rgb)
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(rgb_merged)
axes[1].set_title("Merged RGB image")
axes[1].axis("off")
plt.show()
```

## Output:
- **i)** Read and Display an Image.  
- **ii)** Adjust Image Brightness.  
- **iii)** Modify Image Contrast.  
- **iv)** Generate Third Image Using Bitwise Operations.

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.


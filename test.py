#%%
import cv2
image = cv2.imread("maJM7QoN7-2935.532_2.jpg")

#%%
print(image.shape)
# %%

with open("maJM7QoN7-2935.532_2.txt", "r") as f:
    annotation = f.readlines()

# %%
annotation = [x.strip().split(" ") for x in annotation]

# %%
for anno in annotation:
    category, x, y, w, h = anno
    x, y, w, h = float(x), float(y), float(w), float(h)
    xcenter, ycenter = int(x * image.shape[1]), int(y * image.shape[0])
    w = w * image.shape[1]
    h = h * image.shape[0]
    xmin, ymin = int(xcenter - w / 2), int(ycenter - h / 2)
    xmax, ymax = int(xcenter + w / 2), int(ycenter + h / 2)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

# %%
# save image
cv2.imwrite("test.jpg", image)

# %%

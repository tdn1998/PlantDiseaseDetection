from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 128
base_dir = "D:\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)"

training_set = train_datagen.flow_from_directory(base_dir+'/train',
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

valid_set = valid_datagen.flow_from_directory(base_dir+'/valid',
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical')

class_dict = training_set.class_indices
print(class_dict)

li = list(class_dict.keys())
print(li)

from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

image_path = "D:\\test\\test\\TomatoYellowCurlVirus1.JPG"
new_img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
img = img / 255

model = load_model("D:\models\AlexNetModel.hdf")

print("Following is our prediction:")

prediction = model.predict(img)

d = prediction.flatten()
j = d.max()
for index, item in enumerate(d):
    if item == j:
        class_name = li[index]

plt.figure(figsize=(4, 4))
plt.imshow(new_img)
plt.axis('off')
plt.title(class_name)
plt.show()
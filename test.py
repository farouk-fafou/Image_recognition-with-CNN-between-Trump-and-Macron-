#  - Making predictions

# load the model we saved
from keras.models import load_model
from keras.preprocessing import image
import sys 
import numpy as np

classifier = load_model ('model.h5')

file_path = sys.argv[1]


test_image = image.load_img(file_path , target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image / 255
result = classifier.predict(test_image)


print (result)
if result[0][0] > .5:
    prediction = 'trump'
    print ("we caught You Mister Donald Trump !!!")
else:
    prediction = 'macron'
    print ("we caught You Mister Macron      !!!")
    
from Text2Text.Text2Text import Text2TextModel
from Image2Text.Image2Text import Image2TextModel

batch_size = 100  # Batch size for training.
epochs = 100  # Number of epochs to train for.
# samples = 10000
data_path = 'datasets/mnist-5000/targets.txt'
validation_split = 0.2

model = Image2TextModel()

model.load_data(data_path)
model.train(batch_size, epochs, validation_split)
# model.save('s2s.h5')
# prediction = model.predict(model.encoder_input_data[0:1])
# print(prediction)

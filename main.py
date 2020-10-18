from BinarySeq2Seq import BinarySeq2SeqModel

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
samples = 10000
data_path = 'fra.txt'
validation_split = 0.2

model = BinarySeq2SeqModel()

model.load_data(data_path, samples)
model.train(batch_size, epochs, validation_split)
model.save('s2s.h5')
prediction = model.predict(model.encoder_input_data[0:1])
print(prediction)

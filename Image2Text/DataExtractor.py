import numpy as np
import PIL.Image as Image
import os
import io


def isChar(c):
    return isinstance(c, str) and len(c) == 1


def BinaryArrayToChar(a):
    a = [str(int(bit)) for bit in np.round(a)]
    binary = "".join(a)
    ascii_num = int(binary, 2)
    return chr(ascii_num)


def CharToBinaryArray(c):
    binary = bin(bytes(c, 'utf-8')[0])[2:]
    binary = (8 - len(binary)) * '0' + binary
    return [float(bit) for bit in binary]


def BytesToBinaryArray(bts):
    arr = []
    for byte in bts:
        byte = ord(byte) if isChar(byte) else byte
        binary = bin(byte)[2:]
        binary = (8 - len(binary)) * '0' + binary
        arr.append([float(bit) for bit in binary])
    return arr


def ZeroPaddedBytes(bts, max_seq_length):
    padded = np.zeros(shape=(len(bts), max_seq_length, 8), dtype="float32")
    for i, byte_seq in enumerate(bts):
        bin_array = BytesToBinaryArray(byte_seq)
        padded[i, :len(byte_seq)] = bin_array
    return padded


def VectorizeImage2TextData(data_path, num_samples=5000):

    input_data = []
    target_input_data = []
    target_output_data = []

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    max_encoder_seq_length = 0
    max_decoder_seq_length = 0

    for line in lines[: min(num_samples, len(lines) - 1)]:

        filename, target_text = line.split('\t')
        path = os.path.split(data_path)[0] + "/" + filename

        with open(path, "rb") as f:
            data = f.read()
            target_input = '\t' + target_text + '\n'
            target_output = target_text + '\n'

        input_data.append(data)
        target_input_data.append(target_input)
        target_output_data.append(target_output)

        max_encoder_seq_length = max(max_encoder_seq_length, len(data))
        max_decoder_seq_length = max(max_decoder_seq_length, len(target_input.encode()))

    inputs = ZeroPaddedBytes(input_data, max_encoder_seq_length)
    target_input = ZeroPaddedBytes(target_input_data, max_decoder_seq_length)
    target_output = ZeroPaddedBytes(target_output_data, max_decoder_seq_length)

    return inputs, target_input, target_output, max_encoder_seq_length, max_decoder_seq_length

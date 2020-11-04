import numpy as np
import os


def CharToBinaryArray(c):
    binary = bin(bytes(c, 'utf-8')[0])[2:]
    binary = (8 - len(binary)) * '0' + binary
    return [float(bit) for bit in binary]


def StringToBinaryArray(s):
    arr = []
    for byte in bytes(s, 'utf-8'):
        binary = bin(byte)[2:]
        binary = (8 - len(binary)) * '0' + binary
        arr.append([float(bit) for bit in binary])
    return arr


def BytesToBinaryArray(bts):
    arr = []
    for byte in bts:
        binary = bin(byte)[2:]
        binary = (8 - len(binary)) * '0' + binary
        arr.append([float(bit) for bit in binary])
    return arr


def BinaryArrayToChar(a):
    a = [str(int(bit)) for bit in np.round(a)]
    binary = "".join(a)
    ascii_num = int(binary, 2)
    return chr(ascii_num)


def ByteSecuenceToString(s):
    string = ""
    for byte in s:
        string += BinaryArrayToChar(byte)
    return string


def TextsToBinary(texts):
    return [StringToBinaryArray(text) for text in texts]


def ZeroPaddedData(texts, max_seq_length):
    padded = np.zeros(shape=(len(texts), max_seq_length, 8), dtype="float32")
    padded[:, :, 2] = 1.0
    for i, text in enumerate(texts):
        bin_array = StringToBinaryArray(text)
        padded[i, :len(text.encode())] = bin_array
    return padded


def VectorizeText2TextData(data_path, num_samples=10000):

    input_texts = []
    target_input_texts = []
    target_output_texts = []

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    max_encoder_seq_length = 0
    max_decoder_seq_length = 0

    for line in lines[: min(num_samples, len(lines) - 1)]:

        input_text, target_text, _ = line.split('\t')
        target_input_text = '\t' + target_text + '\n'
        target_output_text = target_text + '\n'

        input_texts.append(input_text)
        target_input_texts.append(target_input_text)
        target_output_texts.append(target_output_text)

        max_encoder_seq_length = max(max_encoder_seq_length, len(input_text.encode()))
        max_decoder_seq_length = max(max_decoder_seq_length, len(target_input_text.encode()))

    input_texts = ZeroPaddedData(input_texts, max_encoder_seq_length)
    target_input_texts = ZeroPaddedData(target_input_texts, max_decoder_seq_length)
    target_output_texts = ZeroPaddedData(target_output_texts, max_decoder_seq_length)

    return input_texts, target_input_texts, target_output_texts, max_encoder_seq_length, max_decoder_seq_length


def VectorizeImage2TextData(data_path, num_samples=5000):

    input_data = []
    target_input_data = []
    target_output_data = []

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    max_encoder_seq_length = 0
    max_decoder_seq_length = 0

    for line in lines[: min(num_samples, len(lines) - 1)]:

        filename, target_text, _ = line.split('\t')

        with open(os.path.split(data_path)[0] + "/" + filename, "rb") as f:
            data = f.read()
            target_input = '\t' + target_text + '\n'
            target_output = target_text + '\n'

        input_data.append(data)
        target_input_data.append(target_input)
        target_output_data.append(target_output)

        max_encoder_seq_length = max(max_encoder_seq_length, len(data))
        max_decoder_seq_length = max(max_decoder_seq_length, len(target_input.encode()))

    input = ZeroPaddedBytes(input_data, max_encoder_seq_length)
    target_input = ZeroPaddedBytes(target_input_data, max_decoder_seq_length)
    target_output = ZeroPaddedBytes(target_output_data, max_decoder_seq_length)

    return input, target_input, target_output, max_encoder_seq_length, max_decoder_seq_length


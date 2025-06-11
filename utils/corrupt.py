import math
import random

def corrupt_message(message, num_bits_to_flip):
    corrupted_message = message.copy()
    message_len = len(message)
    for iteration in range(num_bits_to_flip):
        corrupted_index = random.randint(0,message_len) - 1
        corrupted_message[corrupted_index] = abs(1 - message[corrupted_index])
    return corrupted_message


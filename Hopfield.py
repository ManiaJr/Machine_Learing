import asyncio
from PIL import Image
import numpy as np
import os

def dim_training_data():
    for digit_idx, digit in enumerate(training_data):
        for i_idx, i in enumerate(digit):
            for j_idx, j in enumerate(i):
                # Diastaseis apo to tetragwno
                x = j_idx * square_size
                y = i_idx * square_size

                # Load the image
                if digit_idx==0:
                    image = Image.open('arithmoi/pente.png')
                elif digit_idx==1:
                    image = Image.open('arithmoi/eksi.png')
                elif digit_idx==2:
                    image = Image.open('arithmoi/oktw.png')
                else:
                    image = Image.open('arithmoi/ennia.png')
                #"Croparw" to tetragwno apo thn eikona
                square = image.crop((x, y, x + square_size, y + square_size))

                # to kanw array
                square_data = np.asarray(square)

                # Determine the value for the square
                if np.mean(square_data) > 127:  # An megalytero toy 127, einai leyko
                    training_data[digit_idx, i_idx, j_idx] = -1
                else:  # alliws mayro
                    training_data[digit_idx, i_idx, j_idx] = 1

def dim_test_digit():
    for digit_idx, digit in enumerate(test_digit):
        for i_idx, i in enumerate(digit):
            for j_idx, j in enumerate(i):
                # Diastaseis apo to tetragwno
                x = j_idx * square_size
                y = i_idx * square_size

                # Load the image
                if digit_idx==0:
                    image = Image.open('arithmoi2/pente2.png')
                elif digit_idx==1:
                    image = Image.open('arithmoi2/eksi2.png')
                elif digit_idx==2:
                    image = Image.open('arithmoi2/oktw2.png')
                else:
                    image = Image.open('arithmoi2/ennia2.png')
                #"Croparw" to tetragwno apo thn eikona
                square = image.crop((x, y, x + square_size, y + square_size))

                #to kanw array
                square_data = np.asarray(square)

                if np.mean(square_data) > 127:  # An megalytero toy 127, einai leyko
                    test_digit[digit_idx, i_idx, j_idx] = -1
                else:  # alliws mayro
                    test_digit[digit_idx, i_idx, j_idx] = 1

def writeAndsave(arraytosave):
    f = open('test.txt', 'a', encoding='utf-8')  # Specify UTF-8 encoding

    for digit in arraytosave:
        f.write('Training data:\n')
        for row in digit:
            for element in row:
                if element == 1:
                    f.write("█")
                else:
                    f.write(" ")
            f.write('\n')
    f.write('\n')
    f.close()

def recognize_digit(test_data):
    state = np.array(test_data)
    previous_state = None
    while previous_state is None or not np.array_equal(state, previous_state):
        previous_state = np.copy(state)
        state = np.sign(np.dot(state.flatten(), weights.reshape(num_pixels, num_pixels).T))

    return state.reshape((len(test_data), len(test_data[0])))


def print_recognized_digit(recognized_digit):
    print("Recognized digit:")
    for row in recognized_digit:
        for element in row:
            if element == 1:
                print("█", end=" ")  # print "█" for 1
            else:
                print(" ", end=" ")  # print empty space for -1
        print()  # print a new line after each row


async def recognize_asynchronous(test_data):
    recognized_digit = await asyncio.get_running_loop().run_in_executor(None, recognize_digit, test_data)
    print_recognized_digit(recognized_digit)


def recognize_synchronous(test_data):
    recognized_digit = recognize_digit(test_data)
    print_recognized_digit(recognized_digit)

# Load the image to be training data
image = Image.open('arithmoi/pente.png')
image = image.convert('L')
width, height = image.size
square_size = min(width, height) // 7
training_data = np.zeros((4, 11, 7))
dim_training_data()

#make test digits
test_digit=np.zeros((4, 11, 7))
dim_test_digit()

# Delete an yparxei o test.txt
file_path = "test.txt"
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"The file '{file_path}' has been deleted.")
else:
    print(f"The file '{file_path}' does not exist.")

#Save arrays of training and test 
writeAndsave(training_data)
writeAndsave(test_digit)

num_pixels = len(training_data[0]) * len(training_data[0][0])
weights = np.zeros((num_pixels, num_pixels))

for data in training_data:
    input_data = np.array(data)
    flattened_data = input_data.flatten()
    weights += np.outer(flattened_data, flattened_data)

print("Choose an altered number to recognize: 5, 6, 8, 9")
chosen_number = int(input("Enter the number: "))

if chosen_number in [5, 6, 8, 9]:
    if chosen_number<7:
        chosen_index = chosen_number - 5
    else:
        chosen_index = chosen_number - 6

    print("Choose the execution method:")
    print("1. Asynchronous")
    print("2. Synchronous")
    chosen_method = int(input("Enter the method number: "))

    if chosen_method == 1:
        asyncio.run(recognize_asynchronous(test_digit[chosen_index]))
    elif chosen_method == 2:
        recognize_synchronous(test_digit[chosen_index])
    else:
        print("Invalid method chosen. Please select 1 for Asynchronous or 2 for Synchronous.")
else:
    print("Invalid number chosen. Please select 5, 6, 8, or 9.")

print("Ta apotelesmata exoyn apothikeytei sto txt arxei me onoma test ston idio fakele poy tre3ate to programma")
import numpy as np
import math
from numpy import matrix
from numpy import linalg

message = input('Enter message: (<36 chars):')
alphabet = 'abcdefghijklmnopqrstuvwxyz '
plaintext = []
cyphertext = []
holding = ''
engwish = ''
raw = []
order = input('Key order: (2/3/4/6)')
reversex = order[::-1]

if len(message) > 36:
    print('message too long')
    quit()

def matrix_multiplication(A, B):
    res = np.zeros((A.shape[0], B.shape[1]))

    for _x in range(A.shape[0]):
        for _y in range(B.shape[1]):
            res[_x, _y] = np.sum(A[_x, :] * B[:, _y])
    return res

def engwishate(cytext):
    global engwish
    global raw
    engwish = ''
    for number in cytext:
        engwish += (alphabet[number % 27])
        raw.append(number % 27)
    #print(engwish)
    #print('Raw numbers: ',raw)

def engwishate2(cytext):
    global engwish
    engwish = ''
    for number in cytext:
        engwish += (alphabet[number % 27])
    print('English translation: ',engwish)

def modMatInv(A,p):       # Finds the inverse of matrix A mod p
  n=len(A)
  A=matrix(A)
  adj=np.zeros(shape=(n,n))
  for i in range(0,n):
    for j in range(0,n):
      adj[i][j]=((-1)**(i+j)*int(round(linalg.det(minor(A,j,i)))))%p
  return (modInv(int(round(linalg.det(A))),p)*adj)%p

def modInv(a,p):          # Finds the inverse of a mod p, if it exists
  try:
      for i in range(1,p):
        if (i*a)%p==1:
          return i
  except ValueError: print(str(a)+" has no inverse mod "+str(p))

def minor(A,i,j):    # Return matrix A with the ith row and jth column deleted
  A=np.array(A)
  minor=np.zeros(shape=(len(A)-1,len(A)-1))
  p=0
  for s in range(0,len(minor)):
    if p==i:
      p=p+1
    q=0
    for t in range(0,len(minor)):
      if q==j:
        q=q+1
      minor[s][t]=A[p][q]
      q=q+1
    p=p+1
  return minor

base_two = np.zeros((2, 2))
base_three = np.zeros((3, 3))
base_four = np.zeros((4, 4))
base_six = np.zeros((6, 6))
base = np.zeros((7, 7))
two_key = np.array([[2, 5],
           [2, 3]])
three_key = np.array([[1, 4, 1],
             [7, 1, 6],
             [1, 2, 1]])
four_key = np.array([[2, 56, 4, 3],
            [46, 5, 7, 95],
            [73, 24, 5, 88],
            [45, 66, 22, 71]])
six_key = np.array(
    [[8, 3, 4, 6, 1, 5],
     [2, 4, 6, 8, 3, 4],
     [7, 3, 16, 5, 4, 9],
     [12, 4, 3, 4, 6, 2],
     [56, 48, 2, 46, 5, 7],
     [4, 2, 8, 2, 45, 3]])



while len(message) < 36:
    message += ' '

for letter in range(0, len(message)):
    plaintext.append(alphabet.index(message[letter]))
print(plaintext)

def two(list):
    two_count = 0
    two_1 = []
    two_2 = []
    two_3 = []
    two_4 = []
    two_5 = []
    two_6 = []
    two_7 = []
    two_8 = []
    two_9 = []
    two_10 = []
    two_11 = []
    two_12 = []
    two_13 = []
    two_14 = []
    two_15 = []
    two_16 = []
    two_17 = []
    two_18 = []
    while two_count < 2:
        two_1.append(list[two_count])
        two_count += 1
    while two_count < 4:
        two_2.append(list[two_count])
        two_count += 1
    while two_count < 6:
        two_3.append(list[two_count])
        two_count += 1
    while two_count < 8:
        two_4.append(list[two_count])
        two_count += 1
    while two_count < 10:
        two_5.append(list[two_count])
        two_count += 1
    while two_count < 12:
        two_6.append(list[two_count])
        two_count += 1
    while two_count < 14:
        two_7.append(list[two_count])
        two_count += 1
    while two_count < 16:
        two_8.append(list[two_count])
        two_count += 1
    while two_count < 18:
        two_9.append(list[two_count])
        two_count += 1
    while two_count < 20:
        two_10.append(list[two_count])
        two_count += 1
    while two_count < 22:
        two_11.append(list[two_count])
        two_count += 1
    while two_count < 24:
        two_12.append(list[two_count])
        two_count += 1
    while two_count < 26:
        two_13.append(list[two_count])
        two_count += 1
    while two_count < 28:
        two_14.append(list[two_count])
        two_count += 1
    while two_count < 30:
        two_15.append(list[two_count])
        two_count += 1
    while two_count < 32:
        two_16.append(list[two_count])
        two_count += 1
    while two_count < 34:
        two_17.append(list[two_count])
        two_count += 1
    while two_count < 36:
        two_18.append(list[two_count])
        two_count += 1

    new_two_1 = matrix_multiplication(two_key, np.c_[two_1]).tolist()
    new_two_2 = matrix_multiplication(two_key, np.c_[two_2]).tolist()
    new_two_3 = matrix_multiplication(two_key, np.c_[two_3]).tolist()
    new_two_4 = matrix_multiplication(two_key, np.c_[two_4]).tolist()
    new_two_5 = matrix_multiplication(two_key, np.c_[two_5]).tolist()
    new_two_6 = matrix_multiplication(two_key, np.c_[two_6]).tolist()
    new_two_7 = matrix_multiplication(two_key, np.c_[two_7]).tolist()
    new_two_8 = matrix_multiplication(two_key, np.c_[two_8]).tolist()
    new_two_9 = matrix_multiplication(two_key, np.c_[two_9]).tolist()
    new_two_10 = matrix_multiplication(two_key, np.c_[two_10]).tolist()
    new_two_11 = matrix_multiplication(two_key, np.c_[two_11]).tolist()
    new_two_12 = matrix_multiplication(two_key, np.c_[two_12]).tolist()
    new_two_13 = matrix_multiplication(two_key, np.c_[two_13]).tolist()
    new_two_14 = matrix_multiplication(two_key, np.c_[two_14]).tolist()
    new_two_15 = matrix_multiplication(two_key, np.c_[two_15]).tolist()
    new_two_16 = matrix_multiplication(two_key, np.c_[two_16]).tolist()
    new_two_17 = matrix_multiplication(two_key, np.c_[two_17]).tolist()
    new_two_18 = matrix_multiplication(two_key, np.c_[two_18]).tolist()

    two_count = 0
    while two_count < 2:
        list[two_count] = new_two_1[two_count]
        two_count += 1
    while two_count < 4:
        list[two_count] = new_two_2[two_count - 2]
        two_count += 1
    while two_count < 6:
        list[two_count] = new_two_3[two_count - 4]
        two_count += 1
    while two_count < 8:
        list[two_count] = new_two_4[two_count - 6]
        two_count += 1
    while two_count < 10:
        list[two_count] = new_two_5[two_count - 8]
        two_count += 1
    while two_count < 12:
        list[two_count] = new_two_6[two_count - 10]
        two_count += 1
    while two_count < 14:
        list[two_count] = new_two_7[two_count - 12]
        two_count += 1
    while two_count < 16:
        list[two_count] = new_two_8[two_count - 14]
        two_count += 1
    while two_count < 18:
        list[two_count] = new_two_9[two_count - 16]
        two_count += 1
    while two_count < 20:
        list[two_count] = new_two_10[two_count - 18]
        two_count += 1
    while two_count < 22:
        list[two_count] = new_two_11[two_count - 20]
        two_count += 1
    while two_count < 24:
        list[two_count] = new_two_12[two_count - 22]
        two_count += 1
    while two_count < 26:
        list[two_count] = new_two_13[two_count - 24]
        two_count += 1
    while two_count < 28:
        list[two_count] = new_two_14[two_count - 26]
        two_count += 1
    while two_count < 30:
        list[two_count] = new_two_15[two_count - 28]
        two_count += 1
    while two_count < 32:
        list[two_count] = new_two_16[two_count - 30]
        two_count += 1
    while two_count < 34:
        list[two_count] = new_two_17[two_count - 32]
        two_count += 1
    while two_count < 36:
        list[two_count] = new_two_18[two_count - 34]
        two_count += 1

def inverse_two(list):
    two_count = 0
    two_1 = []
    two_2 = []
    two_3 = []
    two_4 = []
    two_5 = []
    two_6 = []
    two_7 = []
    two_8 = []
    two_9 = []
    two_10 = []
    two_11 = []
    two_12 = []
    two_13 = []
    two_14 = []
    two_15 = []
    two_16 = []
    two_17 = []
    two_18 = []
    while two_count < 2:
        two_1.append(list[two_count])
        two_count += 1
    while two_count < 4:
        two_2.append(list[two_count])
        two_count += 1
    while two_count < 6:
        two_3.append(list[two_count])
        two_count += 1
    while two_count < 8:
        two_4.append(list[two_count])
        two_count += 1
    while two_count < 10:
        two_5.append(list[two_count])
        two_count += 1
    while two_count < 12:
        two_6.append(list[two_count])
        two_count += 1
    while two_count < 14:
        two_7.append(list[two_count])
        two_count += 1
    while two_count < 16:
        two_8.append(list[two_count])
        two_count += 1
    while two_count < 18:
        two_9.append(list[two_count])
        two_count += 1
    while two_count < 20:
        two_10.append(list[two_count])
        two_count += 1
    while two_count < 22:
        two_11.append(list[two_count])
        two_count += 1
    while two_count < 24:
        two_12.append(list[two_count])
        two_count += 1
    while two_count < 26:
        two_13.append(list[two_count])
        two_count += 1
    while two_count < 28:
        two_14.append(list[two_count])
        two_count += 1
    while two_count < 30:
        two_15.append(list[two_count])
        two_count += 1
    while two_count < 32:
        two_16.append(list[two_count])
        two_count += 1
    while two_count < 34:
        two_17.append(list[two_count])
        two_count += 1
    while two_count < 36:
        two_18.append(list[two_count])
        two_count += 1

    new_two_1 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_1]).tolist()
    new_two_2 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_2]).tolist()
    new_two_3 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_3]).tolist()
    new_two_4 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_4]).tolist()
    new_two_5 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_5]).tolist()
    new_two_6 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_6]).tolist()
    new_two_7 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_7]).tolist()
    new_two_8 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_8]).tolist()
    new_two_9 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_9]).tolist()
    new_two_10 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_10]).tolist()
    new_two_11 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_11]).tolist()
    new_two_12 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_12]).tolist()
    new_two_13 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_13]).tolist()
    new_two_14 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_14]).tolist()
    new_two_15 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_15]).tolist()
    new_two_16 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_16]).tolist()
    new_two_17 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_17]).tolist()
    new_two_18 = matrix_multiplication(modMatInv(two_key,27), np.c_[two_18]).tolist()

    two_count = 0
    while two_count < 2:
        list[two_count] = new_two_1[two_count]
        two_count += 1
    while two_count < 4:
        list[two_count] = new_two_2[two_count - 2]
        two_count += 1
    while two_count < 6:
        list[two_count] = new_two_3[two_count - 4]
        two_count += 1
    while two_count < 8:
        list[two_count] = new_two_4[two_count - 6]
        two_count += 1
    while two_count < 10:
        list[two_count] = new_two_5[two_count - 8]
        two_count += 1
    while two_count < 12:
        list[two_count] = new_two_6[two_count - 10]
        two_count += 1
    while two_count < 14:
        list[two_count] = new_two_7[two_count - 12]
        two_count += 1
    while two_count < 16:
        list[two_count] = new_two_8[two_count - 14]
        two_count += 1
    while two_count < 18:
        list[two_count] = new_two_9[two_count - 16]
        two_count += 1
    while two_count < 20:
        list[two_count] = new_two_10[two_count - 18]
        two_count += 1
    while two_count < 22:
        list[two_count] = new_two_11[two_count - 20]
        two_count += 1
    while two_count < 24:
        list[two_count] = new_two_12[two_count - 22]
        two_count += 1
    while two_count < 26:
        list[two_count] = new_two_13[two_count - 24]
        two_count += 1
    while two_count < 28:
        list[two_count] = new_two_14[two_count - 26]
        two_count += 1
    while two_count < 30:
        list[two_count] = new_two_15[two_count - 28]
        two_count += 1
    while two_count < 32:
        list[two_count] = new_two_16[two_count - 30]
        two_count += 1
    while two_count < 34:
        list[two_count] = new_two_17[two_count - 32]
        two_count += 1
    while two_count < 36:
        list[two_count] = new_two_18[two_count - 34]
        two_count += 1

def three(list):
    three_count = 0
    three_1 = []
    three_2 = []
    three_3 = []
    three_4 = []
    three_5 = []
    three_6 = []
    three_7 = []
    three_8 = []
    three_9 = []
    three_10 = []
    three_11 = []
    three_12 = []
    while three_count < 3:
        three_1.append(list[three_count])
        three_count += 1
    while three_count < 6:
        three_2.append(list[three_count])
        three_count += 1
    while three_count < 9:
        three_3.append(list[three_count])
        three_count += 1
    while three_count < 12:
        three_4.append(list[three_count])
        three_count += 1
    while three_count < 15:
        three_5.append(list[three_count])
        three_count += 1
    while three_count < 18:
        three_6.append(list[three_count])
        three_count += 1
    while three_count < 21:
        three_7.append(list[three_count])
        three_count += 1
    while three_count < 24:
        three_8.append(list[three_count])
        three_count += 1
    while three_count < 27:
        three_9.append(list[three_count])
        three_count += 1
    while three_count < 30:
        three_10.append(list[three_count])
        three_count += 1
    while three_count < 33:
        three_11.append(list[three_count])
        three_count += 1
    while three_count < 36:
        three_12.append(list[three_count])
        three_count += 1

    new_three_1 = matrix_multiplication(three_key, np.c_[three_1]).tolist()
    new_three_2 = matrix_multiplication(three_key, np.c_[three_2]).tolist()
    new_three_3 = matrix_multiplication(three_key, np.c_[three_3]).tolist()
    new_three_4 = matrix_multiplication(three_key, np.c_[three_4]).tolist()
    new_three_5 = matrix_multiplication(three_key, np.c_[three_5]).tolist()
    new_three_6 = matrix_multiplication(three_key, np.c_[three_6]).tolist()
    new_three_7 = matrix_multiplication(three_key, np.c_[three_7]).tolist()
    new_three_8 = matrix_multiplication(three_key, np.c_[three_8]).tolist()
    new_three_9 = matrix_multiplication(three_key, np.c_[three_9]).tolist()
    new_three_10 = matrix_multiplication(three_key, np.c_[three_10]).tolist()
    new_three_11 = matrix_multiplication(three_key, np.c_[three_11]).tolist()
    new_three_12 = matrix_multiplication(three_key, np.c_[three_12]).tolist()

    three_count = 0
    while three_count < 3:
        list[three_count] = new_three_1[three_count]
        three_count += 1
    while three_count < 6:
        list[three_count] = new_three_2[three_count - 3]
        three_count += 1
    while three_count < 9:
        list[three_count] = new_three_3[three_count - 6]
        three_count += 1
    while three_count < 12:
        list[three_count] = new_three_4[three_count - 9]
        three_count += 1
    while three_count < 15:
        list[three_count] = new_three_5[three_count - 12]
        three_count += 1
    while three_count < 18:
        list[three_count] = new_three_6[three_count - 15]
        three_count += 1
    while three_count < 21:
        list[three_count] = new_three_7[three_count - 18]
        three_count += 1
    while three_count < 24:
        list[three_count] = new_three_8[three_count - 21]
        three_count += 1
    while three_count < 27:
        list[three_count] = new_three_9[three_count - 24]
        three_count += 1
    while three_count < 30:
        list[three_count] = new_three_10[three_count - 27]
        three_count += 1
    while three_count < 33:
        list[three_count] = new_three_11[three_count - 30]
        three_count += 1
    while three_count < 36:
        list[three_count] = new_three_12[three_count - 33]
        three_count += 1

def inverse_three(list):
    three_count = 0
    three_1 = []
    three_2 = []
    three_3 = []
    three_4 = []
    three_5 = []
    three_6 = []
    three_7 = []
    three_8 = []
    three_9 = []
    three_10 = []
    three_11 = []
    three_12 = []
    while three_count < 3:
        three_1.append(list[three_count])
        three_count += 1
    while three_count < 6:
        three_2.append(list[three_count])
        three_count += 1
    while three_count < 9:
        three_3.append(list[three_count])
        three_count += 1
    while three_count < 12:
        three_4.append(list[three_count])
        three_count += 1
    while three_count < 15:
        three_5.append(list[three_count])
        three_count += 1
    while three_count < 18:
        three_6.append(list[three_count])
        three_count += 1
    while three_count < 21:
        three_7.append(list[three_count])
        three_count += 1
    while three_count < 24:
        three_8.append(list[three_count])
        three_count += 1
    while three_count < 27:
        three_9.append(list[three_count])
        three_count += 1
    while three_count < 30:
        three_10.append(list[three_count])
        three_count += 1
    while three_count < 33:
        three_11.append(list[three_count])
        three_count += 1
    while three_count < 36:
        three_12.append(list[three_count])
        three_count += 1

    new_three_1 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_1]).tolist()
    new_three_2 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_2]).tolist()
    new_three_3 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_3]).tolist()
    new_three_4 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_4]).tolist()
    new_three_5 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_5]).tolist()
    new_three_6 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_6]).tolist()
    new_three_7 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_7]).tolist()
    new_three_8 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_8]).tolist()
    new_three_9 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_9]).tolist()
    new_three_10 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_10]).tolist()
    new_three_11 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_11]).tolist()
    new_three_12 = matrix_multiplication(modMatInv(three_key,27), np.c_[three_12]).tolist()

    three_count = 0
    while three_count < 3:
        list[three_count] = new_three_1[three_count]
        three_count += 1
    while three_count < 6:
        list[three_count] = new_three_2[three_count - 3]
        three_count += 1
    while three_count < 9:
        list[three_count] = new_three_3[three_count - 6]
        three_count += 1
    while three_count < 12:
        list[three_count] = new_three_4[three_count - 9]
        three_count += 1
    while three_count < 15:
        list[three_count] = new_three_5[three_count - 12]
        three_count += 1
    while three_count < 18:
        list[three_count] = new_three_6[three_count - 15]
        three_count += 1
    while three_count < 21:
        list[three_count] = new_three_7[three_count - 18]
        three_count += 1
    while three_count < 24:
        list[three_count] = new_three_8[three_count - 21]
        three_count += 1
    while three_count < 27:
        list[three_count] = new_three_9[three_count - 24]
        three_count += 1
    while three_count < 30:
        list[three_count] = new_three_10[three_count - 27]
        three_count += 1
    while three_count < 33:
        list[three_count] = new_three_11[three_count - 30]
        three_count += 1
    while three_count < 36:
        list[three_count] = new_three_12[three_count - 33]
        three_count += 1

def four(list):
    four_count = 0
    four_1 = []
    four_2 = []
    four_3 = []
    four_4 = []
    four_5 = []
    four_6 = []
    four_7 = []
    four_8 = []
    four_9 = []
    while four_count < 4:
        four_1.append(list[four_count])
        four_count += 1
    while four_count < 8:
        four_2.append(list[four_count])
        four_count += 1
    while four_count < 12:
        four_3.append(list[four_count])
        four_count += 1
    while four_count < 16:
        four_4.append(list[four_count])
        four_count += 1
    while four_count < 20:
        four_5.append(list[four_count])
        four_count += 1
    while four_count < 24:
        four_6.append(list[four_count])
        four_count += 1
    while four_count < 28:
        four_7.append(list[four_count])
        four_count += 1
    while four_count < 32:
        four_8.append(list[four_count])
        four_count += 1
    while four_count < 36:
        four_9.append(list[four_count])
        four_count += 1

    new_four_1 = matrix_multiplication(four_key, np.c_[four_1]).tolist()
    new_four_2 = matrix_multiplication(four_key, np.c_[four_2]).tolist()
    new_four_3 = matrix_multiplication(four_key, np.c_[four_3]).tolist()
    new_four_4 = matrix_multiplication(four_key, np.c_[four_4]).tolist()
    new_four_5 = matrix_multiplication(four_key, np.c_[four_5]).tolist()
    new_four_6 = matrix_multiplication(four_key, np.c_[four_6]).tolist()
    new_four_7 = matrix_multiplication(four_key, np.c_[four_7]).tolist()
    new_four_8 = matrix_multiplication(four_key, np.c_[four_8]).tolist()
    new_four_9 = matrix_multiplication(four_key, np.c_[four_9]).tolist()

    four_count = 0
    while four_count < 4:
        list[four_count] = new_four_1[four_count]
        four_count += 1
    while four_count < 8:
        list[four_count] = new_four_2[four_count - 4]
        four_count += 1
    while four_count < 12:
        list[four_count] = new_four_3[four_count - 8]
        four_count += 1
    while four_count < 16:
        list[four_count] = new_four_4[four_count - 12]
        four_count += 1
    while four_count < 20:
        list[four_count] = new_four_5[four_count - 16]
        four_count += 1
    while four_count < 24:
        list[four_count] = new_four_6[four_count - 20]
        four_count += 1
    while four_count < 28:
        list[four_count] = new_four_7[four_count - 24]
        four_count += 1
    while four_count < 32:
        list[four_count] = new_four_8[four_count - 28]
        four_count += 1
    while four_count < 36:
        list[four_count] = new_four_9[four_count - 32]
        four_count += 1

def inverse_four(list):
    four_count = 0
    four_1 = []
    four_2 = []
    four_3 = []
    four_4 = []
    four_5 = []
    four_6 = []
    four_7 = []
    four_8 = []
    four_9 = []
    while four_count < 4:
        four_1.append(list[four_count])
        four_count += 1
    while four_count < 8:
        four_2.append(list[four_count])
        four_count += 1
    while four_count < 12:
        four_3.append(list[four_count])
        four_count += 1
    while four_count < 16:
        four_4.append(list[four_count])
        four_count += 1
    while four_count < 20:
        four_5.append(list[four_count])
        four_count += 1
    while four_count < 24:
        four_6.append(list[four_count])
        four_count += 1
    while four_count < 28:
        four_7.append(list[four_count])
        four_count += 1
    while four_count < 32:
        four_8.append(list[four_count])
        four_count += 1
    while four_count < 36:
        four_9.append(list[four_count])
        four_count += 1

    new_four_1 = matrix_multiplication(modMatInv(four_key,27), np.c_[four_1]).tolist()
    new_four_2 = matrix_multiplication(modMatInv(four_key,27), np.c_[four_2]).tolist()
    new_four_3 = matrix_multiplication(modMatInv(four_key,27), np.c_[four_3]).tolist()
    new_four_4 = matrix_multiplication(modMatInv(four_key,27), np.c_[four_4]).tolist()
    new_four_5 = matrix_multiplication(modMatInv(four_key,27), np.c_[four_5]).tolist()
    new_four_6 = matrix_multiplication(modMatInv(four_key,27), np.c_[four_6]).tolist()
    new_four_7 = matrix_multiplication(modMatInv(four_key,27), np.c_[four_7]).tolist()
    new_four_8 = matrix_multiplication(modMatInv(four_key,27), np.c_[four_8]).tolist()
    new_four_9 = matrix_multiplication(modMatInv(four_key,27), np.c_[four_9]).tolist()

    four_count = 0
    while four_count < 4:
        list[four_count] = new_four_1[four_count]
        four_count += 1
    while four_count < 8:
        list[four_count] = new_four_2[four_count - 4]
        four_count += 1
    while four_count < 12:
        list[four_count] = new_four_3[four_count - 8]
        four_count += 1
    while four_count < 16:
        list[four_count] = new_four_4[four_count - 12]
        four_count += 1
    while four_count < 20:
        list[four_count] = new_four_5[four_count - 16]
        four_count += 1
    while four_count < 24:
        list[four_count] = new_four_6[four_count - 20]
        four_count += 1
    while four_count < 28:
        list[four_count] = new_four_7[four_count - 24]
        four_count += 1
    while four_count < 32:
        list[four_count] = new_four_8[four_count - 28]
        four_count += 1
    while four_count < 36:
        list[four_count] = new_four_9[four_count - 32]
        four_count += 1

def six(list):
    six_count = 0
    six_1 = []
    six_2 = []
    six_3 = []
    six_4 = []
    six_5 = []
    six_6 = []
    while six_count < 6:
        six_1.append(list[six_count])
        six_count += 1
    while six_count < 12:
        six_2.append(list[six_count])
        six_count += 1
    while six_count < 18:
        six_3.append(list[six_count])
        six_count += 1
    while six_count < 24:
        six_4.append(list[six_count])
        six_count += 1
    while six_count < 30:
        six_5.append(list[six_count])
        six_count += 1
    while six_count < 36:
        six_6.append(list[six_count])
        six_count += 1


    new_six_1 = matrix_multiplication(six_key, np.c_[six_1]).tolist()
    new_six_2 = matrix_multiplication(six_key, np.c_[six_2]).tolist()
    new_six_3 = matrix_multiplication(six_key, np.c_[six_3]).tolist()
    new_six_4 = matrix_multiplication(six_key, np.c_[six_4]).tolist()
    new_six_5 = matrix_multiplication(six_key, np.c_[six_5]).tolist()
    new_six_6 = matrix_multiplication(six_key, np.c_[six_6]).tolist()

    six_count = 0
    while six_count < 6:
        list[six_count] = new_six_1[six_count]
        six_count += 1
    while six_count < 12:
        list[six_count] = new_six_2[six_count - 6]
        six_count += 1
    while six_count < 18:
        list[six_count] = new_six_3[six_count - 12]
        six_count += 1
    while six_count < 24:
        list[six_count] = new_six_4[six_count - 18]
        six_count += 1
    while six_count < 30:
        list[six_count] = new_six_5[six_count - 24]
        six_count += 1
    while six_count < 36:
        list[six_count] = new_six_6[six_count - 30]
        six_count += 1

def inverse_six(list):
    six_count = 0
    six_1 = []
    six_2 = []
    six_3 = []
    six_4 = []
    six_5 = []
    six_6 = []
    while six_count < 6:
        six_1.append(list[six_count])
        six_count += 1
    while six_count < 12:
        six_2.append(list[six_count])
        six_count += 1
    while six_count < 18:
        six_3.append(list[six_count])
        six_count += 1
    while six_count < 24:
        six_4.append(list[six_count])
        six_count += 1
    while six_count < 30:
        six_5.append(list[six_count])
        six_count += 1
    while six_count < 36:
        six_6.append(list[six_count])
        six_count += 1


    new_six_1 = matrix_multiplication(modMatInv(six_key,27), np.c_[six_1]).tolist()
    new_six_2 = matrix_multiplication(modMatInv(six_key,27), np.c_[six_2]).tolist()
    new_six_3 = matrix_multiplication(modMatInv(six_key,27), np.c_[six_3]).tolist()
    new_six_4 = matrix_multiplication(modMatInv(six_key,27), np.c_[six_4]).tolist()
    new_six_5 = matrix_multiplication(modMatInv(six_key,27), np.c_[six_5]).tolist()
    new_six_6 = matrix_multiplication(modMatInv(six_key,27), np.c_[six_6]).tolist()

    six_count = 0
    while six_count < 6:
        list[six_count] = new_six_1[six_count]
        six_count += 1
    while six_count < 12:
        list[six_count] = new_six_2[six_count - 6]
        six_count += 1
    while six_count < 18:
        list[six_count] = new_six_3[six_count - 12]
        six_count += 1
    while six_count < 24:
        list[six_count] = new_six_4[six_count - 18]
        six_count += 1
    while six_count < 30:
        list[six_count] = new_six_5[six_count - 24]
        six_count += 1
    while six_count < 36:
        list[six_count] = new_six_6[six_count - 30]
        six_count += 1

def main():
    global plaintext, cyphertext, cyphertext2, cyphertext3, cyphertext4, cyphertext5, cyphertext6, cyphertext7, cyphertext8, holding

    for number in range(len(order)):
        if order[number] == '6':
            six(plaintext)
            plaintext = [x for l in plaintext for x in l]  # removes double listing
            cyphertext = [round(num) for num in plaintext]
            plaintext = cyphertext
            print('6 Ciphertext: ', plaintext)
        if order[number] == '4':
            four(plaintext)
            plaintext = [x for l in plaintext for x in l]
            cyphertext = [round(num) for num in plaintext]
            plaintext = cyphertext
            print('4 Ciphertext: ', plaintext)
        if order[number] == '3':
            three(plaintext)
            plaintext = [x for l in plaintext for x in l]
            cyphertext = [round(num) for num in plaintext]
            plaintext = cyphertext
            print('3 Ciphertext: ', plaintext)
        if order[number] == '2':
            two(plaintext)
            plaintext = [x for l in plaintext for x in l]
            cyphertext = [round(num) for num in plaintext]
            plaintext = cyphertext
            print('2 Ciphertext: ', plaintext)

    engwishate(plaintext)
    reserve = engwish

    for number in range(len(reversex)):
        if reversex[number] == '2':
            inverse_two(raw)
            plaintext = raw
            plaintext = [x for l in plaintext for x in l]
            cyphertext = [round(num) for num in plaintext]
            plaintext = cyphertext
            print('2 Ciphertext Decrypted: ', plaintext)
        if reversex[number] == '3':
            inverse_three(raw)
            plaintext = raw
            plaintext = [x for l in plaintext for x in l]
            cyphertext = [round(num) for num in plaintext]
            plaintext = cyphertext
            print('3 Ciphertext Decrypted: ', plaintext)
        if reversex[number] == '4':
            inverse_four(raw)
            plaintext = raw
            plaintext = [x for l in plaintext for x in l]
            cyphertext = [round(num) for num in plaintext]
            plaintext = cyphertext
            print('4 Ciphertext Decrypted: ', plaintext)
        if reversex[number] == '6':
            inverse_six(raw)
            plaintext = raw
            plaintext = [x for l in plaintext for x in l]
            cyphertext = [round(num) for num in plaintext]
            plaintext = cyphertext
            print('6 Ciphertext Decrypted: ', plaintext)

    print('')
    engwishate2(plaintext)
    print('Your key pattern was: ', order)
    print('Your encoding was: ', reserve)
    #print('Numerical value of encoding was: ', holding2)

if __name__ == '__main__':
    main()



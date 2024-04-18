temp_array = []

for i in range(1,101):
    if (i%3 == 0) or (i%5 == 0):
        if i % 10 != 0:
            temp_array.append(i)

print(temp_array)
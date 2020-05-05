total = 29810
for i in range(6000, 7000):
    if (total - i) % 18 == 0 and i % 18 == 0:
        print(i)
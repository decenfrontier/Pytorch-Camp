import os
print(__file__)
print(os.path.abspath(__file__))
print(os.getcwd())

for root, dirs, files in os.walk("F:/Study/NCK"):
    print("root:", root)
    print("dirs:", dirs)
    print("files:", files)
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))
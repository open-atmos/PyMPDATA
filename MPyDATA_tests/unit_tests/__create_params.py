import numpy as np
import sys


from os import listdir
path = "C:\\Users\\piotr\\PycharmProjects\\MPyDATA\\MPyDATA_tests\\tmp"
files = [f for f in listdir(path)]
l = []
for file in files:
    if file.endswith("in"):
        Cx = file.find("_Cx=")
        Cy = file.find("_Cy=")
        nt = file.find("_nt=")
        it = file.find("_it=")
        l.append((int(file[3:7]), int(file[11:15]),
         float(file[Cx+4:Cy]),  float(file[Cy+4:nt]),
         int(file[nt+4:it]), int(file[it+4:it+5]),
         np.loadtxt(path + "\\" + file), np.loadtxt(path + "\\" + file.replace("in", "out"))))

with open(path + "\\" + "params.py", "w") as out:
    out.write(str(l))



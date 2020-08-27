import matlab.engine
import sys


eng=matlab.engine.start_matlab()
str = raw_input()
x=str.split(' ')
res=eng.MEEMTrack(x[0],x[1],x[2]=="true",matlab.double([int(x[3]),int(x[4]),int(x[5]),int(x[6])]),int(x[7]))

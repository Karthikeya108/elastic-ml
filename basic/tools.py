#data = [{"data":[[x],[y]], "classes":[c]}, {"data":[[x],[y]], "classes":[c]}]
import sys, re, time, fcntl, os, random, gzip, math
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
from pysgpp import *

# constants
ARFF = 1
SIMPLE = 0
NOTAFILE = -1

def gzOpen(filename, mode="r"):
    # gzip-file
    if re.match(".*\.gz$", filename):
        # mode set for binary data?
        if not mode[-1] == "b":
            mode += "b"
        fd = gzip.open(filename, mode)
    # non gzip-file
    else:
        fd = open(filename, mode)
    return fd


def isARFFFile(filename):
    try:
        # read lines until non-empty line found
        f = gzOpen(filename, 'r')
        data = f.readline().strip()
        while len(data) == 0:
            data = f.readline().strip()
        f.close()
        # check whether it starts with "@"
        if re.match('@', data):
            return ARFF
        else:
            return SIMPLE
    except:
        return NOTAFILE

def readDataTrivial(filename, delim = None, hasclass = True):
    fin = gzOpen(filename, "r")
    data = []
    classes = []
    for line in fin:
        sline = line.strip()
        # skip empty lines and comments
        if sline.startswith("#") or len(sline) == 0:
            continue

        # split and convert 
        values = sline.split(delim)
        values = map(lambda x: x.strip(), values)
        values = filter(lambda x: len(x) > 0, values)
        values = map(lambda x: float(x), values)
        
        if hasclass:
            data.append(values[:-1])
            classes.append(values[-1])
        else:
            data.append(values)

    # cleaning up and return
    fin.close()
    if hasclass:
        return {"data": data, "classes": classes, "filename":filename}
    else:
        return {"data": data, "filename":filename}

def readDataARFF(filename):
    fin = gzOpen(filename, "r")
    data = []
    classes = []
    hasclass = False

    # get the different section of ARFF-File
    for line in fin:
        sline = line.strip().lower()
        # skip comments and empty lines
        if sline.startswith("%") or len(sline) == 0:
            continue

        if sline.startswith("@data"):
            break
        
        if sline.startswith("@attribute"):
            value = sline.split()
            if value[1].startswith("class"):
                hasclass = True
    
    #read in the data stored in the ARFF file
    for line in fin:
        sline = line.strip()
        # skip comments and empty lines
        if sline.startswith("%") or len(sline) == 0:
            continue

        # split and convert 
        values = sline.split(",")
        values = map(lambda x: x.strip(), values)
        values = filter(lambda x: len(x) > 0, values)
        values = map(lambda x: float(x), values)
        
        if hasclass:
            data.append(values[:-1])
            classes.append(values[-1])
        else:
            data.append(values)
            
    # cleaning up and return
    fin.close()
    if hasclass:
        return {"data": data, "classes": classes, "filename":filename}
    else:
        return {"data": data, "filename":filename}

def readData(filename):
    try:
        if isARFFFile(filename):
            data = readDataARFF(filename)
        else:
            data = readDataTrivial(filename)
    except Exception, e:
        print ("An error occured while reading " + filename + "!")
        raise e
        
    if data.has_key("classes") == False:
        print ("No classes found in the given File " + filename + "!")
        sys.exit(1)
        
    return data

class Matrix:
    def __init__(self, grid, x, l, mode):
        self.grid = grid
        self.x = x
        self.l = l
        self.B = createOperationMultipleEval(grid, x)
        self.CMode = mode.lower()
        
        if self.CMode == "laplace":
            self.C = createOperationLaplace(grid)
        elif self.CMode == "identity":
            self.C = createOperationIdentity(grid)
    
    def generateb(self, y):
        b = DataVector(self.grid.getStorage().size())
        self.B.multTranspose(y, b)
        return b
    
    def ApplyMatrix(self, alpha, result):
        M = self.x.getNrows();
        temp = DataVector(M)
    
        self.B.mult(alpha, temp)
        self.B.multTranspose(temp, result)

        if self.CMode == "laplace":
            temp = DataVector(len(alpha))
            self.C.mult(alpha, temp)
            result.axpy(M*self.l, temp)
        elif self.CMode == "identity":
            result.axpy(M*self.l, alpha)

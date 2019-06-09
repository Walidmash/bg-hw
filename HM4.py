import sys
import math
import numpy as np
import time
import random

from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vectors
from functools import partial
from random import randint


#--------------------------------------------------------------
# K-medain distance function
def distance(a, b):
    return np.linalg.norm(a-b)
# K-medain distance between set of points and one center
def bulkDistance(center,points):
    return np.linalg.norm(points - center, axis=1)

# K-median distance from closest point
# p= point, s= set of points, i= valid number of points in s
def d(p,s,i=-1):
    if i == -1 : i=len(s)
    # a = np.array([distance(p,x) for x in s[0:i]])
    a = bulkDistance(p,s[0:i])
    return np.amin(a)

# probability function = wp*(d_p)/(sum_{q in p} w_q*(d_q))
# point = a single point, wpoint = weight of the point
# s= set of centers, wp= set of weights
def probability(point,p,s,wpoint, wp):
    sumation = 0
    for i,q in enumerate(p):
        sumation += d(q,s)*wp[i]
    return wpoint*d(point,s)/sumation

# draw random point from set p,
# with a probability function = w_p*(d_p)/(sum_{q non center} w_q*(d_q))
# s= set of centers, wp= set of weights
# returns index of the selected point
def randomPoint(p,s,wp):
    probSum = 0
    randValue = random.random() # random value between 0 and 1
    sumation = 0
    for i,q in enumerate(p):
        sumation += d(q,s)*wp[i]
        res=0
    for i, point in enumerate(p):
        probSum += wp[i]*d(point,s)/sumation
        if probSum > randValue: return i
    return -1 # error case

def partition(p,s,k):
    C = np.zeros(shape=(p.shape[0]+s.shape[0],p.shape[1]+1))
    pointNum = 0

    for ps in p:
        dist = np.array([distance(ps,x) for x in s])
        l = np.argmin(dist)

        #We add the index of the cluster at the end of the vector representing a point
        C[pointNum,:] = np.concatenate((ps,l),axis = None)
        pointNum += 1
    return C

def centroid(C,k):
    centroid = np.zeros(shape=(k,C.shape[1]-1))
    for i in range(k):
        pointNum = 0
        c_sum = np.zeros(C.shape[1]-1)
        for j in range(C.shape[0]):
            if C[j,-1] == i:
                c_sum += C[j,:-1]
                pointNum += 1
 
        centroid[i,:] = c_sum/pointNum
    return centroid

# K-means++
def kmeansPP(p,wp,k,iter):
    print("Running k-means++, to obtain initial centers...")
    pc = np.copy(p)
    wpc = np.copy(wp)
    s1 = random.randint(0,len(p)-1) # pick random starting point
    s= np.zeros(shape=(k,p.shape[1]))
    s[0] = np.copy(p[s1])
    pc = np.delete(pc,s1,0)
    wpc = np.delete(wpc,s1)
    for i in range(1,k):
        randPoint = randomPoint(pc,s,wpc)
        s[i] = np.copy(pc[randPoint])
        pc = np.delete(pc,randPoint,0)
        wpc = np.delete(wpc,randPoint)

    fi = float('inf')
    print("Running Lloyd’s algorithm...")
    stopping_condition = 0
    for i in range (0,iter):
        C = partition(p,s,k)
        cent = centroid(C,k) #Centroids of each cluster
        fi_kmeans = 0
        for cc in C:
            for i in range(k):
                if cc[-1] == i:
                    fi_kmeans += distance(cc[:-1],s[i,:])

        if fi_kmeans < fi:
            fi = fi_kmeans
            s = cent
        else : break

    return s

# kmeansObj receives in input a set of points P and a set of centers C,
# and returns the average distance of a point of P from C
def kmeansObj(p,c):
    return np.sum([d(point,c) for point in p])/len(p)

#--------------------------------------------------------------

def compute_weights(points, centers):
   weights = np.zeros(len(centers))
   for point in points:
      mycenter = 0
      # mindist = math.sqrt(point.squared_distance(centers[0]))
      mindist = distance(point,centers[0])
      for i in range(1,len(centers)):
         # if(math.sqrt(point.squared_distance(centers[i])) < mindist):
         if(distance(point,centers[i]) < mindist):
            # mindist = math.sqrt(point.squared_distance(centers[i]))
            mindist = distance(point, centers[i])
            mycenter = i
      weights[mycenter] = weights[mycenter] + 1
   return weights

def f2(k, L, iterations, partition):
   points = np.array([vector for vector in iter(partition)])
   weights = np.ones(len(points))
   centers = kmeansPP(points, weights, k, iterations)
   final_weights = compute_weights(points, centers)
   return [(vect, weight) for vect,weight in zip(centers,final_weights)]
def MR_kmedian(pointset, k, L, iterations):
   #---------- ROUND 1 ---------------
   coreset = pointset.mapPartitions(partial(f2, k, L, iterations))
   #---------- ROUND 2 ---------------
   centersR1 = []
   weightsR1 = []
   for pair in coreset.collect():
      centersR1.append(pair[0])
      weightsR1.append(pair[1])
   centers = kmeansPP(np.array(centersR1),np.array(weightsR1), k, iterations)
   #---------- ROUND 3 --------------------------
   obj = pointset.repartition(L).map(lambda p: d(p,centers)).reduce(lambda a,b: a+b)
   return obj

def f1(line):
   return Vectors.dense([float(coord) for coord in line.split(" ") if len(coord) > 0])

def main(argv):
   #Avoided controls on input..
   dataset = argv[1]
   k = int(argv[2])
   L = int(argv[3])
   iterations = int(argv[4])
   conf = SparkConf().setAppName('HM4 python Template')
   sc = SparkContext(conf=conf)
   pointset = sc.textFile(dataset).map(f1).repartition(L).cache()
   N = pointset.count()
   print("Number of points is : " + str(N))
   print("Number of clusters is : " + str(k))
   print("Number of parts is : " + str(L))
   print("Number of iterations is : " + str(iterations))
   obj = MR_kmedian(pointset, k, L, iterations)
   print("Objective function is : " + str(obj/N))

if __name__ == '__main__':
   if(len(sys.argv) != 5):
      print("Usage: <pathToFile> k L iter")
      sys.exit(0)
   main(sys.argv)

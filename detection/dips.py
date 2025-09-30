"""
Draws bounding boxes from the clustered lowest points together with a slope analysis for each cluster.
The lowest point clusters get expanded until a change of slope occurs after a steep slope.
The bounding boxes are anchored by the edge points of the expanded clusters in the x direction. (width = x-distance between edge points)
The bounding boxes are anchored by the lowest point of the expanded clusters in the y direction. (height = y-distance between edge points and lowest point * 2)

"""

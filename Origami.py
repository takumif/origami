# -*- coding: utf-8 -*-
# Origami.py by Takumi Fujimoto

from visual import *
from copy import *
import operator
import wx
import wx.lib.scrolledpanel as scrolled
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom


# SETTINGS
FRONT_COLOR = color.yellow
BACK_COLOR = color.white
BACKGROUND_COLOR = color.gray(0.2) # 0 is white, 1 is black



def vertexOnEdge(edge, coords):
    # gives back the Vertex that exists at the coords on the edge by 
    # calculating its original coordinates
    v1X, v1Y, v1Z = edge.ends[0].coords
    v2X, v2Y, v2Z = edge.ends[1].coords
    if v1X != v2X:
        ratio = (v1X - coords.x) / (v1X - v2X)
    elif v1Y != v2Y:
        ratio = (v1Y - coords.y) / (v1Y - v2Y)
    # calculate the origimalCoords
    originX = v1X - (v1X-v2X)*ratio
    originY = v1Y - (v1Y-v2Y)*ratio
    originZ = v1Z - (v1Z-v2Z)*ratio
    return Vertex(coords, (originX, originY, originZ))

def intersectionEdgeVector(edge, line):
    # ( line == (point vector, direction vector) )
    # returns the Vertex at which the edge and line intersect by calling the 
    # intersection function, returns None if they don't intersect
    ePoint = edge.ends[0].coords
    eDirection = tuple(edge.ends[0].coords[i] - 
        edge.ends[1].coords[i] for i in range(len(ePoint)))
    intPoint = intersection((ePoint, eDirection), line)

    epsilon = 0.000001
    if (intPoint and ((edge.ends[0].x-intPoint.x) *
        (edge.ends[1].x-intPoint.x) < epsilon) and 
        ((edge.ends[0].y-intPoint.y) *
        (edge.ends[1].y-intPoint.y) < epsilon)):
        return vertexOnEdge(edge, intPoint)
    else:
        return None

def intersection(line1, line2):
    # ( line == (point vector, direction vector) )
    # if they intersect then make a point if necessary, add the intersection
    # vertex to the point, and return the vertex
    # otherwise return None
    coords = 3
    (a,b,c,d,e,f,g,h,i,j,k,l) = tuple(line[j][i] for i in range(coords) for 
        line in [line1, line2] for j in range(2))
    if b*h == f*d:
        return None
    else:
        t = (c*h - g*d + e*d - a*h)/(b*h - f*d)
        return vector(a+b*t, e+f*t, i+j*t)

def identifyFaces(newFace1, newFace2, vtx, newVs, normal):
    # cleans the faces and determines which of the two faces is moving

    # get rid of duplicated vertices
    for newFace in [newFace1, newFace2]:
        i = 0
        while i < len(newFace.vertices)-1:
            if newFace.vertices[i] == newFace.vertices[i+1]:
                newFace.vertices.pop(i)
            else:
                i += 1
    if vtx in newFace1.vertices:
        return newFace1, newFace2
    elif vtx in newFace2.vertices:
        return newFace2, newFace1
    # the vtx is not a vertex in the face, so must determine which side it is on
    else:
        for v in newFace1.vertices:
            if v != newVs[0][0] and v != newVs[1][0]:
                v1 = v
                break
        for v in newFace2.vertices:
            if v != newVs[0][0] and v != newVs[1][0]:
                v2 = v
                break
        if intersectionEdgeVector(Edge(v1, vtx), normal):
            return newFace2, newFace1
        elif intersectionEdgeVector(Edge(v2, vtx), normal):
            return newFace1, newFace2

def splitFace(pattern, face, newVs, vtx, newCoord, normal):
    # ( newVs == [(v1, edge_v1_is_on), (v2, edge_v2_is_on)] )
    # splits the face at the edge between the two new vertices
    # returns the pattern with the old face replaced with the two new faces
    # calculates the new coordinates for vertices with moveVertex method
    # unmovedPiece is the movingPiece before moving

    # the indices of the vertices that come before the newly formed vertices
    iBeforeV1 = face.vertices.index(newVs[0][1].ends[0])
    iBeforeV2 = face.vertices.index(newVs[1][1].ends[0])
    newFace1 = Face(face.vertices[:iBeforeV1+1] + [newVs[0][0]] + 
        [newVs[1][0]] + face.vertices[iBeforeV2+1:])
    newFace2 = Face([newVs[0][0]] + face.vertices[iBeforeV1+1:iBeforeV2+1]
        + [newVs[1][0]])
    
    movingPiece, stayingPiece = identifyFaces(newFace1, newFace2, vtx, newVs,
        normal)

    unmovedPiece = movingPiece.copy()

    for v in movingPiece.vertices:
        newV = Vertex(moveVertex(v, normal), v.originalCoords)
        movingPiece.vertices[movingPiece.vertices.index(v)] = newV
    movingPiece.updateEdges() # necessary since their vertices were modified
    stayingPiece.updateEdges()
    unmovedPiece.updateEdges()
    return movingPiece, stayingPiece, unmovedPiece

def changeFace(pattern, face, normal, vertex, newCoord):
    # returns the faces and vertices (if any) newly formed as the result of the
    # fold using intersectionEdgeVector and splitFace methods
    newVs = []
    for edge in face.edges:
        if  len(newVs)<2:
            newV = intersectionEdgeVector(edge, normal)
            for newVPair in newVs:
                if newVPair[0] == newV:
                    newV = None
                    break
            if newV:
                newVs.append((newV, edge))
        else:
            break
    if len(newVs) == 2:
        movingPiece, stayingPiece, unmovedPiece = splitFace(pattern, face, 
            newVs, vertex, newCoord, normal)
    else: # the entire piece is moving
        unmovedPiece = face.copy()
        movingPiece = moveFace(face, normal)
        stayingPiece = None

    return unmovedPiece, movingPiece, stayingPiece, newVs

def insertFace(affectedFaces, pattern, face, movingPiece, stayingPiece, 
    unmovedPiece):
    # alters the affectedFaces dict that contains lists of faces in a pattern
    # so that the newly formed faces are inserted at the right index
    faces = affectedFaces['originalFace']
    moving = affectedFaces['movingPiece']
    staying = affectedFaces['stayingPiece']
    unmoved = affectedFaces['unmovedFaces']
    indices = affectedFaces['faceIndices']

    affectedFaces['unmovedPiece'] = unmovedPiece

    for faceInList in faces:
        if pattern.faces.index(face) < pattern.faces.index(faceInList):
            insertIndex = faces.index(faceInList)
            faces.insert(insertIndex, face)
            moving.insert(insertIndex, movingPiece)
            staying.insert(insertIndex, stayingPiece)
            unmoved.insert(insertIndex, unmovedPiece)
            indices.insert(insertIndex, pattern.faces.index(face))
            break
    # all the faces in the faces list is in front of the face
    if face not in faces:
        faces.append(face)
        moving.append(movingPiece)
        staying.append(stayingPiece)
        unmoved.append(unmovedPiece)
        indices.append(pattern.faces.index(face))

    return affectedFaces

def addFaceAndNeighbors(affectedFaces, pattern, face, vertex, newCoord, normal):
    # 
    faces = affectedFaces['originalFace']
    moving = affectedFaces['movingPiece']
    staying = affectedFaces['stayingPiece']
    unmoved = affectedFaces['unmovedFaces']
    indices = affectedFaces['faceIndices']

    unmovedPiece, movingPiece, stayingPiece, newVs = changeFace(pattern, face, 
        normal, vertex, newCoord)

    affectedFaces = insertFace(affectedFaces, pattern, face, movingPiece, 
        stayingPiece, unmovedPiece)

    # move other faces that contain the moved vertices
    for movingVertex in unmovedPiece.vertices:
        if (not newVs) or (movingVertex not in newVs[0] and 
            movingVertex not in newVs[1]):
            for neighborFace in pattern.faces:
                if (neighborFace not in faces and 
                    movingVertex in neighborFace.vertices):
                    affectedFaces = addFaceAndNeighbors(affectedFaces, pattern, 
                        neighborFace, movingVertex, newCoord, normal)
    return affectedFaces

def moveFace(face, line):
    movedFace = Face([Vertex(moveVertex(v, line), v.originalCoords)
        for v in face.vertices])
    return movedFace

def moveVertex(vertex, line):
    coords = 3
    (x, y, z) = tuple(vertex.coords[i] for i in range(coords))
    (a, b, c, d, e, f) = tuple(line[i][j] for i in range(2) for j in 
        range(coords))
    t = ((d*(x-a) + e*(y-b) + f*(z-c)) / (d**2 + e**2 + f**2))
    diff = [vertex.coords[i]-line[0][i]-line[1][i]*t for i in range(coords)]
    return vector([vertex.coords[i]-2*diff[i] for i in range(coords)])

def closePoints(coords1, coords2):
    diff = 4
    def cube(x):
        return x**3
    return (sum(map(abs, map(cube, map(operator.sub, coords1, coords2)))) <
        diff**3)

def nearestVertex(coords, pattern):
    #points = pattern.getPoints()
    camAngle = scene.forward[2]
    if camAngle < 0: # the colored side is facing front
        for face in pattern.faces[::-1]:
            for vertex in face.vertices:
                if closePoints(coords, vertex.coords):
                    return vertex
    else: # the white side is facing front
        for face in pattern.faces: # go through the points in the reverse order
            for vertex in face.vertices:
                if closePoints(coords, vertex.coords):
                    return vertex
    return None

def nearestVertexWithSpecialVs(coords, pattern):
    specialVs = [Vertex((0,0,0),(0,0,0))]
    for vertex in specialVs:
        if closePoints(coords, vertex.coords):
            return vertex
    return nearestVertex(coords, pattern)

def adjustCoords(coords, pattern):
    vertex = nearestVertexWithSpecialVs(coords, pattern)
    if vertex:
        return vertex.coords
    return coords

# from http://doughellmann.com/2010/03/pymotw-creating-xml-
# documents-with-elementtree.html
def prettify(elem):
    """ Return a pretty-printed XML string for the Element. """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def addFacesOnTop(faces, face, unmovedFace, newPattern, pattern, vertex, 
    newCoord, edge, toFront):
    if toFront:
        ran = range(pattern.faces.index(face))
    else:
        ran =  range(pattern.faces.index(face)+1, len(pattern.faces))
    for i in ran:
        if (pattern.faces[i] not in faces['originalFace'] and
            facesIntersect(pattern.faces[i], unmovedFace)):
            faces = addFaceAndNeighbors(faces, pattern,
                pattern.faces[i], vertex, newCoord, edge)
    return faces

def boxTest(vertex, face):
    xMin = xMax = yMin = yMax = None
    for v in face.vertices:
        if xMin == None:
            xMin = xMax = v.x
            yMin = yMax = v.y
        if xMin > v.x:
            xMin = v.x
        elif xMax < v.x:
            xMax = v.x
        if yMin > v.y:
            yMin = v.y
        elif yMax < v.y:
            yMax = v.y
    if not ((xMin < vertex.x < xMax) and (yMin < vertex.y < yMax)):
        return False
    else:
        return True

def rayCastingTest(vertex, face):
    x1 = x2 = None
    for edge in face.edges:
        intPoint = intersectionEdgeVector(edge, (vertex.coords, (1,0,0)))
        if intPoint:
            if x1 == None:
                x1 = intPoint.x
            else:
                if intPoint.x >= x1:
                    x2 = intPoint.x
                else:
                    x2 = x1
                    x1 = intPoint.x
                break
    if x1 == None or x2 == None:
        # the ray doesn't hit two edges
        return False
    if x1 < vertex.x < x2:
        # the vertex is between the two edges
        return True
    return False

# algorithm from 
# http://stackoverflow.com/questions/217578/point-in-polygon-aka-hit-test
def vertexOnFace(vertex, face):
    # first do the box test
    if not boxTest(vertex, face):
        return False
    
    # then the ray casting test (not exactly, as the faces are convex
    # so we know that we only have two sides hitting the ray at max)
    return rayCastingTest(vertex, face)

def facesIntersect(face1, face2):
    for v1 in face1.vertices:
        if vertexOnFace(v1, face2):
            return True
    for v2 in face2.vertices:
        if vertexOnFace(v2, face1):
            return True
    # we don't detect some intersections where vertices aren't involved
    # which could cause bugs
    return False



class Origami(object):
    def __init__(self, patterns=None, steps=None):
        if patterns == None:
            self.patterns = [Pattern()]
        else:
            self.patterns = patterns
        if steps == None:
            self.steps = []
        else:
            self.steps = steps
        self.draggedPattern = Pattern()
        self.animatedPattern = Pattern() # not drawn until .drawPattern()
        self.draw()
        self.currentPIndex = len(self.patterns)-1
        self.firstThumbnail = max(0, len(self.patterns)-7)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.patterns == other.patterns

    def fold(self, vertex, newCoord, save=True, foldToFront=True):
        newPattern = self.patterns[-1].copy()
        newCoord = adjustCoords(newCoord, newPattern)
        oldCoord = (vertex.x, vertex.y, vertex.z)
        
        if newCoord != oldCoord:
            vertex = vertex.copy()
            self.draggedPattern.frame.visible = False
            self.patterns[-1].frame.visible = False

            coords = 3
            middle = [(oldCoord[i]+newCoord[i])/2 for i in range(coords)]
            diff = [oldCoord[i]-newCoord[i] for i in range(coords)]
            # normal as (point vector, direction vector)
            normal = (middle, (-diff[1], diff[0], diff[2])) # rotated 90 degs
            
            affectedFaces = self.getAffectedFaces(vertex, newPattern, newCoord, 
                normal, foldToFront)
            
            newPattern = self.applyPatternChanges(newPattern, affectedFaces, 
                foldToFront, save, normal)
            newPattern.drawPattern()

    def getAffectedFaces(self, vertex, pattern, newCoord, normal, foldToFront):
        affectedFaces = {'movingPiece': [], 'stayingPiece': [],
            'originalFace': [], 'unmovedFaces': [], 'faceIndices': []}

        for face in self.patterns[-1].faces:
            if (face not in affectedFaces['originalFace'] and 
                vertex in face.vertices):
                affectedFaces = addFaceAndNeighbors(affectedFaces, pattern,
                    face, vertex, newCoord, normal)
                lastFace = face
                unmoved = affectedFaces['unmovedPiece']

        if lastFace:
            affectedFaces = addFacesOnTop(affectedFaces, lastFace, unmoved, 
                pattern, self.patterns[-1], vertex, newCoord, normal, 
                foldToFront)

        return affectedFaces

    def applyPatternChanges(self, pattern, affectedFaces, foldToFront, save, 
        normal):
        originalFace = affectedFaces['originalFace']
        unmovedFaces = affectedFaces['unmovedFaces']
        indices = affectedFaces['faceIndices']

        for face in originalFace:
            newFace = affectedFaces['stayingPiece'][originalFace.index(face)]
            if newFace:
                pattern.faces[pattern.faces.index(face)] = newFace
            else:
                pattern.faces.pop(pattern.faces.index(face))

        if foldToFront:
            pattern.faces = affectedFaces['movingPiece'][::-1] + pattern.faces
        else: # folding to back
            pattern.faces = pattern.faces + affectedFaces['movingPiece'][::-1]

        if save:
            self.patterns.append(pattern)
            self.steps.append(Step(indices, unmovedFaces, 
                affectedFaces['stayingPiece'], normal, foldToFront))
            self.currentPIndex += 1
            self.firstThumbnail = max(0, len(self.patterns)-7)
        else:
            self.draggedPattern = pattern
        return pattern

    def draw(self):
        self.patterns[-1].frame.visible = True
        self.patterns[-1].drawPattern()

    def remove(self):
        for pattern in self.patterns:
            pattern.remove()

    def undo(self):
        if len(self.patterns) > 1:
            self.patterns[-1].frame.visible = False
            self.patterns.pop(-1)
            self.draw()
            self.currentPIndex -= 1
            self.steps.pop(-1)
            self.firstThumbnail = max(0, self.firstThumbnail-1)

    def xml(self):
        origamiXml = ET.Element('origami')
        patternsXml = ET.SubElement(origamiXml, 'patterns')
        stepsXml = ET.SubElement(origamiXml, 'steps')
        for pattern in self.patterns:
            patternsXml = pattern.xml(patternsXml)
        for step in self.steps:
            stepsXml = step.xml(stepsXml)
        #ET.dump(origamiXml)
        return origamiXml

    @classmethod
    def loadXml(self, origamiXml):
        patterns = []
        for patternXml in origamiXml[0]: # patterns node
            patterns.append(Pattern.loadXml(patternXml))
        steps = []
        for stepXml in origamiXml[1]: # steps node
            steps.append(Step.loadXml(stepXml))
        return Origami(patterns, steps)

class Pattern(object):
    def __init__(self, faces=None):
        if faces == None:
            self.faces = [Face()]
        else:
            self.faces = faces
        self.frame = frame()
        self.drawn = False

    def __repr__(self):
        text = "<Pattern with\n  Faces:"
        for face in self.faces:
            text += "\n  %s" % str(face)
        text += "\n>"
        return text
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.faces == other.faces
    
    def drawPattern(self):
        self.drawn = True
        for i in range(len(self.faces)):
            if self.faces[i] != None:
                self.faces[i].drawFace(i, self.frame)
    
    def remove(self):
        self.frame.visible = False
        for face in self.faces:
            face.remove()

    def copy(self):
        return Pattern([face.copy() for face in self.faces])

    def xml(self, patternsXml):
        patternXml = ET.SubElement(patternsXml, 'pattern')
        for face in self.faces:
            patternXml = face.xml(patternXml)
        return patternsXml

    @classmethod
    def loadXml(self, patternXml):
        faces = []
        for faceXml in patternXml:
            faces.append(Face.loadXml(faceXml))
        return Pattern(faces)


class Face(object):
    def __init__(self, vertices=None):
        if vertices == None:
            length = 100.0 # from center of the origami to a vertex
            p1 = vector(length, 0.0, 0.0)
            p2 = vector(0.0, length, 0.0)
            p3 = vector(-length, 0.0, 0.0)
            p4 = vector(0.0, -length, 0.0)
            self.vertices = [Vertex(p, p) for p in [p1,p2,p3,p4]]
        else:
            self.vertices = vertices
        self.updateEdges()
        self.subfaces = []

    def __repr__(self):
        # With edges version
        text = "<Face with\n    Edges:"
        for edge in self.edges:
            text += "\n    %s" % str(edge)
        text += "\n    Vertices:"
        for vertex in self.vertices:
            text += "\n    %s" % str(vertex)
        text += "\n  >"
        # Vertices only version
        text = "<Face with\n    Vertices:"
        for vertex in self.vertices:
            text += "\n    %s" % str(vertex)
        text += "\n  >"
        return text

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return ((self.vertices, self.edges) == (other.vertices, other.edges))

    def updateEdges(self):
        edges = []
        for i in range(len(self.vertices)):
            edges.append(Edge(self.vertices[i], self.vertices[(i+1) %
                len(self.vertices)]))
        self.edges = edges

    def drawFace(self, index, frame):
        subfaces = []
        coeff = 0.2
        for i in range(len(self.vertices)-2):
            coords0 = list(self.vertices[0].coords)
            coords1 = list(self.vertices[i+1].coords)
            coords2 = list(self.vertices[i+2].coords)
            for coords in [coords0, coords1, coords2]:
                coords[2] += index*coeff
            f1 = faces( pos=[coords0, coords1, coords2],
                frame=frame, color=BACK_COLOR)
            f2 = faces( pos=[coords0, coords2, coords1],
                frame=frame, color=FRONT_COLOR)
            subfaces.extend([f1, f2])

        for edge in self.edges:
            edge.drawEdge(index, coeff, frame)

        self.subfaces = subfaces

    def remove(self):
        for i in range(len(self.subfaces)-1, -1, -1):
            subface = self.subfaces.pop(0)
            subface.visible = False
            del subface

    def copy(self):
        return Face([vertex.copy() for vertex in self.vertices])

    def xml(self, patternXml):
        faceXml = ET.SubElement(patternXml, 'face')
        for vertex in self.vertices:
            faceXml = vertex.xml(faceXml)
        return patternXml

    @classmethod
    def loadXml(self, faceXml):
        vertices = []
        for vertexXml in faceXml:
            vertices.append(Vertex.loadXml(vertexXml))
        return Face(vertices)

class Edge(object):
    def __init__(self, v1, v2):
        self.ends = [v1, v2]

    def __repr__(self):
        coords = 3
        return "<E ((%6.1f,%6.1f,%6.1f), (%6.1f,%6.1f,%6.1f))>" % tuple(
            self.ends[i].coords[j] for i in range(2) for j in range(coords))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.ends == other.ends

    def drawEdge(self, index, coeff, frame):
        coords = [[],[]]
        for i in range(2):
            for j in range(2):
                coords[i].append(self.ends[i].coords[j])
            coords[i].append(self.ends[i].coords[2]+index*coeff)
        curve(pos=coords, color=color.black, radius=coeff, frame=frame)

class Vertex(object):
    def __init__(self, coords, originalCoords):
        self.originalCoords = originalCoords
        self.coords = self.x, self.y, self.z = coords
    
    def __repr__(self):
        return ("<V on (%6.1f,%6.1f,%6.1f) from (%6.1f,%6.1f,%6.1f)>" % 
            (self.coords.x, self.coords.y, self.coords.z, 
            self.originalCoords[0], 
            self.originalCoords[1], 
            self.originalCoords[2]))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return almostSameCoords(self.originalCoords, other.originalCoords)

    def copy(self):
        return Vertex(self.coords, self.originalCoords)

    def xml(self, faceXml):
        vertexXml = ET.SubElement(faceXml, 'vertex')
        originalCoords = ['originalX', 'originalY', 'originalZ']
        coords = 3
        for i in range(coords):
            vertexXml.set(originalCoords[i], "%10.5f" % self.originalCoords[i])
        coords = ['x', 'y', 'z']
        for i in range(coords):
            vertexXml.set(coords[i], "%10.5f" % self.coords[i])
        return faceXml

    @classmethod
    def loadXml(self, vertexXml):
        v = vertexXml.attrib
        coords = vector([float(coord) for coord in [v['x'], v['y'], v['z']]])
        originalCoords = [v['originalX'], v['originalY'], v['originalZ']]
        originalCoords = [float(coord) for coord in originalCoords]
        return Vertex(coords, originalCoords)

def closeEnough(var1, var2):
    epsilon = 0.00001
    return abs(var1 - var2) < epsilon

def testCloseEnough():
    assert(closeEnough(1, 1.0000001))
    assert(not closeEnough(-1, -5))
    assert(not closeEnough(1, 1.001))

def almostSameCoords(coord1, coord2):
    return reduce(operator.mul, map(closeEnough, coord1, coord2))

def testAlmostSameCoords():
    assert(almostSameCoords((1, 2, 3), (1.00000001, 2.0, 3.000001)))
    assert(not almostSameCoords((1, 2, 3), (1, 2.001, 3)))
    assert(not almostSameCoords((-1, 2, 3), (1, 2, 3)))

def drawThumbPattern(pattern, i, gc, l, m, length):
    faces = pattern.faces
    center = vector(m*(i+1) + l*(i+1/2.0), l/2 + m, 0)
    for face in faces:
        edge1 = face.vertices[1].coords-face.vertices[0].coords
        edge2 = face.vertices[-1].coords-face.vertices[0].coords
        if cross(edge1, edge2).z > 0: # the angle from edge1 to 2 is negative
            gc.SetBrush(wx.Brush('white'))
        else:
            gc.SetBrush(wx.Brush('yellow'))
        path = gc.CreatePath()
        for i in range(len(face.vertices)):
            vertex = face.vertices[i]
            coords = vector(vertex.coords.x, -vertex.coords.y)
            length = 100 # from center of the origami to a vertex
            vPosition = center + coords/length*l/2
            if i == 0:
                path.MoveToPoint(vPosition.x, vPosition.y)
            else:
                path.AddLineToPoint(vPosition.x, vPosition.y)
        path.CloseSubpath()
        gc.DrawPath(path)

class Step(object):
    def __init__(self, faceIndices, movingFaces, stayingFaces, edge, direction):
        self.faceIndices = faceIndices
        self.movingFaces = movingFaces
        self.stayingFaces = stayingFaces
        self.edge = edge
        self.direction = direction

    def __repr__(self):
        text = "<Step with"
        text += "\n  Face Indices: " + str(self.faceIndices)
        text += "\n  Moving Faces: " + str(self.movingFaces)
        text += "\n  Staying Faces: " + str(self.stayingFaces)
        text += "\n  Edge: " + str(self.edge)
        text += "\n  Direction: " + str(self.direction)
        text += "\n>"
        return text

    def xml(self, stepsXml):
        stepXml = ET.SubElement(stepsXml, 'step')

        fIsXml = ET.SubElement(stepXml, 'faceIndices')
        for index in self.faceIndices:
            indexXml = ET.SubElement(fIsXml, 'index')
            indexXml.text = str(index)

        mFsXml = ET.SubElement(stepXml, 'movingFaces')
        for face in self.movingFaces:
            mFsXml = face.xml(mFsXml)

        sFsXml = ET.SubElement(stepXml, 'stayingFaces')
        # returns error if stayingFace is None
        for face in self.stayingFaces:
            if face == None: ET.SubElement(sFsXml, 'None')
            else: sFsXml = face.xml(sFsXml)

        edgeXml = ET.SubElement(stepXml, 'edge')
        point = ET.SubElement(edgeXml, 'point')
        axis = ET.SubElement(edgeXml, 'axis')
        coords = ['x', 'y', 'z']
        for i in range(len(coords)):
            point.set(coords[i], "%10.5f" % self.edge[0][i])
            axis.set(coords[i], "%10.5f" % self.edge[1][i])

        dirXml = ET.SubElement(stepXml, 'direction')
        dirXml.text = str(int(self.direction))

        return stepsXml
        
    @classmethod
    def loadXml(self, stepXml):
        fIs = []
        fIsXml = stepXml.find('faceIndices')
        for index in fIsXml:
            fIs.append(int(index.text))

        mFs = []
        mFsXml = stepXml.find('movingFaces')
        for faceXml in mFsXml:
            mFs.append(Face.loadXml(faceXml))

        sFs = []
        sFsXml = stepXml.find('stayingFaces')
        for faceXml in sFsXml:
            if faceXml.tag == 'None':
                sFs.append(None)
            else:
                sFs.append(Face.loadXml(faceXml))

        edgeXml = stepXml.find('edge')
        p = edgeXml.find('point').attrib
        a = edgeXml.find('axis').attrib
        strEdge = [[p['x'], p['y'], p['z']], [a['x'], a['y'], a['z']]]
        edge = [[float(elem) for elem in coords] for coords in strEdge]

        direction = int(stepXml.find('direction').text)

        return Step(fIs, mFs, sFs, edge, direction)

class MainLoop(object):
    def __init__(self):
        size, toolsHeight, heightMargin = self.getFrameSize()
        grayDarkness = 0.9 # 1 is black
        self.w = window(width=size, height=size+toolsHeight, 
            title="Origami Simulator")
        self.scene = display(window=self.w, x=0, y=0, width=size, height=size, 
            ambient=color.gray(grayDarkness), background = BACKGROUND_COLOR)

        self.initToolbar()
        # from http://www.wxpython.org/doriginalCoords/api/
        # wx.Window-class.html#SetSizeHints
        self.w.win.SetSizeHints(minW=size, minH=size+toolsHeight+heightMargin, 
            maxW=size, maxH=size+toolsHeight+heightMargin)

        # == THE THUMBNAIL PANEL ==
        panelHeight = 100
        self.panel = wx.Panel(parent=self.w.win, pos=(0,size), 
            size=(size,panelHeight))
        self.panel.Bind(wx.EVT_PAINT, lambda event: self.OnPaint(event))

        self.origami = Origami()
        self.mode = 1
        self.tb.EnableTool(self.editID, False) # disable edit
        self.tb.EnableTool(self.prevID, False) # disable prev
        self.tb.EnableTool(self.nextID, False) # disable next

        self.scene.bind('mousedown', self.mouseDown)
        self.scene.bind('keydown', self.keyPressed) 
        self.init()
        self.loop()

    def getFrameSize(self):
        if os.name == "posix": # OSX
            size = 600
            toolsHeight = 120
            heightMargin = 40
        else: # assuming it's Windows
            size = 625
            toolsHeight = 200
            heightMargin = 0
        return size, toolsHeight, heightMargin

    def initToolbar(self):
        # toolbar icons from http://twitpic.com/ad2eps
        self.tb = wx.Frame.CreateToolBar(self.w.win, style=wx.TB_TEXT)
        functions = [lambda e: self.init(), lambda e: self.loadOrigami(),
            lambda e: self.saveOrigami(), lambda e: self.switchToEditMode(),
            lambda e: self.switchToViewMode(), lambda e: self.prevStep(),
            lambda e: self.animateStep(), lambda e: self.undo(),
            lambda e: self.flip()]
        self.initToolbarButtons(functions)
        self.tb.Realize()

    def initToolbarButtons(self, functions):
        IDs = (initID, loadID, saveID, editID, viewID, undoID, prevID, nextID, 
            flipID) = (5001, 5002, 5003, 5004, 5005, 5008, 5006, 5007, 5009)

        initB=self.tb.AddLabelTool(initID,'New (n)',wx.Bitmap('img/init.png'))
        loadB=self.tb.AddLabelTool(loadID,'Open (o)',wx.Bitmap('img/load.png'))
        saveB=self.tb.AddLabelTool(saveID,'Save (s)',wx.Bitmap('img/save.png'))
        self.tb.AddSeparator()
        editB=self.tb.AddLabelTool(editID,'Edit Mode',wx.Bitmap('img/edit.png'))
        viewB=self.tb.AddLabelTool(viewID,'View Mode',wx.Bitmap('img/view.png'))
        self.tb.AddSeparator()
        undoB=self.tb.AddLabelTool(undoID,'Undo (u)',wx.Bitmap('img/undo.png'))
        prevB=self.tb.AddLabelTool(prevID,u'Prev (←)',wx.Bitmap('img/prev.png'))
        nextB=self.tb.AddLabelTool(nextID,u'Next (→)',wx.Bitmap('img/next.png'))
        flipB=self.tb.AddLabelTool(flipID,'Flip (f)',wx.Bitmap('img/flip.png'))

        buttons = [initB,loadB,saveB,editB,viewB,prevB,nextB,undoB,flipB]

        (self.initID, self.loadID, self.saveID, self.editID, self.viewID, 
            self.undoID, self.prevID, self.nextID, self.flipID) = [IDs[i] for
            i in range(len(IDs))]

        for i in range(len(buttons)):
            self.w.win.Bind(wx.EVT_TOOL, functions[i], buttons[i])

    def init(self):
        self.origami.remove()
        self.origami = Origami()
        self.scene.forward = (0,0,-1)
        self.scene.autoscale = False
        self.animating = False
        self.angle = 0
        self.panel.Refresh()

    def loop(self):
        # needed for running outside VIDLE
        while True:
            framesPerSec = 15
            rate(framesPerSec) # x frames per second at max
            if self.animating:
                if self.angle == 0:
                    self.animateStep()
                elif self.angle < pi:
                    self.animateRotation()
                else: # the angle is over pi
                    self.endRotation()

    def mouseDown(self, event):
        # project the coordinates of the click onto the plane of the origami
        coords = self.scene.mouse.project(normal=(0,0,1))
        vertex = nearestVertex(coords, self.origami.patterns[-1])
        if vertex != None:
            self.scene.bind('mousemove', lambda e: self.mouseDrag(e, vertex))
            self.scene.bind('mouseup', lambda e: self.mouseUp(e, vertex))

    def mouseDrag(self, event, vertex):
        # project the coordinates of the click onto the plane of the origami
        coords = self.scene.mouse.project(normal=(0,0,1))
        camAngle = self.scene.forward[2]
        if camAngle < 0:
            self.origami.fold(vertex, coords, save=False, foldToFront=False)
        else:
            self.origami.fold(vertex, coords, save=False, foldToFront=True)

    def mouseUp(self, event, vertex):
        # project the coordinates of the click onto the plane of the origami
        coords = self.scene.mouse.project(normal=(0,0,1))
        camAngle = self.scene.forward[2]
        if camAngle < 0:
            self.origami.fold(vertex, coords, save=True, foldToFront=False)
        else:
            self.origami.fold(vertex, coords, save=True, foldToFront=True)
        self.scene.unbind('mousemove')
        self.scene.unbind('mouseup')
        self.panel.Refresh() # for drawwing thumbnails

    def keyPressed(self, event):
        if self.mode == 1: # interactive
            if event.key == "i" or event.key == "n": # initialize origami
                self.init()
            if event.key == "u" or event.key == "z": # undo
                self.undo()
            if event.key == "s": # save
                self.saveOrigami()
            if event.key == "l" or event.key == "o": # load
                self.loadOrigami()
            if event.key == "2":
                self.switchToViewMode()
        else: # instructive
            if event.key == "right":
                if self.origami.currentPIndex < len(self.origami.patterns) - 1:
                    self.animateStep()
            if event.key == "left":
                if self.origami.currentPIndex > 0:
                    self.prevStep()
            if event.key == "1":
                self.switchToEditMode()
        if event.key == "f": # flip
            self.flip()

    def undo(self):
        self.origami.undo()
        self.panel.Refresh()

    def switchToEditMode(self):
        self.scene.bind('mousedown', self.mouseDown)
        self.mode = 1
        self.origami.animatedPattern.frame.visible = False
        self.origami.patterns[self.origami.currentPIndex].frame.visible = False
        self.origami.currentPIndex = len(self.origami.patterns)-1
        self.origami.patterns[-1].frame.visible = True
        self.origami.firstThumbnail = max(0, len(self.origami.patterns)-7)
        self.panel.Refresh()
        self.tb.EnableTool(self.initID, True) # enable init
        self.tb.EnableTool(self.loadID, True) # enable load
        self.tb.EnableTool(self.saveID, True) # enable save
        self.tb.EnableTool(self.editID, False) # disable edit
        self.tb.EnableTool(self.viewID, True) # enable view
        self.tb.EnableTool(self.prevID, False) # disable prev
        self.tb.EnableTool(self.nextID, False) # disable next 
        self.tb.EnableTool(self.undoID, True) # enable undo 

    def switchToViewMode(self):
        self.scene.unbind('mousedown')
        self.mode = 2
        self.tb.EnableTool(self.initID, False) # disable init
        self.tb.EnableTool(self.loadID, False) # disable load
        self.tb.EnableTool(self.saveID, False) # disable save
        self.tb.EnableTool(self.editID, True) # enable edit
        self.tb.EnableTool(self.viewID, False) # disable view
        if self.origami.currentPIndex > 0:
            self.tb.EnableTool(self.prevID, True) # enable prev
        self.tb.EnableTool(self.undoID, False) # disable undo

    # method adapted from http://wiki.wxpython.org/WxSmallApp
    def loadOrigami(self):
        dlg = wx.FileDialog(self.w.win,
            wildcard="XML Files (*.xml)|*.xml|All Files|*.*", style=wx.FD_OPEN)
        if (dlg.ShowModal() == wx.ID_OK):
            fileName = dlg.GetFilename()
            dirName = dlg.GetDirectory()

            self.origami.remove()

            try:
                xml = ET.parse(os.path.join(dirName, fileName)).getroot()
                self.origami = Origami.loadXml(xml)
            except:
                print "Error: invalid file!"
                self.origami = Origami()
            self.scene.forward = (0,0,-1)
            self.animating = False
            self.angle = 0
            self.panel.Refresh()
        dlg.Destroy()

    # method adapted from http://wiki.wxpython.org/WxSmallApp
    def saveOrigami(self):
        dlg = wx.FileDialog(self.w.win, "Save as a file",
            wildcard="XML Files (*.xml)|*.xml", style=wx.FD_SAVE
            |wx.FD_OVERWRITE_PROMPT)
        if (dlg.ShowModal() == wx.ID_OK):
            fileName = dlg.GetFilename()
            dirName = dlg.GetDirectory()

            origamiXml = self.origami.xml()
            f = open(os.path.join(dirName, fileName), 'w')
            f.write(prettify(origamiXml))
            f.close()
        dlg.Destroy()

    def flip(self):
        self.scene.forward *= -1

    def OnPaint(self, event):
        l = 75.0 # length (width/height) of the origami
        m = 10.0 # margins around thumbnails
        self.panel.SetMinSize = ((m+l)*len(self.origami.patterns), l+m)
        dc = wx.PaintDC(self.panel)
        gc = wx.GraphicsContext.Create(dc)
        gc.SetPen(wx.Pen('black'))
        path = gc.CreatePath()
        numOfThumbs = 7
        for i in range(min(len(self.origami.patterns), numOfThumbs)):
            j = i + self.origami.firstThumbnail
            drawThumbPattern(self.origami.patterns[j], i, gc, l, m,
                len(self.origami.patterns))

    def animateStep(self):
        self.animating = True
        if self.angle == 0: # to avoid speeding up
            deltaAngle = 0.2 # the change in the angle in each loop
            self.angle += deltaAngle
        numOfThumbs = 7
        if (self.origami.firstThumbnail+numOfThumbs-1 < 
            self.origami.currentPIndex+1):
            self.origami.firstThumbnail += 1
            self.panel.Refresh()
        if self.origami.currentPIndex+2 >= len(self.origami.patterns):
            self.tb.EnableTool(self.nextID, False) # disable next

    def animateRotation(self):
        prevPFrame = self.origami.patterns[self.origami.currentPIndex].frame
        if prevPFrame.visible:
            prevPFrame.visible = False
        self.origami.animatedPattern.frame.visible = False
        cPI = self.origami.currentPIndex
        step = self.origami.steps[cPI]
        faces = [face.copy() for face in self.origami.patterns[cPI].faces]
        for i in step.faceIndices:
            faces[i] = step.stayingFaces[step.faceIndices.index(i)]
        # rotate the movingFaces
        rotatedFaces = rotateFaces(step.movingFaces, step.edge, self.angle,
            step.direction)

        # add the rotated faces to faces list
        if step.direction: # folding to front
            faces = rotatedFaces[::-1] + faces # [::-1] for reversing the order
        else:
            faces = faces + rotatedFaces[::-1]

        self.origami.animatedPattern = Pattern(faces)
        self.origami.animatedPattern.drawPattern()
        deltaAngle = 0.2 # the change in the angle in each loop
        self.angle += deltaAngle

    def endRotation(self):
        self.origami.animatedPattern.frame.visible = False
        self.origami.currentPIndex += 1
        self.origami.patterns[self.origami.currentPIndex].frame.visible = True
        self.animating = False
        self.angle = 0
        if self.origami.currentPIndex > 0:
            self.tb.EnableTool(self.prevID, True) # enable prev

    def prevStep(self):
        self.origami.patterns[self.origami.currentPIndex].frame.visible = False
        self.origami.currentPIndex  -= 1
        if self.origami.firstThumbnail > self.origami.currentPIndex:
            self.origami.firstThumbnail -= 1
            self.panel.Refresh()
        cPI = self.origami.currentPIndex
        self.origami.patterns[cPI].frame.visible = True
        if not self.origami.patterns[cPI].drawn:
            self.origami.patterns[cPI].drawPattern()
        if self.origami.currentPIndex <= 0:
            self.tb.EnableTool(self.prevID, False) # disable prev
        if self.origami.currentPIndex+1 < len(self.origami.patterns):
            self.tb.EnableTool(self.nextID, True) # enable next

def rotateFaces(faces, line, angle, direction):
    point = vector(line[0]) # a point on the line
    axis = vector(line[1]) # the direction of the line
    rotatedFaces = []
    theta = 0.05 # the difference between faces to avoid z-fighting
    for face in faces:
        rotatedVs = []
        for vertex in face.vertices:
            diff = vertex.coords - point
            if direction: # fold to front
                rotatedDiff = rotate(diff, angle-theta*faces.index(face), axis)
            else:
                rotatedDiff = rotate(diff, -angle-theta*faces.index(face), axis)
            rotatedV = Vertex(point+rotatedDiff, vertex.originalCoords)
            rotatedVs.append(rotatedV)
        rotatedFace = Face(rotatedVs)
        rotatedFaces.append(rotatedFace)
    return rotatedFaces

mainLoop = MainLoop()


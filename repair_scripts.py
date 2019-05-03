from skimage import io
from skimage import draw
from skimage.morphology import skeletonize
from scipy import ndimage as nd
import numpy as np
from PIL import Image
import math, time, random
from operator import itemgetter
from math import atan2
THR_distance = 20
THR_angle = 35
CONST_k = 5

'''
'all great ideas are simple - how come there aren't more of them?
well, because frequently, that simplicity involves finding a couple tricks and making
a couple of observations. so usually we humans hardly ever go beyond one trick or one
observation, but if you cascade a few together, sometimes something miraculous falls out that
looks in retrospect extremely simple.' - prof winston
'''

#---------------------------------------------[step 1 - 1]--------------------------------------------#
#
#   - convert canny edges to bitmap ([0, 255] -> [0, 1]),
#   - thin bitmap

def to_bitmap(im):
    height, width = im.shape
    bitmap = np.zeros_like(im, dtype=int)
    for n in range(0, height):
        for m in range(0, width):
            value = im[n, m]
            if value > 0:
                bitmap[n, m] = 1
    return bitmap

def thin(im_bitmap):
    im_thin = skeletonize(im_bitmap)
    return im_thin
#-----------------------------------------------------------------------------------------------------#

#---------------------------------------------[step 1 - 2]--------------------------------------------#
#
#   - build strips: join all edge pixels belonging to an
#     adjacent, branchless region together in an array

def link_strips(thinned_bitmap):
    ends, intermediates, order_map = label_edges(thinned_bitmap)
    strips, loops = grow_strips(ends, intermediates, order_map)
    return strips, loops
'''
exports an array similar to the edge map with a value assigned to
each edge pixel coordinate based on the number of adjacent edge pixels
'''
def label_edges(bitmap):
    height, width = bitmap.shape
    #edge points with two neighbors: intermediates between edges (in a thinned edge map)
    intermediates = []
    #edge maps with 1 or >2 neighbors: strip ends and branching points
    ends = []
    #non-edge points identified by the -1 value
    order_map = np.full_like(bitmap, -1, dtype=tuple)
    for n in range(0, height):
        for m in range(0, width):
            point = (n, m)
            #if point is edge
            if bitmap[n, m] != 0:
                neighbors = adjacent(point, bitmap)
                order = len(neighbors)
                if order == 2:
                    #intermediate point
                    order_map[n, m] = (order, neighbors)
                    intermediates.append(point)
                elif order == 0:
                    #isolated point - ignore
                    pass
                else:
                    #end point
                    order_map[n, m] = (order, neighbors)
                    ends.append(point)
    return ends, intermediates, order_map

#calculate number of adjacent edge pixels
def adjacent(point, bitmap):
    adjacent_edges = []
    height, width = bitmap.shape
    y, x = point
    xmin = x - 1
    ymin = y - 1
    xmax = x + 1
    ymax = y + 1
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > width - 1:
        xmax = width - 1
    if ymax > height - 1:
        ymax = height - 1
    adjacent_points = [(ymin, xmin), (ymin, x), (ymin, xmax),
                       (   y, xmin),            (   y, xmax),
                       (ymax, xmin), (ymax, x), (ymax, xmax)]
    for candidate in adjacent_points:
        n, m = candidate
        value = bitmap[n, m]
        if value != 0:
            adjacent_edges.append(candidate)
    return adjacent_edges

#return continuous loops of edge pixels and discontinuous strips
def grow_strips(ends, intermediates, order_map):
    #value of 1 if point y, x has been previously iterated, otherwise 0.
    used_map = np.full_like(order_map, 0, dtype=int)
    #extracting strips
    strips = []
    for end in ends:
        strip = []
        next_point = end
        while next_point:
            y, x = next_point
            used_map[y, x] = 1
            try:
                intermediates.remove(next_point)
            except ValueError: pass
            next_point = None
            order, neighbors = order_map[y, x]
            strip.append((y, x))
            for point in neighbors:
                n, m = point
                #checking for end of loop
                if used_map[n, m] == 0 and order_map[n, m][0] == 2:
                    next_point = point
                    break
        if len(strip) > 1:
            strips.append(strip)
    #extracting loops
    loops = []
    l = len(intermediates)
    if l > 0:
        for i in range(0, l):
            loop = []
            current_point = intermediates[i]
            y, x = current_point
            if used_map[y, x] == 0 and order_map[y, x][0] == 2:
                next_point = current_point
            while next_point:
                y, x = next_point
                used_map[y, x] = 1
                next_point = None
                order, neighbors = order_map[y, x]
                loop.append((y, x))
                for point in neighbors:
                    n, m = point
                    if used_map[n, m] == 0 and order_map[n, m][0] == 2:
                        next_point = point
                        break
            if len(loop) > 1:
                loops.append(loop)
    return strips, loops
#-----------------------------------------------------------------------------------------------------#

#---------------------------------------------[step 1 - 3]--------------------------------------------#
#
# extract lines from strips: split edge pixel strips
# into lines which represent its original curvature

def extract_lines(strips, THR_short = 90, THR_long = 10):
    len_min = 5
    len_long = 10
    
    strip_line_segments = []
    for strip in strips:
        line_segments = []
        line = []
        len_strip = len(strip)
        next_point = strip[0]
        for i in range(1, len_strip):
            point = strip[i]
            line.append(point)
            len_line = len(line)
            #finding the first, last, and middle point coordinates
            y_1, x_1 = line[0]
            y_2, x_2 = line[math.floor(len_line / 2)]
            y_3, x_3 = line[len_line - 1]
            #calculating slope between first & middle and middle & last
            degree_1 = (180/math.pi) * atan2((y_1 - y_2), (x_1 - x_2))
            degree_2 = (180/math.pi) * atan2((y_2 - y_3), (x_2 - x_3))
            difference = abs(degree_1 - degree_2)
            '''
            a line is identified once the difference between
            slope one and slope two exceeds a threshold, dependent
            on the length of the line. when the difference exceeds
            this threshold, the current line (an array of points)
            is added to a list of line segments, then cleared,
            and the algorithm proceeds. this process creates an
            array of line segments which accurately represent
            the original curvature of the strip.
            '''
            if len_line < len_long and len_line > len_min:
                if difference > THR_short:
                    #print('short: ' + str(difference))
                    end_one = line[0]
                    end_two = line[len(line) - 1]
                    line_segments.append((end_one, end_two))
                    line = []
            else:
                if difference > THR_long and len_line > len_min:
                    #print('long: ' + str(difference))
                    end_one = line[0]
                    end_two = line[len(line) - 1]
                    line_segments.append((end_one, end_two))
                    line = []
        if len(line) > 1:
            end_one = line[0]
            end_two = line[len(line) - 1]
            line_segments.append((end_one, end_two))
        strip_line_segments.append(line_segments)
    return strip_line_segments
#-----------------------------------------------------------------------------------------------------#

#---------------------------------------------[step 2 - 1]--------------------------------------------#
#
#   - returns tuples of all line segments within the threshold distance

def pair_lines(edge_strip_lines, THR_distance):
    pairs = []
    esl = edge_strip_lines.copy()
    esl_copy = edge_strip_lines.copy()
    
    #O(1/2 n^2), n = number of strips
    i = 0
    while i < len(esl):
        strip_one = esl[i]
        #remove reduntant iteration by removing this element from the second iterable
        esl_copy.remove(strip_one)
        if len(strip_one) < 1:
            i += 1
            continue
        j = 0
        while j < len(esl_copy):
            strip_two = esl_copy[j]
            if len(strip_two) < 1:
                j += 1
                continue
            
            #lines on either end of strips
            endlineone_1 = strip_one[0]
            endlineone_2 = strip_one[len(strip_one) - 1]
            endlinetwo_1 = strip_two[0]
            endlinetwo_2 = strip_two[len(strip_two) - 1]
            #four end points
            p11 = endlineone_1[0]
            p12 = endlineone_2[len(endlineone_1) - 1]
            p21 = endlinetwo_1[0]
            p22 = endlinetwo_2[len(endlinetwo_2) - 1]
            #distances
            d11_21 = math.sqrt(math.pow(p11[0] - p21[0], 2)
                               + math.pow(p11[1] - p21[1], 2))
            d11_22 = math.sqrt(math.pow(p11[0] - p22[0], 2)
                               + math.pow(p11[1] - p22[1], 2))
            d12_21 = math.sqrt(math.pow(p12[0] - p21[0], 2)
                               + math.pow(p12[1] - p21[1], 2))
            d12_22 = math.sqrt(math.pow(p12[0] - p22[0], 2)
                               + math.pow(p12[1] - p22[1], 2))
            if d11_21 < THR_distance:
                end_pair = (p11, p21)
                strip_pair = (strip_one, strip_two)
                composite_pair = (end_pair, strip_pair, d11_21)
                pairs.append(composite_pair)
                
            if d11_22 < THR_distance:
                end_pair = (p11, p22)
                strip_pair = (strip_one, strip_two)
                composite_pair = (end_pair, strip_pair, d11_22)
                pairs.append(composite_pair)
                
            if d12_21 < THR_distance:
                end_pair = (p12, p21)
                strip_pair = (strip_one, strip_two)
                composite_pair = (end_pair, strip_pair, d12_21)
                pairs.append(composite_pair)
                
            if d12_22 < THR_distance:
                end_pair = (p12, p22)
                strip_pair = (strip_one, strip_two)
                composite_pair = (end_pair, strip_pair, d12_22)
                pairs.append(composite_pair)

            j += 1
        
        i += 1
    return pairs
#-----------------------------------------------------------------------------------------------------#

#---------------------------------------------[step 2 - 2]--------------------------------------------#
#
#   - calculate merging degree (very rough calculation of confidence in likelihood of pair to be merged)
#     and sort pairs accordingly

def sort_pairs(pairs, CONST_k):
    labeled_pairs = []
    for pair in pairs:
        end_pair, strip_pair, distance = pair
        strip1, strip2 = strip_pair
        aor1 = avg_rotation(strip1, distance, CONST_k)
        aor2 = avg_rotation(strip2, distance, CONST_k)
        aor_pair = (aor1, aor2)
        abs_dif = abs(aor1 - aor2)
        merging_degree = 0
        try:
            merging_degree = 1 / (distance * abs_dif)
        except ZeroDivisionError:
            pass
        labeled_pairs.append((merging_degree, end_pair, strip_pair, aor_pair))
    sorted_pairs = sorted(labeled_pairs, key=itemgetter(0))
    sorted_pairs.reverse()
    return sorted_pairs
'''
sum of the angles of each line segment within calculating distance of the
first point in an edge strip, divided by the number of iterated line segments 
'''
def avg_rotation(strip, distance, CONST_k):
    calculating_distance = CONST_k * distance
    n = 0
    aor = 0

    sum_dtheta = 0
    cumulative_distance = 0
    for i in range(0, len(strip)):
        try:
            n += 1
            line1 = strip[i]
            start1 = line1[0]
            end1 = line1[len(line1) - 1]
            len1_x = start1[1] - end1[1]
            len1_y = start1[0] - end1[0]
            length1 = math.sqrt(math.pow(len1_x, 2)
                               + math.pow(len1_y, 2))
            line2 = strip[i+1]
            start2 = line2[0]
            end2 = line2[len(line2) - 1]
            len2_x = start2[1] - end2[1]
            len2_y = start2[0] - end2[0]
            
            theta1 = math.atan2(len1_y, len1_x)
            theta2 = math.atan2(len2_y, len2_x)
            dtheta = theta1 - theta2
            
            cumulative_distance += length1
            sum_dtheta += dtheta
            
            if cumulative_distance >= calculating_distance:
                break
        except IndexError:
            break
    aor = sum_dtheta/n
    return aor
#-----------------------------------------------------------------------------------------------------#

#---------------------------------------------[step 2 - 3]--------------------------------------------#
#
#   - descending through sorted pairs by merging degree, perform merge
#     (joining with a line) when merge line connects strips within the threshold angle

def merge_pairs(sorted_pairs, THR_angle):
    merge_lines = []
    already_merged = dict()
    for pair_data in sorted_pairs:
        merging_degree, end_pair, strip_pair, aor_pair = pair_data
        end_1, end_2 = end_pair
        aor_1, aor_2 = aor_pair
        
        merge_x = end_1[1] - end_2[1]
        merge_y = end_1[0] - end_2[0]
        merge_angle = math.atan2(merge_y, merge_x)
        
        dif_1 = abs(aor_1 - merge_angle)
        dif_2 = abs(aor_2 - merge_angle)
        dif_total = dif_1 + dif_2
        check_1 = False
        check_2 = False
        try:
            check_1 = already_merged[end_1]
        except KeyError: pass
        try:
            check_2 = already_merged[end_2]
        except KeyError: pass
        if dif_total < THR_angle and not (check_1 or check_2):
            already_merged[end_1] = True
            already_merged[end_2] = True
            y1, x1 = end_1
            y2, x2 = end_2
            line_y, line_x = draw.line(y1, x1, y2, x2)
            line_points = []
            for i in range(0, len(line_y)):
                line_points.append((line_y[i], line_x[i]))
            merge_lines.append(line_points)
    return merge_lines
#-----------------------------------------------------------------------------------------------------#

def process(canny, thr_dist, thr_angle, const_k):
    bitmap = to_bitmap(canny)
    bitmap = thin(bitmap)
    strips, loops = link_strips(bitmap)
    line_segments = extract_lines(strips)
    pairs = pair_lines(line_segments, thr_dist)
    sorted_pairs = sort_pairs(pairs, const_k)
    repairs = merge_pairs(sorted_pairs, thr_angle)
    return bitmap, repairs, line_segments, strips, loops

def get_loops(canny):
    bitmap = to_bitmap(canny)
    bitmap = thin(bitmap)
    strips, loops = link_strips(bitmap)
    return loops

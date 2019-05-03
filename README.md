# Summary
modified the approach outlined in "A Fast and Robust Ellipse Detection-Method ..." (https://www.researchgate.net/publication/261999152_A_Fast_and_Robust_Ellipse-Detection_Method_Based_on_Sorted_Merging) to use line segments and calculated curvature to repair errors in edge maps caused by noise or occlusion.
# Modules
using numpy, scipy, scikit, and PIL.
# Use
This script was designed to be applied to canny edge maps.
Three constants adjust the script's behavior. The distance threshold, THR_distance, is the maximum distance for which two broken edge segments may be repaired into one. The angle threshold, THR_angle, is the maximum difference between the angle of the line repairing two broken edge segments and the angles of each edge segment. CONST_K is used in calculating the average angle of rotation (an estimate of curvature) of a strip of edge segments. The average angle of rotation is calculated by iterating across a strip of edge segments from the 'broken' - first or last - element in the array, until the calculating distance is reached. The calculating distance is the product of CONST_K and the length of the original broken elment.

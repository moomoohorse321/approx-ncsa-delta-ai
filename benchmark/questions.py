import os
import subprocess
import re
from typing import Callable, List, Optional, Any, Tuple, Set

class Question:
    """
    Describes a question and its specific function for computing accuracy. [cite: 390]
    """
    def __init__(self, text: str, accuracy_fn: Callable[[str], float]):
        self.text = text
        self.accuracy_fn = accuracy_fn
        
################## start grading #########################

def _find_best_match_float(text: str, expected_val: float) -> float | None:
    """
    Finds all numbers in the text and returns the one closest to the expected value.
    This makes the parsing tolerant to conversational text from the LLM.
    """
    # This regex correctly identifies integers and floating-point numbers.
    found_numbers = [float(num) for num in re.findall(r'-?\d+(?:\.\d+)?', text)]
    if not found_numbers:
        return None

    # Find the number in the list that is closest to the expected value
    best_match = min(found_numbers, key=lambda val: abs(val - expected_val))
    return best_match

# --- Specific Accuracy Functions ---
REL_TOLERANCE = 0.1  # 10% relative tolerance

def _find_best_match_float(text: str, expected_val: float) -> float | None:
    """
    Finds all numbers in the text and returns the one closest to the expected value.
    This makes the parsing tolerant to conversational text from the LLM.
    """
    # This regex correctly identifies integers and floating-point numbers.
    found_numbers = [float(num) for num in re.findall(r'-?\d+(?:\.\d+)?', text)]
    if not found_numbers:
        return None

    # Find the number in the list that is closest to the expected value
    best_match = min(found_numbers, key=lambda val: abs(val - expected_val))
    return best_match

# --- Specific Accuracy Functions ---

def acc_kmeans_2m_smallest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the smallest cluster (2M points) is found (relative tolerance), 0.0 otherwise."""
    # Ground truth: Cluster 9: 163677 points
    expected_val = 163677.0
    
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None:
        return 0.0
    
    # Using relative tolerance
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_kmeans_2m_bottom_left_pair(answer: str) -> float:
    """Accuracy is 1.0 if the bottom-left centroid (2M points) is found (relative tolerance)."""
    # Ground truth: Centroid 1: (17.37, 11.86)
    # (This is closest to (0,0) based on L1 norm: 17.37 + 11.86 = 29.23)
    expected_pair = (17.37, 11.86)
    
    found_centroids = re.findall(r"\(([\d.]+),\s*([\d.]+)\)", answer)
    parsed_set = {(float(x), float(y)) for x, y in found_centroids}
    
    # Check if any found tuple matches the expected pair within relative tolerance
    for p_c in parsed_set:
        # 10% L1 norm tolerance for x and y axes
        tol_x = REL_TOLERANCE * abs(expected_pair[0]) + 1e-9
        tol_y = REL_TOLERANCE * abs(expected_pair[1]) + 1e-9
        if abs(p_c[0] - expected_pair[0]) < tol_x and abs(p_c[1] - expected_pair[1]) < tol_y:
            return 1.0
            
    return 0.0

def acc_kmeans_all_centroids_strict(answer: str) -> float:
    """Accuracy is 1.0 if all 10 centroids are found (relative tolerance), 0.0 otherwise."""
    ground_truth_centroids: Set[Tuple[float, float]] = {
        (82.54, 84.45), (47.23, 15.19), (45.17, 56.70), (15.88, 15.40),
        (15.83, 82.18), (63.97, 36.69), (84.65, 15.71), (15.91, 46.67),
        (49.36, 85.72), (85.45, 53.82)
    }
    found_centroids = re.findall(r"\(([\d.]+),\s*([\d.]+)\)", answer)
    parsed_set = {(float(x), float(y)) for x, y in found_centroids}

    matches = 0
    for gt_c in ground_truth_centroids:
        # Check if any found centroid matches the ground truth centroid within the relative tolerance
        if any(abs(p_c[0] - gt_c[0]) < (REL_TOLERANCE * abs(gt_c[0]) + 1e-9) and \
               abs(p_c[1] - gt_c[1]) < (REL_TOLERANCE * abs(gt_c[1]) + 1e-9) for p_c in parsed_set):
            matches += 1
    
    return 1.0 if matches == 10 else 0.0

def acc_kmeans_largest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the largest cluster is found, 0.0 otherwise."""
    expected_val = 112292.0
    
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None:
        return 0.0
    
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_kmeans_smallest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the smallest cluster is found, 0.0 otherwise."""
    expected_val = 78448.0
    
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None:
        return 0.0
    
    # Expect an exact match for an integer value
    return 1.0 if abs(actual_val - expected_val) < 1e-6 else 0.0

def acc_kmeans_bottom_left_pair(answer: str) -> float:
    """Accuracy is 1.0 if the specific centroid (15.88, 15.40) is found (relative tolerance)."""
    expected_pair = (15.88, 15.40)
    
    found_centroids = re.findall(r"\(([\d.]+),\s*([\d.]+)\)", answer)
    parsed_set = {(float(x), float(y)) for x, y in found_centroids}
    
    # Check if any found tuple matches the expected pair within relative tolerance
    for p_c in parsed_set:
        tol_x = REL_TOLERANCE * abs(expected_pair[0]) + 1e-9
        tol_y = REL_TOLERANCE * abs(expected_pair[1]) + 1e-9
        if abs(p_c[0] - expected_pair[0]) < tol_x and abs(p_c[1] - expected_pair[1]) < tol_y:
            return 1.0
            
    return 0.0

def acc_kmeans_leftmost_strict(answer: str) -> float:
    """Accuracy is 1.0 if the x-coordinate of the leftmost centroid is found (relative tolerance), 0.0 otherwise."""
    expected_x = 15.83  # Leftmost point
    
    actual_val = _find_best_match_float(answer, expected_x)
    if actual_val is None:
        return 0.0
    
    tolerance = REL_TOLERANCE * abs(expected_x) + 1e-9
    return 1.0 if abs(actual_val - expected_x) < tolerance else 0.0

def acc_lavamd_avg_energy(answer: str) -> float:
    """Accuracy for the average potential energy (relative tolerance, 0/1 check)."""
    expected_energy = 1322.079678
    
    actual_val = _find_best_match_float(answer, expected_energy)
    if actual_val is None:
        return 0.0
    
    tolerance = REL_TOLERANCE * abs(expected_energy) + 1e-9
    return 1.0 if abs(actual_val - expected_energy) < tolerance else 0.0


def acc_lavamd_boxes8_avg_energy(answer: str) -> float:
    """Accuracy for the average potential energy (boxes1d = 8 run, relative tolerance, 0/1 check)."""
    expected_val = 1254.070689  # Average
    
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None:
        return 0.0
    
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_lavamd_boxes10_max_force_z(answer: str) -> float:
    """Accuracy for the max force vector (z) (boxes1d = 10 run, relative tolerance, 0/1 check)."""
    expected_val = 1766.183018  # Max
    
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None:
        return 0.0
    
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_lavamd_boxes8_particles_per_box(answer: str) -> float:
    """Accuracy for 'Particles per Box' (boxes1d = 8 run)."""
    # Value from boxes1d=8 simulation statistics
    expected_val = 128.0
    
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None:
        return 0.0
    
    # Exact match required for this integer value
    return 1.0 if abs(actual_val - expected_val) < 1e-6 else 0.0

def acc_lavamd_boxes10_particle42_potential(answer: str) -> float:
    """Accuracy for the potential of particle 42 (boxes1d = 10 run, relative tolerance, 0/1 check)."""
    expected_val = 382.914775  # Potential (v) for Particle Index 42
    
    actual_val = _find_best_match_float(answer, expected_val)
    
    if actual_val is None:
        return 0.0
        
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_sum_puzzle(answer: str) -> float:
    """
    Accuracy for the math puzzle. The expected answer is 22.
    (2 + 5 + 10 + 3 + 2 'g's in 'piggy' = 22)
    """
    # Use regex to find the whole number '22' to avoid matching '122', etc.
    if re.search(r'\b22\b', answer):
        return 1.0
    return 0.0

def acc_siebel_college(answer: str) -> float:
    """
    Accuracy for the Thomas Siebel question. He attended the University of Illinois.
    Checks for variations like 'University of Illinois', 'Urbana-Champaign', or 'UIUC'.
    """
    lower_answer = answer.lower()
    if ('university of illinois' in lower_answer or 
        'urbana-champaign' in lower_answer or 
        'uiuc' in lower_answer):
        return 1.0
    return 0.0

def acc_nq(answer: str, expected_answers: List[str]) -> float:
    """
    Accuracy for the non-tool questions.
    """
    
    return 1.0 if 'nq' in answer.lower() else 0.0


def acc_kmeans_2m_largest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the largest cluster (2M points) is found (relative tolerance), 0.0 otherwise."""
    # Ground truth: Cluster 6: 232478 points
    expected_val = 232478.0
    
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None:
        return 0.0
    
    # Using relative tolerance
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_kmeans_2m_all_centroids_strict(answer: str) -> float:
    """Accuracy is 1.0 if all 10 centroids (2M points) are found (relative tolerance), 0.0 otherwise."""
    # Ground truth from the 2M run
    ground_truth_centroids: Set[Tuple[float, float]] = {
        (48.24, 53.73), (17.37, 11.86), (19.07, 87.21), (82.03, 48.69), (53.59, 84.63),
        (84.94, 82.41), (51.85, 18.05), (18.74, 36.37), (84.22, 16.01), (14.82, 62.09)
    }
    found_centroids = re.findall(r"\(([\d.]+),\s*([\d.]+)\)", answer)
    parsed_set = {(float(x), float(y)) for x, y in found_centroids}

    matches = 0
    for gt_c in ground_truth_centroids:
        # Check if any found centroid matches the ground truth centroid within the relative tolerance
        if any(abs(p_c[0] - gt_c[0]) < (REL_TOLERANCE * abs(gt_c[0]) + 1e-9) and \
               abs(p_c[1] - gt_c[1]) < (REL_TOLERANCE * abs(gt_c[1]) + 1e-9) for p_c in parsed_set):
            matches += 1
    
    return 1.0 if matches == 10 else 0.0

def acc_kmeans_2m_leftmost_x(answer: str) -> float:
    """Accuracy is 1.0 if the x-coordinate of the leftmost centroid (2M points) is found (relative tolerance), 0.0 otherwise."""
    # Ground truth: Centroid 9: (14.82, 62.09)
    expected_x = 14.82
    
    actual_val = _find_best_match_float(answer, expected_x)
    if actual_val is None:
        return 0.0
    
    tolerance = REL_TOLERANCE * abs(expected_x) + 1e-9
    return 1.0 if abs(actual_val - expected_x) < tolerance else 0.0

def acc_kmeans_2m_top_right_pair(answer: str) -> float:
    """Accuracy is 1.0 if the top-right centroid (2M points) is found (relative tolerance)."""
    # Ground truth: Centroid 5: (84.94, 82.41)
    # (This is closest to (100, 100) based on L1 norm)
    expected_pair = (84.94, 82.41)
    
    found_centroids = re.findall(r"\(([\d.]+),\s*([\d.]+)\)", answer)
    parsed_set = {(float(x), float(y)) for x, y in found_centroids}
    
    # Check if any found tuple matches the expected pair within relative tolerance
    for p_c in parsed_set:
        tol_x = REL_TOLERANCE * abs(expected_pair[0]) + 1e-9
        tol_y = REL_TOLERANCE * abs(expected_pair[1]) + 1e-9
        if abs(p_c[0] - expected_pair[0]) < tol_x and abs(p_c[1] - expected_pair[1]) < tol_y:
            return 1.0
            
    return 0.0

def acc_lavamd_boxes11_particles_per_box(answer: str) -> float:
    """Accuracy for 'Particles per Box' (boxes1d = 11 run)."""
    # Value from boxes1d=11 simulation statistics
    expected_val = 128.0
    
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None:
        return 0.0
    
    # Exact match required for this integer value
    return 1.0 if abs(actual_val - expected_val) < 1e-6 else 0.0

def acc_lavamd_boxes11_avg_potential(answer: str) -> float:
    """Accuracy for the average potential energy (boxes1d = 11 run, relative tolerance, 0/1 check)."""
    # Ground truth: Potential Energy (v) | Average | 1348.530474
    expected_val = 1348.530474
    
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None:
        return 0.0
    
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_lavamd_boxes11_max_force_z(answer: str) -> float:
    """Accuracy for the max force vector (z) (boxes1d = 11 run, relative tolerance, 0/1 check)."""
    # Ground truth: Force Vector (z) | Max | 1763.076327
    expected_val = 1763.076327
    
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None:
        return 0.0
    
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_lavamd_boxes11_particle42_potential(answer: str) -> float:
    """Accuracy for the potential of particle 42 (boxes1d = 11 run, relative tolerance, 0/1 check)."""
    # Ground truth: Particle Index 42 | Potential (v) | 384.098276
    expected_val = 384.098276
    
    actual_val = _find_best_match_float(answer, expected_val)
    
    if actual_val is None:
        return 0.0
        
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

# --- K-means 1.2M functions ---
def acc_kmeans_1m2_all_centroids(answer: str) -> float:
    """Accuracy is 1.0 if all 10 centroids (1.2M points) are found (relative tolerance), 0.0 otherwise."""
    ground_truth_centroids: Set[Tuple[float, float]] = {
        (38.09, 18.33), (16.29, 83.12), (82.99, 83.52), (12.60, 16.48), (83.93, 50.63),
        (63.32, 18.43), (49.23, 84.52), (88.00, 16.62), (50.71, 52.71), (16.88, 50.09)
    }
    found_centroids = re.findall(r"\(([\d.]+),\s*([\d.]+)\)", answer)
    parsed_set = {(float(x), float(y)) for x, y in found_centroids}
    matches = 0
    for gt_c in ground_truth_centroids:
        if any(abs(p_c[0] - gt_c[0]) < (REL_TOLERANCE * abs(gt_c[0]) + 1e-9) and \
               abs(p_c[1] - gt_c[1]) < (REL_TOLERANCE * abs(gt_c[1]) + 1e-9) for p_c in parsed_set):
            matches += 1
    return 1.0 if matches == 10 else 0.0

def acc_kmeans_1m2_smallest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the smallest cluster (1.2M points) is found (relative tolerance)."""
    # Ground truth: Cluster 7: 98804 points
    expected_val = 98804.0
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_kmeans_1m2_largest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the largest cluster (1.2M points) is found (relative tolerance)."""
    # Ground truth: Cluster 2: 133638 points
    expected_val = 133638.0
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

# --- K-means 1.6M functions ---
def acc_kmeans_1m6_all_centroids(answer: str) -> float:
    """Accuracy is 1.0 if all 10 centroids (1.6M points) are found (relative tolerance), 0.0 otherwise."""
    ground_truth_centroids: Set[Tuple[float, float]] = {
        (49.35, 84.04), (16.45, 82.80), (82.28, 63.20), (49.21, 17.19), (83.12, 12.34),
        (83.13, 87.97), (48.26, 51.23), (16.50, 16.12), (81.79, 37.73), (16.08, 48.91)
    }
    found_centroids = re.findall(r"\(([\d.]+),\s*([\d.]+)\)", answer)
    parsed_set = {(float(x), float(y)) for x, y in found_centroids}
    matches = 0
    for gt_c in ground_truth_centroids:
        if any(abs(p_c[0] - gt_c[0]) < (REL_TOLERANCE * abs(gt_c[0]) + 1e-9) and \
               abs(p_c[1] - gt_c[1]) < (REL_TOLERANCE * abs(gt_c[1]) + 1e-9) for p_c in parsed_set):
            matches += 1
    return 1.0 if matches == 10 else 0.0

def acc_kmeans_1m6_smallest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the smallest cluster (1.6M points) is found (relative tolerance)."""
    # Ground truth: Cluster 5: 132615 points
    expected_val = 132615.0
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_kmeans_1m6_largest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the largest cluster (1.6M points) is found (relative tolerance)."""
    # Ground truth: Cluster 1: 178718 points
    expected_val = 178718.0
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

# --- K-means 1.8M functions ---
def acc_kmeans_1m8_all_centroids(answer: str) -> float:
    """Accuracy is 1.0 if all 10 centroids (1.8M points) are found (relative tolerance), 0.0 otherwise."""
    ground_truth_centroids: Set[Tuple[float, float]] = {
        (44.99, 11.30), (83.68, 83.16), (17.77, 51.85), (53.31, 61.71), (16.17, 83.74),
        (51.13, 35.19), (84.90, 48.49), (49.02, 87.37), (82.17, 15.39), (14.55, 18.37)
    }
    found_centroids = re.findall(r"\(([\d.]+),\s*([\d.]+)\)", answer)
    parsed_set = {(float(x), float(y)) for x, y in found_centroids}
    matches = 0
    for gt_c in ground_truth_centroids:
        if any(abs(p_c[0] - gt_c[0]) < (REL_TOLERANCE * abs(gt_c[0]) + 1e-9) and \
               abs(p_c[1] - gt_c[1]) < (REL_TOLERANCE * abs(gt_c[1]) + 1e-9) for p_c in parsed_set):
            matches += 1
    return 1.0 if matches == 10 else 0.0

def acc_kmeans_1m8_smallest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the smallest cluster (1.8M points) is found (relative tolerance)."""
    # Ground truth: Cluster 0: 146113 points
    expected_val = 146113.0
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_kmeans_1m8_largest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the largest cluster (1.8M points) is found (relative tolerance)."""
    # Ground truth: Cluster 2: 202547 points
    expected_val = 202547.0
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0


# --- LavaMD boxes1d=10 Specific Functions ---

def acc_lavamd_boxes10_avg_potential(answer: str) -> float:
    """Accuracy for the average potential energy (boxes1d=10 run, relative tolerance)."""
    # Ground truth: Average Potential Energy (v) | 1322.079678
    expected_val = 1322.079678
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_lavamd_boxes10_min_potential(answer: str) -> float:
    """Accuracy for the minimum potential energy (boxes1d=10 run, relative tolerance)."""
    # Ground truth: Min Potential Energy (v) | 294.916728
    expected_val = 294.916728
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_lavamd_boxes10_max_potential(answer: str) -> float:
    """Accuracy for the maximum potential energy (boxes1d=10 run, relative tolerance)."""
    # Ground truth: Max Potential Energy (v) | 1920.328419
    expected_val = 1920.328419
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_lavamd_boxes10_avg_force_x(answer: str) -> float:
    """Accuracy for the average force vector (x) (boxes1d=10 run, relative tolerance)."""
    # Ground truth: Average Force Vector (x) | 1.628118
    expected_val = 1.628118
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9 # Add small epsilon for near-zero expected values
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_lavamd_boxes10_min_force_y(answer: str) -> float:
    """Accuracy for the minimum force vector (y) (boxes1d=10 run, relative tolerance)."""
    # Ground truth: Min Force Vector (y) | -1732.352245
    expected_val = -1732.352245
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

# Note: acc_lavamd_boxes10_max_force_z already exists from previous requests, ensure it uses the correct value: 1766.183018

def acc_lavamd_boxes10_particle99_potential(answer: str) -> float:
    """Accuracy for the potential of particle 99 (boxes1d=10 run, relative tolerance)."""
    # Ground truth: Particle Index 99 | Potential (v) | 541.363563
    expected_val = 541.363563
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_lavamd_boxes10_particle199_force_x(answer: str) -> float:
    """Accuracy for the force vector (x) of particle 199 (boxes1d=10 run, relative tolerance)."""
    # Ground truth: Particle Index 199 | Force (x) | 538.945711
    expected_val = 538.945711
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    tolerance = REL_TOLERANCE * abs(expected_val) + 1e-9
    return 1.0 if abs(actual_val - expected_val) < tolerance else 0.0

def acc_lavamd_boxes10_particles_per_box(answer: str) -> float:
    """Accuracy for 'Particles per Box' (boxes1d = 10 run)."""
    # Value from boxes1d=10 simulation statistics
    expected_val = 128.0
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None: return 0.0
    # Exact match required for this integer value
    return 1.0 if abs(actual_val - expected_val) < 1e-6 else 0.0

################## end grading #########################


evalutation_questions = [
    # --- KMeans Questions ---
    Question(
        text="What is the x-coordinate of the left-most centroid for the 1000000-point, 10 cluster, K-means run?",
        accuracy_fn=acc_kmeans_leftmost_strict
    ),
    Question(
        text="For clustering the 1000000 data points, 10 cluster, what is the size (number of points) of the largest cluster?",
        accuracy_fn=acc_kmeans_largest_cluster
    ),
    Question(
        text="From the 1000000-point, 10 cluster, K-means run, what is the (x, y) coordinate pair for the centroid closest to the (0, 0) origin (i.e., the bottom-left-most)?",
        accuracy_fn=acc_kmeans_bottom_left_pair
    ),
    Question(
        text="For clustering 2000000 data points into 10 clusters, what is the size (number of points) of the smallest cluster?",
        accuracy_fn=acc_kmeans_2m_smallest_cluster
    ),
    Question(
        text="What is the size (number of points) of the largest cluster from the 2000000-point, 10-cluster K-means run?",
        accuracy_fn=acc_kmeans_2m_largest_cluster
    ),
    Question(
        text="I have 2000000 data points that I need to group into 10 distinct clusters. Can you compute all 10 final centroids?",
        accuracy_fn=acc_kmeans_2m_all_centroids_strict
    ),
    Question(
        text="What is the x-coordinate of the left-most centroid for the 2000000-point, 10-cluster K-means run?",
        accuracy_fn=acc_kmeans_2m_leftmost_x
    ),
    Question(
        text="From the 2000000-point, 10-cluster K-means run, what is the (x, y) coordinate pair for the centroid closest to the (100, 100) corner?",
        accuracy_fn=acc_kmeans_2m_top_right_pair
    ),
    # --- 1.2M Node Questions ---
    Question(
        text="I have 1200000 data points that I need to group into 10 distinct clusters. Can you compute all 10 centroids?",
        accuracy_fn=acc_kmeans_1m2_all_centroids
    ),
    Question(
        text="For the K-means run on 1200000 nodes, 10 cluster, what is the size of the smallest cluster?",
        accuracy_fn=acc_kmeans_1m2_smallest_cluster
    ),
    Question(
        text="For clustering the 1200000 data points (10 clusters), what is the size (number of points) of the largest cluster?",
        accuracy_fn=acc_kmeans_1m2_largest_cluster
    ),
    # --- 1.6M Node Questions ---
    Question(
        text="I need to group 1600000 data points into 10 distinct clusters. Can you compute all 10 final centroids?",
        accuracy_fn=acc_kmeans_1m6_all_centroids
    ),
    Question(
        text="From the K-means run on 1600000 nodes (10 clusters), what is the number of points in the smallest cluster?",
        accuracy_fn=acc_kmeans_1m6_smallest_cluster
    ),
    Question(
        text="What is the size (number of points) of the largest cluster from the 1600000-point, 10-cluster K-means run?",
        accuracy_fn=acc_kmeans_1m6_largest_cluster
    ),
    # --- 1.8M Node Questions ---
    Question(
        text="Can you compute all 10 final centroids for the K-means clustering of 1800000 data points into 10 clusters?",
        accuracy_fn=acc_kmeans_1m8_all_centroids
    ),
    Question(
        text="For clustering 1800000 data points into 10 clusters, what is the size (number of points) of the smallest cluster?",
        accuracy_fn=acc_kmeans_1m8_smallest_cluster
    ),
    Question(
        text="For the 1800000-node, 10-cluster K-means run, what is the size of the largest cluster?",
        accuracy_fn=acc_kmeans_1m8_largest_cluster
    ),
    # --- LavaMD Questions ---
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 8, what's their average potential energy?",
        accuracy_fn=acc_lavamd_boxes8_avg_energy
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10, what's the maximum value for the force vector (z)?",
        accuracy_fn=acc_lavamd_boxes10_max_force_z
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 8. How many particles per box are reported in the statistics?",
        accuracy_fn=acc_lavamd_boxes8_particles_per_box
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 11. How many particles per box are reported in the statistics?",
        accuracy_fn=acc_lavamd_boxes11_particles_per_box
    ),
    Question(
        text="What is the average potential energy (v) reported in the summary for the lavaMD run with boxes1d = 11?",
        accuracy_fn=acc_lavamd_boxes11_avg_potential
    ),
    Question(
        text="From the lavaMD simulation (boxes1d=11) summary, what is the maximum value for the force vector (z)?",
        accuracy_fn=acc_lavamd_boxes11_max_force_z
    ),
    Question(
        text="In the particle data dump for the lavaMD boxes1d=11 run, what is the potential (v) for particle index 42?",
        accuracy_fn=acc_lavamd_boxes11_particle42_potential
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10, what's their average potential energy (v)?",
        accuracy_fn=acc_lavamd_boxes10_avg_potential # Use the specific one for clarity
    ),
     Question(
        text="What is the minimum potential energy (v) reported in the summary for the lavaMD run with boxes1d = 10?",
        accuracy_fn=acc_lavamd_boxes10_min_potential
    ),
    Question(
        text="From the lavaMD simulation (boxes1d=10) summary, what is the maximum potential energy (v)?",
        accuracy_fn=acc_lavamd_boxes10_max_potential
    ),
    Question(
        text="What is the average force vector (x) reported in the summary for the lavaMD run with boxes1d = 10?",
        accuracy_fn=acc_lavamd_boxes10_avg_force_x
    ),
    Question(
        text="From the lavaMD simulation (boxes1d=10) summary, what is the minimum value for the force vector (y)?",
        accuracy_fn=acc_lavamd_boxes10_min_force_y
    ),
    Question(
        text="In the particle data dump for the lavaMD boxes1d=10 run, what is the potential (v) for particle index 99?",
        accuracy_fn=acc_lavamd_boxes10_particle99_potential
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10. From the data dump, what is the force vector (x) for particle index 199?",
        accuracy_fn=acc_lavamd_boxes10_particle199_force_x
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10. How many particles per box are reported in the statistics?",
        accuracy_fn=acc_lavamd_boxes10_particles_per_box # Use the specific one
    ),
    # -- Non-tool questions ---- 
    Question(
        text="How many days are in a leap year?",
        accuracy_fn=lambda a: 1.0 if re.search(r'\b366\b', a) else 0.0
    ),
    Question(
        text="If you have 3 apples and you buy 7 more, then give away 4, how many apples do you have?",
        accuracy_fn=lambda a: 1.0 if re.search(r'\b6\b', a) else 0.0
    ),
    # --- bm25 / kb Questions ---
    Question(
        text="When was the first moon landing?",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["July 20, 1969", "1969"]) else 0.0
    ),
    Question(
        text="When did Harry Potter and the Deathly Hallows come out?",
        accuracy_fn= lambda a: 1.0 if any(expected in a for expected in ["21 July 2007", "July 21, 2007", "2007"]) else 0.0
    ),
    Question(
        text="When did India participate in Olympics for first time?",
        accuracy_fn= lambda a: 1.0 if any(expected in a for expected in ["1900"]) else 0.0
    ),
    Question(
        text="Who began the mass printing of bibles five centuries ago?",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["Johannes Gutenberg", "Gutenberg"]) else 0.0
    ),
    Question(
        text="Who painted the Mona Lisa?",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["Leonardo da Vinci"]) else 0.0
    ),
    Question(
        text="What is the size of the angles of an equilateral triangle?",
        accuracy_fn= lambda a: 1.0 if any(expected in a for expected in ["60", "60°", "60 degrees"]) else 0.0
    ),
    Question(
        text="Where did the US get the Statue of Liberty?",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["France", "people of France"]) else 0.0
    ),
    Question(
        text="Who is next in line if the president dies?",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["vice president", "the vice president"]) else 0.0
    ),
    Question(
        text="who sings the theme song for miami vice",
        accuracy_fn=lambda a, expected_list=["Jan Hammer"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="what are the two main political parties in france",
        accuracy_fn=lambda a, expected_list=["The Republicans", "the Socialist Party"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="the depth of the wave base is approximately",
        accuracy_fn=lambda a, expected_list=["half the wavelength"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="who was the first band to play at woodstock",
        accuracy_fn=lambda a, expected_list=["Richie Havens"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="who has scored the most goals in a premier league debut season",
        accuracy_fn=lambda a, expected_list=["Kevin Phillips"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="what is the largest tree in the united states",
        accuracy_fn=lambda a, expected_list=["a giant sequoia", "General Sherman"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
]

# select 40% questions from the evaluation questions
questions_to_run = [
    Question(
        text="I have 1000000 data points that I need to group into 10 distinct clusters. Can you compute all 10 centroids?",
        accuracy_fn=acc_kmeans_all_centroids_strict
    ),
    Question(
        text="For the K-means run on 1000000 nodes, 10 cluster, what is the size of the smallest cluster?",
        accuracy_fn=acc_kmeans_smallest_cluster
    ),
    Question(
        text="From the K-means run on 2,000,000 data points (10 clusters), what is the (x, y) coordinate pair for the centroid closest to the (0, 0) origin?",
        accuracy_fn=acc_kmeans_2m_bottom_left_pair
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10, what's their average energy?",
        accuracy_fn=acc_lavamd_avg_energy
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10. From the data dump, what is the potential (v) for particle index 42?",
        accuracy_fn=acc_lavamd_boxes10_particle42_potential
    ),
    Question(
        text="Who wrote the novel '1984'?",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["George Orwell"]) else 0.0
    ),
    Question(
        text="Who does Patrick Stewart play in Star Trek?",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["Captain Jean-Luc Picard", "Picard", "Jean-Luc Picard"]) else 0.0
    ),
    Question(
        text="what is the southern end of the appalachian trail",
        accuracy_fn=lambda a, expected_list=["Springer Mountain"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="who was the highest-ranking black officer in the u.s. army at the beginning of the first world war",
        accuracy_fn=lambda a, expected_list=["Charles Young"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="when was disney the fox and the hound first released",
        accuracy_fn=lambda a, expected_list=["1981"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="approximately what percentage of earth's surface is covered with water",
        accuracy_fn=lambda a, expected_list=["71"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="writer of the monk who sold his ferrari",
        accuracy_fn=lambda a, expected_list=["Robin Sharma"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="membership in the european union requires countries to have which type of government",
        accuracy_fn=lambda a, expected_list=["democracy"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
]

print(len(evalutation_questions))
import os
import subprocess
import re
import math
from typing import Callable, List, Optional, Any, Tuple, Set

class Question:
    """
    Describes a question and its specific function for computing accuracy. [cite: 390]
    """
    def __init__(self, text: str, accuracy_fn: Callable[[str], float]):
        self.text = text
        self.accuracy_fn = accuracy_fn
        
################## start grading #########################

# Relative tolerance (default 10%) plus an integer minimum tolerance.
# Effective tolerance per value is:
#   max(INT_ABS_TOLERANCE, floor(abs(expected_value) * REL_TOLERANCE))
REL_TOLERANCE = float(os.environ.get("LLM_TOOL_REL_TOLERANCE", "0.1"))
INT_ABS_TOLERANCE = max(0, int(os.environ.get("LLM_TOOL_INT_ABS_TOLERANCE", "1")))


def _int_tolerance_for_expected(expected_val: float) -> int:
    rel_tol = int(math.floor(abs(expected_val) * REL_TOLERANCE))
    return max(INT_ABS_TOLERANCE, rel_tol)

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


def _find_best_match_floor_int(text: str, expected_val: float) -> int | None:
    found_numbers = [float(num) for num in re.findall(r'-?\d+(?:\.\d+)?', text)]
    if not found_numbers:
        return None
    expected_int = math.floor(expected_val)
    found_ints = [math.floor(num) for num in found_numbers]
    best_match = min(found_ints, key=lambda val: abs(val - expected_int))
    return int(best_match)


def _score_numeric_floor(answer: str, expected_val: float) -> float:
    actual_val = _find_best_match_floor_int(answer, expected_val)
    if actual_val is None:
        return 0.0
    expected_int = math.floor(expected_val)
    tol = _int_tolerance_for_expected(expected_val)
    return 1.0 if abs(actual_val - expected_int) <= tol else 0.0


def _parse_centroids_floor(answer: str) -> Set[Tuple[int, int]]:
    found_centroids = re.findall(r"\(([\d.]+),\s*([\d.]+)\)", answer)
    return {(math.floor(float(x)), math.floor(float(y))) for x, y in found_centroids}


def _score_pair_floor(answer: str, expected_pair: Tuple[float, float]) -> float:
    parsed_set = _parse_centroids_floor(answer)
    expected = (math.floor(expected_pair[0]), math.floor(expected_pair[1]))
    tol_x = _int_tolerance_for_expected(expected_pair[0])
    tol_y = _int_tolerance_for_expected(expected_pair[1])
    for pred_x, pred_y in parsed_set:
        if abs(pred_x - expected[0]) <= tol_x and abs(pred_y - expected[1]) <= tol_y:
            return 1.0
    return 0.0


def _score_all_centroids_floor(answer: str, expected_centroids: Set[Tuple[float, float]]) -> float:
    parsed = list(_parse_centroids_floor(answer))
    expected = [(math.floor(x), math.floor(y)) for x, y in expected_centroids]

    # One-to-one matching avoids counting a single predicted point for multiple expected points.
    for exp_x, exp_y in expected:
        tol_x = _int_tolerance_for_expected(exp_x)
        tol_y = _int_tolerance_for_expected(exp_y)
        best_idx = -1
        best_dist = None
        for idx, (pred_x, pred_y) in enumerate(parsed):
            dx = abs(pred_x - exp_x)
            dy = abs(pred_y - exp_y)
            if dx <= tol_x and dy <= tol_y:
                dist = dx + dy
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
        if best_idx < 0:
            return 0.0
        parsed.pop(best_idx)
    return 1.0

# --- Specific Accuracy Functions ---

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
    return _score_numeric_floor(answer, 163677.0)

def acc_kmeans_2m_bottom_left_pair(answer: str) -> float:
    """Accuracy is 1.0 if the bottom-left centroid (2M points) is found (relative tolerance)."""
    # Ground truth: Centroid 1: (17.37, 11.86)
    # (This is closest to (0,0) based on L1 norm: 17.37 + 11.86 = 29.23)
    return _score_pair_floor(answer, (17.37, 11.86))

def acc_kmeans_all_centroids_strict(answer: str) -> float:
    """Accuracy is 1.0 if all 10 centroids are found (relative tolerance), 0.0 otherwise."""
    ground_truth_centroids: Set[Tuple[float, float]] = {
        (82.54, 84.45), (47.23, 15.19), (45.17, 56.70), (15.88, 15.40),
        (15.83, 82.18), (63.97, 36.69), (84.65, 15.71), (15.91, 46.67),
        (49.36, 85.72), (85.45, 53.82)
    }
    return _score_all_centroids_floor(answer, ground_truth_centroids)

def acc_kmeans_largest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the largest cluster is found, 0.0 otherwise."""
    return _score_numeric_floor(answer, 112292.0)

def acc_kmeans_smallest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the smallest cluster is found, 0.0 otherwise."""
    return _score_numeric_floor(answer, 78448.0)

def acc_kmeans_bottom_left_pair(answer: str) -> float:
    """Accuracy is 1.0 if the specific centroid (15.88, 15.40) is found (relative tolerance)."""
    return _score_pair_floor(answer, (15.88, 15.40))

def acc_kmeans_leftmost_strict(answer: str) -> float:
    """Accuracy is 1.0 if the x-coordinate of the leftmost centroid is found (relative tolerance), 0.0 otherwise."""
    return _score_numeric_floor(answer, 15.83)

def acc_lavamd_avg_energy(answer: str) -> float:
    """Accuracy for the average potential energy (relative tolerance, 0/1 check)."""
    return _score_numeric_floor(answer, 1322.079678)


def acc_lavamd_boxes8_avg_energy(answer: str) -> float:
    """Accuracy for the average potential energy (boxes1d = 8 run, relative tolerance, 0/1 check)."""
    return _score_numeric_floor(answer, 1254.070689)

def acc_lavamd_boxes10_max_force_z(answer: str) -> float:
    """Accuracy for the max force vector (z) (boxes1d = 10 run, relative tolerance, 0/1 check)."""
    return _score_numeric_floor(answer, 1766.183018)

def acc_lavamd_boxes8_particles_per_box(answer: str) -> float:
    """Accuracy for 'Particles per Box' (boxes1d = 8 run)."""
    # Value from boxes1d=8 simulation statistics
    return _score_numeric_floor(answer, 128.0)

def acc_lavamd_boxes10_particle42_potential(answer: str) -> float:
    """Accuracy for the potential of particle 42 (boxes1d = 10 run, relative tolerance, 0/1 check)."""
    return _score_numeric_floor(answer, 382.914775)

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
    return _score_numeric_floor(answer, 232478.0)

def acc_kmeans_2m_all_centroids_strict(answer: str) -> float:
    """Accuracy is 1.0 if all 10 centroids (2M points) are found (relative tolerance), 0.0 otherwise."""
    # Ground truth from the 2M run
    ground_truth_centroids: Set[Tuple[float, float]] = {
        (48.24, 53.73), (17.37, 11.86), (19.07, 87.21), (82.03, 48.69), (53.59, 84.63),
        (84.94, 82.41), (51.85, 18.05), (18.74, 36.37), (84.22, 16.01), (14.82, 62.09)
    }
    return _score_all_centroids_floor(answer, ground_truth_centroids)

def acc_kmeans_2m_leftmost_x(answer: str) -> float:
    """Accuracy is 1.0 if the x-coordinate of the leftmost centroid (2M points) is found (relative tolerance), 0.0 otherwise."""
    return _score_numeric_floor(answer, 14.82)

def acc_kmeans_2m_top_right_pair(answer: str) -> float:
    """Accuracy is 1.0 if the top-right centroid (2M points) is found (relative tolerance)."""
    # Ground truth: Centroid 5: (84.94, 82.41)
    # (This is closest to (100, 100) based on L1 norm)
    return _score_pair_floor(answer, (84.94, 82.41))

def acc_lavamd_boxes11_particles_per_box(answer: str) -> float:
    """Accuracy for 'Particles per Box' (boxes1d = 11 run)."""
    return _score_numeric_floor(answer, 128.0)

def acc_lavamd_boxes11_avg_potential(answer: str) -> float:
    """Accuracy for the average potential energy (boxes1d = 11 run, relative tolerance, 0/1 check)."""
    return _score_numeric_floor(answer, 1348.530474)

def acc_lavamd_boxes11_max_force_z(answer: str) -> float:
    """Accuracy for the max force vector (z) (boxes1d = 11 run, relative tolerance, 0/1 check)."""
    return _score_numeric_floor(answer, 1763.076327)

def acc_lavamd_boxes11_particle42_potential(answer: str) -> float:
    """Accuracy for the potential of particle 42 (boxes1d = 11 run, relative tolerance, 0/1 check)."""
    return _score_numeric_floor(answer, 384.098276)

# --- K-means 1.2M functions ---
def acc_kmeans_1m2_all_centroids(answer: str) -> float:
    """Accuracy is 1.0 if all 10 centroids (1.2M points) are found (relative tolerance), 0.0 otherwise."""
    ground_truth_centroids: Set[Tuple[float, float]] = {
        (38.09, 18.33), (16.29, 83.12), (82.99, 83.52), (12.60, 16.48), (83.93, 50.63),
        (63.32, 18.43), (49.23, 84.52), (88.00, 16.62), (50.71, 52.71), (16.88, 50.09)
    }
    return _score_all_centroids_floor(answer, ground_truth_centroids)

def acc_kmeans_1m2_smallest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the smallest cluster (1.2M points) is found (relative tolerance)."""
    return _score_numeric_floor(answer, 98804.0)

def acc_kmeans_1m2_largest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the largest cluster (1.2M points) is found (relative tolerance)."""
    return _score_numeric_floor(answer, 133638.0)

# --- K-means 1.6M functions ---
def acc_kmeans_1m6_all_centroids(answer: str) -> float:
    """Accuracy is 1.0 if all 10 centroids (1.6M points) are found (relative tolerance), 0.0 otherwise."""
    ground_truth_centroids: Set[Tuple[float, float]] = {
        (49.35, 84.04), (16.45, 82.80), (82.28, 63.20), (49.21, 17.19), (83.12, 12.34),
        (83.13, 87.97), (48.26, 51.23), (16.50, 16.12), (81.79, 37.73), (16.08, 48.91)
    }
    return _score_all_centroids_floor(answer, ground_truth_centroids)

def acc_kmeans_1m6_smallest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the smallest cluster (1.6M points) is found (relative tolerance)."""
    return _score_numeric_floor(answer, 132615.0)

def acc_kmeans_1m6_largest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the largest cluster (1.6M points) is found (relative tolerance)."""
    return _score_numeric_floor(answer, 178718.0)

# --- K-means 1.8M functions ---
def acc_kmeans_1m8_all_centroids(answer: str) -> float:
    """Accuracy is 1.0 if all 10 centroids (1.8M points) are found (relative tolerance), 0.0 otherwise."""
    ground_truth_centroids: Set[Tuple[float, float]] = {
        (44.99, 11.30), (83.68, 83.16), (17.77, 51.85), (53.31, 61.71), (16.17, 83.74),
        (51.13, 35.19), (84.90, 48.49), (49.02, 87.37), (82.17, 15.39), (14.55, 18.37)
    }
    return _score_all_centroids_floor(answer, ground_truth_centroids)

def acc_kmeans_1m8_smallest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the smallest cluster (1.8M points) is found (relative tolerance)."""
    return _score_numeric_floor(answer, 146113.0)

def acc_kmeans_1m8_largest_cluster(answer: str) -> float:
    """Accuracy is 1.0 if the size of the largest cluster (1.8M points) is found (relative tolerance)."""
    return _score_numeric_floor(answer, 202547.0)


# --- LavaMD boxes1d=10 Specific Functions ---

def acc_lavamd_boxes10_avg_potential(answer: str) -> float:
    """Accuracy for the average potential energy (boxes1d=10 run, relative tolerance)."""
    return _score_numeric_floor(answer, 1322.079678)

def acc_lavamd_boxes10_min_potential(answer: str) -> float:
    """Accuracy for the minimum potential energy (boxes1d=10 run, relative tolerance)."""
    return _score_numeric_floor(answer, 294.916728)

def acc_lavamd_boxes10_max_potential(answer: str) -> float:
    """Accuracy for the maximum potential energy (boxes1d=10 run, relative tolerance)."""
    return _score_numeric_floor(answer, 1920.328419)

def acc_lavamd_boxes10_avg_force_x(answer: str) -> float:
    """Accuracy for the average force vector (x) (boxes1d=10 run, relative tolerance)."""
    return _score_numeric_floor(answer, 1.628118)

def acc_lavamd_boxes10_min_force_y(answer: str) -> float:
    """Accuracy for the minimum force vector (y) (boxes1d=10 run, relative tolerance)."""
    return _score_numeric_floor(answer, -1732.352245)

# Note: acc_lavamd_boxes10_max_force_z already exists from previous requests, ensure it uses the correct value: 1766.183018

def acc_lavamd_boxes10_particle99_potential(answer: str) -> float:
    """Accuracy for the potential of particle 99 (boxes1d=10 run, relative tolerance)."""
    return _score_numeric_floor(answer, 541.363563)

def acc_lavamd_boxes10_particle199_force_x(answer: str) -> float:
    """Accuracy for the force vector (x) of particle 199 (boxes1d=10 run, relative tolerance)."""
    return _score_numeric_floor(answer, 538.945711)

def acc_lavamd_boxes10_particles_per_box(answer: str) -> float:
    """Accuracy for 'Particles per Box' (boxes1d = 10 run)."""
    return _score_numeric_floor(answer, 128.0)

################## end grading #########################


evalutation_questions = [
    # --- KMeans Questions ---
    Question(
        text="What is the x-coordinate of the left-most centroid for the 1000000-point, 10 cluster, K-means run? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_leftmost_strict
    ),
    Question(
        text="For clustering the 1000000 data points, 10 cluster, what is the size (number of points) of the largest cluster? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_largest_cluster
    ),
    Question(
        text="From the 1000000-point, 10 cluster, K-means run, what is the (x, y) coordinate pair for the centroid closest to the (0, 0) origin (i.e., the bottom-left-most)? You must answer as exactly one tuple in the form (x,y). You must floor each coordinate to an integer, for example (12,34) instead of (12.91,34.78).",
        accuracy_fn=acc_kmeans_bottom_left_pair
    ),
    Question(
        text="For clustering 2000000 data points into 10 clusters, what is the size (number of points) of the smallest cluster? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_2m_smallest_cluster
    ),
    Question(
        text="What is the size (number of points) of the largest cluster from the 2000000-point, 10-cluster K-means run? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_2m_largest_cluster
    ),
    Question(
        text="I have 2000000 data points that I need to group into 10 distinct clusters. Can you compute all 10 final centroids? You must answer with exactly 10 tuples in this form: (x1,y1),(x2,y2),...,(x10,y10). You must floor every coordinate to an integer, for example (12,34),(56,78) instead of (12.91,34.78),(56.44,78.92).",
        accuracy_fn=acc_kmeans_2m_all_centroids_strict
    ),
    Question(
        text="What is the x-coordinate of the left-most centroid for the 2000000-point, 10-cluster K-means run? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_2m_leftmost_x
    ),
    Question(
        text="From the 2000000-point, 10-cluster K-means run, what is the (x, y) coordinate pair for the centroid closest to the (100, 100) corner? You must answer as exactly one tuple in the form (x,y). You must floor each coordinate to an integer, for example (12,34) instead of (12.91,34.78).",
        accuracy_fn=acc_kmeans_2m_top_right_pair
    ),
    # --- 1.2M Node Questions ---
    Question(
        text="I have 1200000 data points that I need to group into 10 distinct clusters. Can you compute all 10 centroids? You must answer with exactly 10 tuples in this form: (x1,y1),(x2,y2),...,(x10,y10). You must floor every coordinate to an integer, for example (12,34),(56,78) instead of (12.91,34.78),(56.44,78.92).",
        accuracy_fn=acc_kmeans_1m2_all_centroids
    ),
    Question(
        text="For the K-means run on 1200000 nodes, 10 cluster, what is the size of the smallest cluster? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_1m2_smallest_cluster
    ),
    Question(
        text="For clustering the 1200000 data points (10 clusters), what is the size (number of points) of the largest cluster? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_1m2_largest_cluster
    ),
    # --- 1.6M Node Questions ---
    Question(
        text="I need to group 1600000 data points into 10 distinct clusters. Can you compute all 10 final centroids? You must answer with exactly 10 tuples in this form: (x1,y1),(x2,y2),...,(x10,y10). You must floor every coordinate to an integer, for example (12,34),(56,78) instead of (12.91,34.78),(56.44,78.92).",
        accuracy_fn=acc_kmeans_1m6_all_centroids
    ),
    Question(
        text="From the K-means run on 1600000 nodes (10 clusters), what is the number of points in the smallest cluster? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_1m6_smallest_cluster
    ),
    Question(
        text="What is the size (number of points) of the largest cluster from the 1600000-point, 10-cluster K-means run? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_1m6_largest_cluster
    ),
    # --- 1.8M Node Questions ---
    Question(
        text="Can you compute all 10 final centroids for the K-means clustering of 1800000 data points into 10 clusters? You must answer with exactly 10 tuples in this form: (x1,y1),(x2,y2),...,(x10,y10). You must floor every coordinate to an integer, for example (12,34),(56,78) instead of (12.91,34.78),(56.44,78.92).",
        accuracy_fn=acc_kmeans_1m8_all_centroids
    ),
    Question(
        text="For clustering 1800000 data points into 10 clusters, what is the size (number of points) of the smallest cluster? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_1m8_smallest_cluster
    ),
    Question(
        text="For the 1800000-node, 10-cluster K-means run, what is the size of the largest cluster? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_1m8_largest_cluster
    ),
    # --- LavaMD Questions ---
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 8, what's their average potential energy? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes8_avg_energy
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10, what's the maximum value for the force vector (z)? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes10_max_force_z
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 8. How many particles per box are reported in the statistics? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes8_particles_per_box
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 11. How many particles per box are reported in the statistics? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes11_particles_per_box
    ),
    Question(
        text="What is the average potential energy (v) reported in the summary for the lavaMD run with boxes1d = 11? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes11_avg_potential
    ),
    Question(
        text="From the lavaMD simulation (boxes1d=11) summary, what is the maximum value for the force vector (z)? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes11_max_force_z
    ),
    Question(
        text="In the particle data dump for the lavaMD boxes1d=11 run, what is the potential (v) for particle index 42? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes11_particle42_potential
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10, what's their average potential energy (v)? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes10_avg_potential # Use the specific one for clarity
    ),
     Question(
        text="What is the minimum potential energy (v) reported in the summary for the lavaMD run with boxes1d = 10? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes10_min_potential
    ),
    Question(
        text="From the lavaMD simulation (boxes1d=10) summary, what is the maximum potential energy (v)? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes10_max_potential
    ),
    Question(
        text="What is the average force vector (x) reported in the summary for the lavaMD run with boxes1d = 10? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes10_avg_force_x
    ),
    Question(
        text="From the lavaMD simulation (boxes1d=10) summary, what is the minimum value for the force vector (y)? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes10_min_force_y
    ),
    Question(
        text="In the particle data dump for the lavaMD boxes1d=10 run, what is the potential (v) for particle index 99? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes10_particle99_potential
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10. From the data dump, what is the force vector (x) for particle index 199? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes10_particle199_force_x
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10. How many particles per box are reported in the statistics? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes10_particles_per_box # Use the specific one
    ),
    # -- Non-tool questions ---- 
    Question(
        text="How many days are in a leap year? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=lambda a: 1.0 if re.search(r'\b366\b', a) else 0.0
    ),
    Question(
        text="If you have 3 apples and you buy 7 more, then give away 4, how many apples do you have? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=lambda a: 1.0 if re.search(r'\b6\b', a) else 0.0
    ),
    # --- bm25 / kb Questions ---
    Question(
        text="When was the first moon landing? You must answer with a date or year only.",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["July 20, 1969", "1969"]) else 0.0
    ),
    Question(
        text="When did Harry Potter and the Deathly Hallows come out? You must answer with a date or year only.",
        accuracy_fn= lambda a: 1.0 if any(expected in a for expected in ["21 July 2007", "July 21, 2007", "2007"]) else 0.0
    ),
    Question(
        text="When did India participate in Olympics for first time? You must answer with a year only.",
        accuracy_fn= lambda a: 1.0 if any(expected in a for expected in ["1900"]) else 0.0
    ),
    Question(
        text="Who began the mass printing of bibles five centuries ago? You must answer with a short name or phrase only.",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["Johannes Gutenberg", "Gutenberg"]) else 0.0
    ),
    Question(
        text="Who painted the Mona Lisa? You must answer with a short name or phrase only.",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["Leonardo da Vinci"]) else 0.0
    ),
    Question(
        text="What is the size of the angles of an equilateral triangle? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn= lambda a: 1.0 if any(expected in a for expected in ["60", "60°", "60 degrees"]) else 0.0
    ),
    Question(
        text="Where did the US get the Statue of Liberty? You must answer with a short name or phrase only.",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["France", "people of France"]) else 0.0
    ),
    Question(
        text="Who is next in line if the president dies? You must answer with a short name or phrase only.",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["vice president", "the vice president"]) else 0.0
    ),
    Question(
        text="Who sings the theme song for Miami Vice? You must answer with a short name or phrase only.",
        accuracy_fn=lambda a, expected_list=["Jan Hammer"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="What are the two main political parties in France? You must answer with short comma-separated phrases only.",
        accuracy_fn=lambda a, expected_list=["The Republicans", "the Socialist Party"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="The depth of the wave base is approximately what? You must answer with a short phrase only.",
        accuracy_fn=lambda a, expected_list=["half the wavelength"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="Who was the first band to play at Woodstock? You must answer with a short name or phrase only.",
        accuracy_fn=lambda a, expected_list=["Richie Havens"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="Who has scored the most goals in a Premier League debut season? You must answer with a short name or phrase only.",
        accuracy_fn=lambda a, expected_list=["Kevin Phillips"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="What is the largest tree in the United States? You must answer with a short name or phrase only.",
        accuracy_fn=lambda a, expected_list=["a giant sequoia", "General Sherman"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
]

# select 40% questions from the evaluation questions
questions_to_run = [
    Question(
        text="I have 1000000 data points that I need to group into 10 distinct clusters. Can you compute all 10 centroids? You must answer with exactly 10 tuples in this form: (x1,y1),(x2,y2),...,(x10,y10). You must floor every coordinate to an integer, for example (12,34),(56,78) instead of (12.91,34.78),(56.44,78.92).",
        accuracy_fn=acc_kmeans_all_centroids_strict
    ),
    Question(
        text="For the K-means run on 1000000 nodes, 10 cluster, what is the size of the smallest cluster? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_kmeans_smallest_cluster
    ),
    Question(
        text="From the K-means run on 2,000,000 data points (10 clusters), what is the (x, y) coordinate pair for the centroid closest to the (0, 0) origin? You must answer as exactly one tuple in the form (x,y). You must floor each coordinate to an integer, for example (12,34) instead of (12.91,34.78).",
        accuracy_fn=acc_kmeans_2m_bottom_left_pair
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10, what's their average energy? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_avg_energy
    ),
    Question(
        text="Run a molecular dynamics simulation for a system with boxes1d = 10. From the data dump, what is the potential (v) for particle index 42? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=acc_lavamd_boxes10_particle42_potential
    ),
    Question(
        text="Who wrote the novel '1984'? You must answer with a short name or phrase only.",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["George Orwell"]) else 0.0
    ),
    Question(
        text="Who does Patrick Stewart play in Star Trek? You must answer with a short name or phrase only.",
        accuracy_fn= lambda a: 1.0 if any(expected.lower() in a.lower() for expected in ["Captain Jean-Luc Picard", "Picard", "Jean-Luc Picard"]) else 0.0
    ),
    Question(
        text="What is the southern end of the Appalachian Trail? You must answer with a short name or phrase only.",
        accuracy_fn=lambda a, expected_list=["Springer Mountain"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="Who was the highest-ranking Black officer in the U.S. Army at the beginning of the First World War? You must answer with a short name or phrase only.",
        accuracy_fn=lambda a, expected_list=["Charles Young"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="When was Disney's The Fox and the Hound first released? You must answer with a year only.",
        accuracy_fn=lambda a, expected_list=["1981"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="Approximately what percentage of Earth's surface is covered with water? You must answer with one integer only. You must floor any decimal value.",
        accuracy_fn=lambda a, expected_list=["71"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="Who is the writer of The Monk Who Sold His Ferrari? You must answer with a short name or phrase only.",
        accuracy_fn=lambda a, expected_list=["Robin Sharma"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
    Question(
        text="Membership in the European Union requires countries to have which type of government? You must answer with a short phrase only.",
        accuracy_fn=lambda a, expected_list=["democracy"]: 1.0 if any(expected.lower() in a.lower() for expected in expected_list) else 0.0
    ),
]


from utils import *

import numpy as np
from typing import Tuple


def lsdtw(
    sequence1: np.ndarray, 
    sequence2: np.ndarray, 
    num_alphabets: int =5, 
    window_size: int =3, 
    num_neighbors: int =7, 
    alpha: float =0.32
    ) -> Tuple[np.ndarray, float]:
    '''
    This function mean Locally Slope-based Dynamic Time Warping

    Params:
        sequence1: 1d time series
        sequence2: 1d time series
        num_alphabets: alphabet size for slope discretization
        window_size: filter window size
        num_neighbors: length of subsequence
        alpha: influence of local shape features

    Returns:
        path: warping path
        similarity_score: similarity calculated by lsdtw
    '''
    features = {chr(i): i for i in range(97, 123)}

    # element-wise function apply
    def _convert_feature_into_num(char: str):
        return features.get(char)
    convert_feature_into_num = np.vectorize(_convert_feature_into_num)

    # features
    filtered_sequence1 = filtering(sequence1, window_size)
    filtered_sequence2 = filtering(sequence2, window_size)

    subsequences1 = slide_sequence(filtered_sequence1, num_neighbors)
    subsequences2 = slide_sequence(filtered_sequence2, num_neighbors)

    local_slope_features1 = local_characteristics_of_sequence(subsequences1)
    local_slope_features2 = local_characteristics_of_sequence(subsequences2)

    feature_coding_matrix_1 = feature_coding(local_slope_features1, num_alphabets)
    feature_coding_matrix_2 = feature_coding(local_slope_features2, num_alphabets)

    feature_coding_matrix_converted_into_num_1 = convert_feature_into_num(
        feature_coding_matrix_1.reshape(-1)).reshape(sequence1.shape[0], -1
        )
    feature_coding_matrix_converted_into_num_2 = convert_feature_into_num(
        feature_coding_matrix_2.reshape(-1)).reshape(sequence2.shape[0], -1
        )
    
    # warping path
    distances, similarity_score = distance(
        filtered_sequence1,
        filtered_sequence2,
        feature_coding_matrix_converted_into_num_1,
        feature_coding_matrix_converted_into_num_2,
        alpha=alpha
    )

    path = find_path(len(sequence1)-1, len(sequence2)-1, distances)

    return path, similarity_score


if __name__ == '__main__':
    t1 = np.random.normal(0, 1, 100)
    t2 = np.random.normal(0, 1, 150)

    lsdtw(t1, t2)

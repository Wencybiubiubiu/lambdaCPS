# File used to convert sample data into suitable one to input to corresponding learning model

# One single sample will be in the following format:
# [graph, list_of_samples]
# graph = [adjacency_matrix, feature_matrix],
#          adjacency_matrix with size N x N, feature_matrix with size N x d, d is the number of features
# list_of_samples = [[initial_state, final_state], [initial_state, final_state], ...]
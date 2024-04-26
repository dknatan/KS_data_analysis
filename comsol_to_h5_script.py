import comsol_to_h5

# get pandas df from ';'-separated csv file
data_df = comsol_to_h5.read_from_comsol_native_csv("/scratch/n/Natan.Dominko/comsol_data/logistic_t_1e6_eps_4e2.txt")
# data_df = comsol_to_h5.read_from_comsol_native_csv("/home/n/Natan.Dominko/Downloads/test42.txt")

# convert csv file to dict that structures into ahierarchy of 
# {
#   "variable_name": {
#                      "parameter_name": {
#                                          "time_step_name": data_np_array
#                                          }
#                     }, 
#   "positions": {"X": x_positions, "Y": y_positions, ...}
# }
structured_data = comsol_to_h5.extract_data_frames_grouped_by_variable_name_parameters_and_time_steps(data_df)
# write the structured data to file
comsol_to_h5.write_data_to_h5(data_df, "/scratch/n/Natan.Dominko/h5_data/logistic_t_1e6_eps_4e2.h5")
# comsol_to_h5.write_data_to_h5(data_df, "/home/n/Natan.Dominko/Downloads/test42.h5")



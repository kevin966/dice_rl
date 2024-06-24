import numpy as np
import os
import pickle
import subprocess
import matplotlib.pyplot as plt


def calculate_kl_divergence(p_distribution, q_distribution):
	"""Calculates the Kullback-Leibler divergence."""
	kl_divergence = 0.0
	for (state_action), probability in p_distribution.items():
		distribution_correction_ratio = q_distribution.get(state_action, None)
		if distribution_correction_ratio is None:
			continue
		kl_divergence += probability * np.log(1.0 / distribution_correction_ratio)

	return kl_divergence


def main():

	alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	num_datasets = 5

	env_name = 'frozenlake'
	env_size = 4
	default = True  # True = Default environment; False = Random environment of size 'env_size'

	num_trajectory = 200
	max_trajectory_length = 100

	base_path = os.path.dirname(os.path.abspath(__file__))

	step_interval = 500


	############################################################################
	########################### 1. DATASET CREATION ############################
	############################################################################

	# Creates the datasets and stores the state visitation distributions for the behaviour policy
	dataset_command = [
			'python3', 'scripts/create_dataset.py',
			'--save_dir=./tests/testdata',
			'--load_dir=./tests/testdata',
			f'--env_name={env_name}',
			f'--env_size={env_size}',
			f'--default={default}',
			f'--num_trajectory={num_trajectory}',
			f'--max_trajectory_length={max_trajectory_length}',
			'--force'
	]

	# Creates multiple datasets per alpha value using the seed for numbering of the datasets
	# for alpha in alpha_values:
	# 	for seed in range(num_datasets):
	# 		dataset_command += [f'--alpha={alpha}', f'--seed={seed}']
	# 		subprocess.run(dataset_command)


	############################################################################
	########################### 2. POLICY EVALUATION ###########################
	############################################################################

	# Runs the Neural DICE estimator and stores the state-visitation distributions for the target policy (distribution correction ratios)
	estimator_command = [
			'python3', 'scripts/run_neural_dice.py',
			'--save_dir=./tests/testdata',
			'--load_dir=./tests/testdata',
			f'--env_name={env_name}',
			f'--env_size={env_size}',
			f'--default={default}',
			f'--num_trajectory={num_trajectory}',
			f'--max_trajectory_length={max_trajectory_length}'
	]

	# Runs the estimator command for the same alpha values
	# for alpha in alpha_values:  # Same alpha is needed to get the correct dataset
	# 	for seed in range(num_datasets):
	# 		estimator_command += [f'--alpha={alpha}', f'--seed={seed}']
	# 		subprocess.run(estimator_command)


	############################################################################
	####################### 3. KL DIVERGENCE CALCULATION #######################
	############################################################################

	kl_divergence_values = []  # (alpha, KL divergence)

	for alpha in alpha_values:
		kl_divergence_values_per_alpha = []
		for seed in range(num_datasets):
			# Creates the relative paths to the files
			state_distribution_file = os.path.join(base_path, f'3a_state_visitation_distributions_{env_name}_size{env_size}_default{default}_tabularTrue_alpha{alpha}_seed{seed}_numtraj{num_trajectory}_maxtraj{max_trajectory_length}')
			distribution_correction_ratios_file = os.path.join(base_path, f'3b_distribution_correction_ratios_{env_name}_size{env_size}_default{default}_tabularTrue_alpha{alpha}_seed{seed}_numtraj{num_trajectory}_maxtraj{max_trajectory_length}')
			# Loads the state visitation distribution of the behaviour policy
			with open(state_distribution_file, 'rb') as f:
				state_distribution_behaviour = pickle.load(f)
			# Loads the state visitation distribution of the target policy
			with open(distribution_correction_ratios_file, 'rb') as f:
				distribution_correction_ratios = pickle.load(f)

			# Calculates the KL divergence
			kl_divergence = calculate_kl_divergence(state_distribution_behaviour, distribution_correction_ratios)
			# Adds the KL divergence to a temporary list
			kl_divergence_values_per_alpha.append(kl_divergence)

		# Calculates the average KL divergence per alpha value
		kl_divergence_values.append((alpha, np.mean(kl_divergence_values_per_alpha)))

	# Prints the list of KL divergence values
	for alpha, kl_divergence in kl_divergence_values:
		print(f'Alpha: {alpha}, KL Divergence: {kl_divergence}')


	############################################################################
	############################ 4. MSE CALCULATION ############################
	############################################################################

	mse_values = []  # (alpha, MSE)
	ground_truths = []
	last_estimated_rewards = []

	# Collect ground truths and last estimated rewards
	for alpha in alpha_values:
		for seed in range(num_datasets):
			# Creates the relative path to the distribution file
			average_reward_file = os.path.join(base_path, f'4_cumulative_reward_{env_name}_size{env_size}_default{default}_tabularTrue_alpha{alpha}_seed{seed}_numtraj{num_trajectory}_maxtraj{max_trajectory_length}')
			# Loads the average reward per step tuple (ground truth, last estimated)
			with open(average_reward_file, 'rb') as f:
					average_reward = pickle.load(f)

			# Adds the ground truth and last estimated reward per step to temporary lists
			ground_truths.append(average_reward[0])
			last_estimated_rewards.append(average_reward[1])

	# Calculates the average ground truth over all datasets
	average_ground_truth = np.mean(ground_truths)

	# Calculates the MSE of the average reward per step (ground truth and last estimate)
	index = 0
	for alpha in alpha_values:
		for seed in range(num_datasets):  
			mse = np.mean(np.square(average_ground_truth - last_estimated_rewards[index]))
			mse_values.append((alpha, mse))
			index += 1

	# Prints the list of MSE values
	for alpha, mse in mse_values:
		print(f'Alpha: {alpha}, MSE: {mse}')


	############################################################################
	##################### 5. CUMULATIVE REWARD CONVERGENCE #####################
	############################################################################

	reward_convergences = []

	for alpha in alpha_values:
		reward_over_time_per_alpha = []
		for seed in range(num_datasets):
			# Creates the relative path to the file
			reward_over_time_file = os.path.join(base_path, f'5_cumulative_reward_convergence_{env_name}_size{env_size}_default{default}_tabularTrue_alpha{alpha}_seed{seed}_numtraj{num_trajectory}_maxtraj{max_trajectory_length}')
			# Loads the cumulative reward over time
			with open(reward_over_time_file, 'rb') as f:
				reward_over_time = pickle.load(f)

			# Adds the reward over time array to a temporary array
			reward_over_time_per_alpha.append(reward_over_time)

		# Converts the temporary array into a NumPy array for easier manipulation
		reward_over_time_np_array = np.array(reward_over_time_per_alpha)
		# Calculates the average cumulative reward over time per alpha
		reward_convergences.append(np.mean(reward_over_time_np_array, axis=0))

	# Generate the step indices
	num_steps = len(reward_convergences[0])
	step_indices = [index * step_interval for index in range(num_steps)]

	# Plots all the cumulative rewards over time in a single plot
	plt.figure(figsize=(10, 6))
	for i, reward_convergence in enumerate(reward_convergences):
		plt.plot(step_indices, reward_convergence, label=f'Alpha = {alpha_values[i]}')
	plt.title('Cumulative Reward Convergence for Different Alpha Values')
	plt.xlabel('Time Step')
	plt.ylabel('Cumulative Reward')

	# Adds the dashed line for the average ground truth
	plt.axhline(y=average_ground_truth, color='r', linestyle='--', label='Average Ground Truth')

	plt.legend()
	plt.grid(True)
	plt.savefig(os.path.join(base_path, f'5_plot_cumulative_reward_convergence_{env_name}_size{env_size}_default{default}_tabularTrue_alphas_numtraj{num_trajectory}_maxtraj{max_trajectory_length}.png'))

	# Plots all the cumulative rewards over time in separate plots per alpha
	for i, reward_convergence in enumerate(reward_convergences):
		plt.figure(figsize=(10, 6))
		plt.plot(step_indices, reward_convergence, label=f'Alpha = {alpha_values[i]}')
		plt.title(f'Cumulative Reward Convergence for Alpha = {alpha_values[i]}')
		plt.xlabel('Time Step')
		plt.ylabel('Cumulative Reward')

		# Adds the dashed line for the average ground truth
		plt.axhline(y=average_ground_truth, color='r', linestyle='--', label='Average Ground Truth')

		plt.legend()
		plt.grid(True)
		plt.savefig(os.path.join(base_path, f'5_plot_cumulative_reward_convergence_{env_name}_size{env_size}_default{default}_tabularTrue_alpha{alpha_values[i]}_numtraj{num_trajectory}_maxtraj{max_trajectory_length}.png'))


	############################################################################
	############################# 6. Visualisation #############################
	############################################################################

	# Extracts the KL divergences and puts them in a list
	kl_divergence_list = [kl_divergence for (_, kl_divergence) in kl_divergence_values]

	# Creates a list of MSE lists where each sublist contains 5 MSE values for each alpha value
	mse_lists = [
		[mse for (a, mse) in mse_values if a == alpha]
		for alpha in alpha_values
	]

	# Creates a double box plot with 6 boxes (one for each alpha value)
	plt.figure(figsize=(10, 6))
	# plt.boxplot(mse_lists, tick_labels=[f'KL={kd:.2f}' for kd in kl_divergence_list])
	plt.boxplot(mse_lists, tick_labels=[f'KL={kd:.2f}' for kd in kl_divergence_list], whis=10.0)
	plt.title('KL Divergence vs. MSE (Size: Default)')
	plt.xlabel('KL Divergence')
	plt.ylabel('MSE')
	plt.legend()

	# Adds the alpha values below each double box plot
	for i in range(len(kl_divergence_list)):
		plt.text(i + 1, plt.ylim()[1] * -0.05, f'α={alpha_values[i]}', ha='center', va='bottom')

	plt.xticks(rotation=45)  # Rotates the x-axis labels for better readability

	# Saves the double box plot
	double_box_plot_filename = os.path.join(base_path, f'6_double_box_plot_{env_name}_size{env_size}_default{default}_kl_divergence_mse.png')
	plt.savefig(double_box_plot_filename)

	print('Done!')

if __name__ == '__main__':
	main()

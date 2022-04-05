"""
Time Based Maintenance Assignment - Asset Management 2021/2022
Lecturer: Nadalina Merkelijn (n.g.merkelijn@rug.nl)
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

############# Student Specific Data #############

student_nr = "s3497887" # Add your student number, same as in data file names
data_path = "./data/" # Example: 'C:/Documents/AssetManagement/Assignment/' - Replace with folder path containing your files.

############# Data Preparation #############

# Manual Chapter 1:
def data_preperation(input_data_machine):
    # Add 'Durations' column to DataFrame:
    # Calculate duration value to 'Durations' for each row:
    output_data = input_data_machine
    output_data["Duration"] = input_data_machine["Time"].diff()
    output_data["Duration"][0] = input_data_machine["Time"][0]

    # Sort DataFrame by increasing values for column 'Durations' (and reset index):
    output_data.sort_values(by=["Duration"], ascending=True, inplace=True, ignore_index=True)

    # Check order of events in case of equal durations (event durations before censored durations):
    for i in range(len(output_data)-1):
        if output_data["Duration"][i] == output_data["Duration"][i+1] and output_data["Event"][i] == "PM" and output_data["Event"][i+1] == "failure":
            current_row = output_data.loc[i]
            output_data.loc[i] = output_data.loc[i+1]
            output_data.loc[i+1] = current_row

    return output_data

# Manual Chapter 2:
def create_kaplanmeier_curve_data(input_data_prepareddata):
    output_data = input_data_prepareddata

    # Add starting probability data to dataFrame:
    output_data["Probability"] = 1/len(output_data)

    # Calculate the weighted probability:
    for i in range(len(output_data)):
        if output_data["Event"].loc[i] == "PM":
            output_data["Probability"].loc[i + 1:] += output_data["Probability"].loc[i] / len(
                output_data["Probability"].loc[i + 1:])
            output_data["Probability"].loc[i] = 0

    indices = []
    for i in range(len(output_data)-1):
        if output_data["Duration"].loc[i] == output_data["Duration"].loc[i+1]:
            output_data["Probability"].loc[i] += output_data["Probability"].loc[i+1]
            indices.append(i)

    output_data = output_data.drop(output_data.index[indices]).reset_index(drop=True)
    # Calculate Reliability for each row:
    output_data["Reliability"] = 1 - output_data["Probability"].cumsum()

    return output_data

# Manual Chapter 3:
# def visualization(input_data_kaplanmeier):
# The above line is replaced by the following in Chapter 8:
def visualization(input_data_kaplanmeier, input_data_weibull):
    plt.step(input_data_kaplanmeier["Duration"], input_data_kaplanmeier["Reliability"], c="k", linewidth=0.6)
    plt.plot(input_data_weibull["t"], input_data_weibull["R_t"])
    plt.ylabel("$R(t)$")
    plt.xlabel("$time\ t\ (hours)$")
    plt.title("Kaplan-Meier and Weibull Curve")
    plt.show()

# Manual Chapter 4:
def meantimebetweenfailure_KM(input_data_kaplanmeier):

    MTBF = (input_data_kaplanmeier["Duration"] * input_data_kaplanmeier["Probability"]).sum()
    return MTBF

# Manual Chapter 5:
def find_lambdakappa(input_data_kaplanmeier):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Create two ranges of values for potential lambda/kappa combinations (1-35 integer for lambda, 0-3.5 one decimal for kappa)
    l_range = np.linspace(1, 35, 35)
    k_range = np.linspace(0, 3.5, 36)
    # Create an dataframe for data collection
    output_data = pd.DataFrame(index=pd.MultiIndex.from_product([l_range, k_range], names=["lambda", "kappa"])).reset_index()

    # Create the Optimization Matrix (Chapter x, step x)
    i = 0
    for t in input_data_kaplanmeier['Duration']: # We are adding a column for each 'Duration' in input_data
        column_name = 'Duration '+str(round(t,3))
        output_data[column_name] = 0 # fill column with zeros
        j = 0
        for l in l_range:
            for k in k_range: # Triple for-loop: we are now looking at a specific row in a specific column!
                # Check if this column (= duration) is a failure event:
                if input_data_kaplanmeier['Event'].loc[i] == "failure":
                    f_t = k/l * (t/l)**(k-1) * math.exp(-((t/l)**k))
                    output_data[column_name][j] = np.log(f_t)
                    j += 1
                else:
                    R_t = math.exp(-(t/l)**k)
                    output_data[column_name][j] = np.log(R_t)
                    j += 1
        i += 1

    # Add column to output_data that sums (only) the calculated log-values for each row.
    output_data["sums"] = output_data.iloc[:, 2:].sum(axis=1)

    # Find which row has the highest sum value; determine the corresponding lambda and kappa of that row.
    index = np.argmax(output_data["sums"])
    found_lambda = output_data["lambda"].loc[index]
    found_kappa = output_data["kappa"].loc[index]

    return found_lambda, found_kappa

# Manual Chapter 6:
def create_weibull_curve_data(input_data_kaplanmeier,l,k):

    # Create a range of time steps; length 0 to maximum duration input_data in increments of 0.01
    max_T = int(input_data_kaplanmeier["Duration"].max())
    deltaT = np.linspace(0, max_T, max_T*100)

    # Create an output_data DataFrame with two columns: t and R_t
    output_data = pd.DataFrame(columns=["t", "R_t"])
    output_data["t"] = deltaT
    # Calculate R_t for each deltaT value and add to output_data
    output_data["R_t"] = output_data.apply(lambda row: math.exp(-(row["t"]/l)**k), axis=1)

    return output_data

# Manual Chapter 7:
def meantimebetweenfailure_WB(l,k):

    MTBF = l * math.gamma(1+1/k)

    return MTBF

# Manual Chapter 9:
def create_cost_data(input_data_kaplanmeier,l,k,PM_cost,CM_cost):

    # Create an output_data DataFrame with 6 columns: t, R_t, F_t, Cost per Cycle, Mean Cycle Length, create_cost_data
    output_data = pd.DataFrame(columns=["t", "R_t", "F_t", "Costs per Cycle", "Mean Cycle Length", "Cost_t"])

    # Fill columns: t, R_t, F_t, Cost per Cycle
    max_T = int(input_data_kaplanmeier["Duration"].max())
    deltaT = np.linspace(0, max_T, max_T * 100)
    output_data["t"] = deltaT

    output_data["R_t"] = output_data.apply(lambda row: math.exp(-(row["t"] / l) ** k), axis=1)
    output_data["F_t"] = output_data.apply(lambda row: 1 - math.exp(-(row["t"] / l) ** k), axis=1)
    output_data["Costs per Cycle"] = output_data["R_t"].multiply(int(PM_cost)) + output_data["F_t"].multiply(int(CM_cost))

    # Fill column: Mean Cycle Length
    output_data["Mean Cycle Length"] = output_data["R_t"].multiply(0.01).cumsum()

    # Fill column: Cost_t
    output_data["Cost_t"] = output_data["Costs per Cycle"]/output_data["Mean Cycle Length"]

    # Find the minimum maintenance costs
    min_costs_data = output_data[output_data["Cost_t"] == output_data["Cost_t"].min()]

    # cost savings
    cost_savings = float(CM_cost) / meantimebetweenfailure_WB(l, k) - float(min_costs_data["Cost_t"])
    print("Optimal PM time is every: ", float(min_costs_data["t"]), " hours")
    print("Costs savings with optimal PM: ", cost_savings)
    print("Optimal PM policy saves: ", cost_savings/(float(CM_cost) / meantimebetweenfailure_WB(l, k))*100, "%")

    # Visualize the cost data
    plt.plot(output_data["t"], output_data["Cost_t"])
    plt.vlines(min_costs_data["t"], ymin=0, ymax=min_costs_data["Cost_t"], linestyles="dashed")
    ylim = float(min_costs_data["Cost_t"])*2.5
    plt.xlabel("$T (hours)$")
    plt.ylabel("$\eta (H)$")
    plt.title("Cost Curve and Optimal PM")
    plt.annotate("$T_{opt}$", (min_costs_data["t"], min_costs_data["Cost_t"]/2))
    plt.ylim(0, ylim)
    plt.show()

    return output_data

# Manual Chapter 10:
def data_preperation_variable_usage(input_data_machine, input_data_usage):

    # Prepare a dataframe
    output_data = input_data_usage

    # Calculate Virtual Age at Mode Change
    output_data["Speed Change Timestamp"] = output_data["Time"]
    output_data["Virtual Age at Timestamp"] = 0

    for i in range(len(output_data["Speed Change Timestamp"])-1):
        if output_data["Mode"][i] == "full speed":
            output_data["Virtual Age at Timestamp"][i+1] = output_data["Speed Change Timestamp"][i+1] - output_data["Speed Change Timestamp"][i]
        elif output_data["Mode"][i] == "half speed":
            output_data["Virtual Age at Timestamp"][i + 1] = (output_data["Speed Change Timestamp"][i + 1] - output_data["Speed Change Timestamp"][i])/2
        elif output_data["Mode"][i] == "off":
            output_data["Virtual Age at Timestamp"][i + 1] = 0
    output_data["Virtual Age at Timestamp"] = output_data["Virtual Age at Timestamp"].cumsum()

    # Merge output_data and input_data_machine
    output_data = pd.merge_asof(input_data_machine, output_data, on="Time", direction="backward")

    # Calculate Virtual Age at event
    output_data["Virtual Age at Event"] = 0
    for i in range(len(output_data["Speed Change Timestamp"])):
        if output_data["Mode"][i] == "full speed":
            output_data["Virtual Age at Event"][i] = output_data["Time"][i] - output_data["Speed Change Timestamp"][i] + output_data["Virtual Age at Timestamp"][i]
        elif output_data["Mode"][i] == "half speed":
            output_data["Virtual Age at Event"][i] = (output_data["Time"][i] - output_data["Speed Change Timestamp"][i])/2 + output_data["Virtual Age at Timestamp"][i]

    # Create return dataframe containing only useful information
    output_data = output_data[["Event", "Virtual Age at Event"]]
    output_data.rename(columns={"Virtual Age at Event": "Time"}, inplace=True)

    return output_data


############# Run Tool #############

def run_analysis():
    # supress warning in pandas
    pd.options.mode.chained_assignment = None

    machine_name = input("Enter the machine name here: ")     # which machine number (takes e.g. "1")
    variable_usage_check = input("Do you want to consider usage data in the analysis? yes/no: ").lower()

    if variable_usage_check == "yes":
        machine_data = pd.read_csv(f'{data_path}{student_nr}-Machine-{machine_name}.csv')
        machine_data_usage = pd.read_csv(f'{data_path}{student_nr}-Machine-{machine_name}-usage.csv')
        prepared_data_variable_usage = data_preperation_variable_usage(machine_data, machine_data_usage)
        prepared_data = data_preperation(prepared_data_variable_usage)
    elif variable_usage_check == "no":
        machine_data = pd.read_csv(f'{data_path}{student_nr}-Machine-{machine_name}.csv')
        prepared_data = data_preperation(machine_data)
    else:
        raise exit('Please indicate "yes" or "no" when asked if you want to use the usage data')

    PM_cost = int(input("Enter the PM cost here: "))
    CM_cost = int(input("Enter the CM cost here: "))

    KM_data = create_kaplanmeier_curve_data(prepared_data)
    # visualization(KM_data)
    MTBF_KM = meantimebetweenfailure_KM(KM_data)
    print('The MTBF-KaplanMeier is: ', MTBF_KM)

    l, k = find_lambdakappa(KM_data)
    print("optimal lambda: ", l, " and optimal kappa: ", k)

    input_data_weibull = create_weibull_curve_data(KM_data, l, k)
    MTBF_WB = meantimebetweenfailure_WB(l, k)
    print('The MTBF-Weibull is: ', MTBF_WB)

    visualization(KM_data, input_data_weibull)
    create_cost_data(KM_data, l, k, PM_cost, CM_cost)

if __name__=="__main__":
    run_analysis()

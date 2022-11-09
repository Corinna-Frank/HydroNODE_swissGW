# --------------------------------------------------
# Neural ODE models in Hydrology
# - load data
# - build models
# - train models
#
# marvin.hoege@eawag.ch, Nov. 2022 (update)
# --------------------------------------------------

cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Revise

using DataFrames, Dates, Statistics
using DelimitedFiles, CSV

using OrdinaryDiffEq, DiffEqFlux, Flux
using SciMLSensitivity

using Optimization, BlackBoxOptim
using Zygote

using Interpolations

import SpecialFunctions # for gamma_inc
import DSP # for conv

using Random
Random.seed!(123)


# ===========================================================
# USER INPUT:

# set data directory
data_path = joinpath(pwd(),"data")
data_filename = "Crêtelongue.csv"

# choose model M50 or M100
chosen_model_id = "M100"

# choose basin id
basin_id = "01013500"

# define training and testing period
train_start_date = Date(2000,01,01)
train_stop_date = Date(2015,12,31)
test_start_date = Date(2016,01,01)
test_stop_date = Date(2020,12,31)

# if `false`, read the bucket model (M0) parameters from "bucket_opt_init.csv"
train_bucket_model = true

# ===========================================================

includet("HydroNODE_data.jl")
includet("HydroNODE_models.jl")
includet("HydroNODE_training.jl")


# -------------------------------------------------------
# Objective function: Nash-Sutcliffe Efficiency

NSE(pred, obs) = 1 - sum((pred .- obs).^2) / sum((obs .- mean(obs)).^2)

function NSE_loss(pred_model, params, batch, time_batch)

    pred, = pred_model(params, time_batch)
    loss = -NSE(pred,batch)

    return loss, pred
end

# -------------------------------------------------------
# Load and preprocess data

#Date,heads,precipitation,temperature,evaporation,rivers
input_var_names = ["precipitation","temperature","evaporation"]#["Daylight(h)", "Prec(mm/day)", "Tmean(C)"]
output_var_name = "heads"#"Flow(mm/s)"

df = load_data(data_path, data_filename)

# drop unused cols
select!(df, Not(Symbol("rivers")));

# adjust start and stop date if necessary
if df[1, "Date"] != train_start_date
    train_start_date = maximum([df[1, "Date"],train_start_date])
end

if df[end, "Date"] != test_stop_date
    test_stop_date = minimum([df[end, "Date"], test_stop_date])
end

# format data
data_x, data_y, data_timepoints,
train_x, train_y, train_timepoints, = prepare_data(df,
(train_start_date, train_stop_date, test_start_date, test_stop_date),input_var_names,output_var_name)

# normalize data
norm_moments_in = [mean(data_x, dims=1); std(data_x, dims = 1)]

norm_P = prep_norm(norm_moments_in[:,2])
norm_T = prep_norm(norm_moments_in[:,3])

# -------------------------------------------------------
# interpolation

itp_method = SteffenMonotonicInterpolation()

# ["precipitation","temperature","evaporation"]
itp_P = interpolate(data_timepoints, data_x[:,1], itp_method)
itp_T = interpolate(data_timepoints, data_x[:,2], itp_method)
itp_PET = interpolate(data_timepoints, data_x[:,3], itp_method)


# ===============================================================
# Bucket model training and full model preparation

NSE_loss_bucket_w_states(p) =  NSE_loss(basic_bucket_incl_states, p, train_y, train_timepoints)[1]

@info "Bucket model training..."

# Parameter ranges for bucket model:
# f: Rate of decline in flow from catchment bucket   | Range: (0, 0.1)
# smax: Maximum storage of the catchment bucket      | Range: (100, 1500)
# qmax: Maximum subsurface flow at full bucket       | Range: (10, 50)
# ddf: Thermal degree‐day factor                     | Range: (0, 5.0)
# tmax: Temperature above which snow starts melting  | Range: (0, 3.0)
# tmin: Temperature below which precipitation is snow| Range: (-3.0, 0)


if train_bucket_model == true
    # p_all_init = [S0_init, S1_init, S2_init, p0, p1, p2,     k,   T_t , k_v, S1max, l_p,   log_S2max, log_k_s, log_gamma]
    lower_bounds = [0.0,0.0,0.0,               1e-9,1e-2,1e-2,  1.0, 0.0, 0.5, 2.0, 0.25,     1.0,     -9.0,     -5.0 ] # [0.01, 100.0, 0.0, 100.0, 10.0, 0.01, 0.0, -3.0]
    upper_bounds = [1000.0,1000.0,1000.0,      1e4,  1e2, 2e3, 20.0, 0.0, 1.5, 2.0, 0.25,     3.0,      4.0,      1.3] # [1500.0, 1500.0, 0.1, 1500.0, 50.0, 5.0, 3.0, 0.0]

    SearchRange = [(lower_bounds[i], upper_bounds[i]) for i in 1:length(lower_bounds)]

    p_all_opt_bucket = best_candidate(BlackBoxOptim.bboptimize(NSE_loss_bucket_w_states; SearchRange, MaxSteps = 5000))


    S_bucket_precalib = p_all_opt_bucket[1:3]
    p_bucket_precalib = p_all_opt_bucket[4:end]
else

    bucket_opt_init = readdlm("bucket_opt_init.csv", ',')
    basins_available = lpad.(string.(Int.(bucket_opt_init[:,1])), 8, "0")

    basin_wanted = findall(x -> x==basin_id, basins_available)[1]

    p_all_opt_bucket = bucket_opt_init[basin_wanted, 2:end]

    S_bucket_precalib = p_all_opt_bucket[1:2]
    p_bucket_precalib = p_all_opt_bucket[3:end]
end
@info "... complete!"

# Forward Call
p_all_expected = [200.0,200.0,200.0,         10.0,10.0,100.0,   10.0, 0.0, 1.0, 2.0,  0.25, 1.0,       1.0,     0.0 ]
Q_bucket, S_bucket  = swissGW_buckets(p_all_expected, train_timepoints) 
# Q_bucket, S_bucket  = basic_bucket_incl_states([S_bucket_precalib..., p_bucket_precalib...], train_timepoints)

NSE_opt_bucket = -NSE_loss_bucket_w_states(p_all_opt_bucket)


# ===============================================================
# Neural ODE models

# -------------
# preparation

norm_S0 = prep_norm([mean(S_bucket[1,:]), std(S_bucket[1,:])])
norm_S1 = prep_norm([mean(S_bucket[2,:]), std(S_bucket[2,:])])

NN_NODE, p_NN_init = initialize_NN_model(chosen_model_id)

S0_bucket_ = S_bucket[1,:]
S1_bucket_ = S_bucket[2,:]
Lday_bucket_ = train_x[:,1]
P_bucket_ = train_x[:,2]
T_bucket_ = train_x[:,3]

NN_input = [norm_S0.(S0_bucket_) norm_S1.(S1_bucket_) norm_P.(P_bucket_) norm_T.(T_bucket_)]

@info "NN pre-training..."
p_NN_init = pretrain_NNs_for_bucket_processes(chosen_model_id, NN_NODE, p_NN_init,
    NN_input, p_bucket_precalib, S0_bucket_, S1_bucket_, Lday_bucket_, P_bucket_, T_bucket_)
@info "... complete!"

pred_NODE_model= prep_pred_NODE(NN_NODE, p_bucket_precalib[6:-1:4], S_bucket_precalib, length.(initial_params.(NN_NODE)))


# -------------
# training

@info "Neural ODE model training..."
p_opt_NODE = train_model(pred_NODE_model, p_NN_init, train_y, train_timepoints; optmzr = ADAM(0.001), max_N_iter = 75)
@info "... complete."

NSE_opt_NODE = -NSE_loss(pred_NODE_model,p_opt_NODE, train_y, train_timepoints)[1]

# -------------
# comparison bucket vs. Neural ODE
@info "Nash-Sutcliffe-Efficiency comparison (optimal value: 1):"

@info "NSE bucket model: $NSE_opt_bucket"

@info "NSE NeuralODE model: $NSE_opt_NODE"

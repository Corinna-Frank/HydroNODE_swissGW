# -------------------------------------------------------
# HydroNODE Model Functions
#
# 1) EXP-Hydro
#
#   time-continuous implementation of EXP-Hydro
#   (originally presented in "Patil, S. and Stieglitz, M. 2014. Modelling daily streamflow
#   at ungauged catchments: what information is necessary? Hydrological Processes, 28(3)")
#
#   inspired by the discrete time step Python implementation by
#   "Jiang, S., Zheng, Y., and Solomatine, D. 2020. Improving AI system awareness
#   of geoscience knowledge: symbiotic integration of physical approaches and deep
#   learning. Geophysical Research Letters, 47(13)"
#
# 2) Neural Networks and Neural ODE models
#
# marvin.hoege@eawag.ch, Mar. 2022
# --------------------------------------------------


# ===========================================================
# 1) EXP-Hydro model equations

# smooting step function
step_fct(x) = (tanh(5.0*x) + 1.0)*0.5

# snow precipitation
Ps(P, T, Tmin) = step_fct(Tmin-T)*P

# rain precipitation
Pr(P, T, Tmin) = step_fct(T-Tmin)*P

# snow melt
M(S0, T, Df, Tmax) = step_fct(T-Tmax)*step_fct(S0)*minimum([S0, Df*(T-Tmax)])

# evapotranspiration
PET(T, Lday) = 29.8 * Lday * 0.611 * exp((17.3*T)/(T+237.3)) / (T + 273.2)
ET(S1, T, Lday, Smax) = step_fct(S1)*step_fct(S1-Smax)*PET(T,Lday) + step_fct(S1)*step_fct(Smax-S1)*PET(T,Lday)*(S1/Smax)

# base flow
Qb(S1,f,Smax,Qmax) = step_fct(S1)*step_fct(S1-Smax)*Qmax + step_fct(S1)*step_fct(Smax-S1)*Qmax*exp(-f*(Smax-S1))

# peak flow
Qs(S1, Smax) = step_fct(S1)*step_fct(S1-Smax)*(S1-Smax)


# ---------------------------------------------------------------------------
# EXP-Hydro model: Two buckets (water and snow), 5 processes and 6 parameters


function basic_bucket_incl_states(p_, t_out)


    function exp_hydro_optim_states!(dS,S,ps,t)
        f, Smax, Qmax, Df, Tmax, Tmin = ps

        Lday = itp_Lday(t)
        P    = itp_P(t)
        T    = itp_T(t)

        Q_out = Qb(S[2],f,Smax,Qmax) + Qs(S[2], Smax)

        dS[1] = Ps(P, T, Tmin) - M(S[1], T, Df, Tmax)
        dS[2] = Pr(P, T, Tmin) + M(S[1], T, Df, Tmax) - ET(S[2], T, Lday, Smax) - Q_out
    end

    prob = ODEProblem(exp_hydro_optim_states!, p_[1:2], Float64.((t_out[1], maximum(t_out))))

    sol = solve(prob, BS3(), u0 = p_[1:2], p=p_[3:end], saveat=t_out, reltol=1e-3, abstol=1e-3, sensealg= ForwardDiffSensitivity())

    Qb_ = Qb.(sol[2,:], p_[3], p_[4], p_[5])
    Qs_ = Qs.(sol[2,:], p_[4])

    Qout_ = Qb_.+Qs_

    return Qout_, sol

end


# ============================================================================================================
# 2) Neural Networks and Neural ODE models

function initialize_NN_model(chosen_model_id)

    if chosen_model_id == "M50"

        # NN for ET
        NN_in_fct1 = FastChain((x,p) -> x.^1,FastDense(3, 16, tanh), FastDense(16,16, leakyrelu), FastDense(16,1))
        # NN for Q
        NN_in_fct2 = FastChain((x,p) -> x.^1,FastDense(2, 16, tanh), FastDense(16,16, leakyrelu), FastDense(16,1))

        NN_in_fct = [NN_in_fct1 NN_in_fct2]
        p_NN_init = [initial_params(NN_in_fct1), initial_params(NN_in_fct2)]

    elseif chosen_model_id == "M100"

        NN_in_fct = FastChain((x,p) -> x.^1,FastDense(4, 32, tanh), FastDense(32,32, leakyrelu), FastDense(32,32, leakyrelu), FastDense(32,32, leakyrelu), FastDense(32,32, leakyrelu), FastDense(32,5))
        p_NN_init = initial_params(NN_in_fct)

    end

    return NN_in_fct, p_NN_init
end

# --------------------------------------------
# M50

function NeuralODE_M50(p, t_out, ann, p_bucket_precal, addfeature; S_init = [0.0, 0.0])

    ann_ET = ann[1]
    ann_Q = ann[2]

    idcs_params_ann_ET = 1:addfeature[1]
    idcs_params_ann_Q = addfeature[1]+1:sum(addfeature)

    function NeuralODE_M50_core!(dS,S,p,t)

        Tmin, Tmax, Df = (p_bucket_precal...,)

        Lday = itp_Lday(t)
        P    = itp_P(t)
        T    = itp_T(t)

        g_ET = ann_ET([norm_S0(S[1]), norm_S1(S[2]), norm_T(T)],p[idcs_params_ann_ET])
        g_Q = ann_Q([norm_S1(S[2]), norm_P(P)],p[idcs_params_ann_Q])

        melting = M(S[1], T, Df, Tmax)
        dS[1] = Ps(P, T, Tmin) - melting
        dS[2] = Pr(P, T, Tmin) + melting - step_fct(S[2])*Lday*exp(g_ET[1])- step_fct(S[2])*exp(g_Q[1])

    end

    prob = ODEProblem(NeuralODE_M50_core!, S_init, Float64.((t_out[1], maximum(t_out))), p)

    sol = solve(prob, BS3(), dt=1.0, saveat=t_out, reltol=1e-3, abstol=1e-3, sensealg=BacksolveAdjoint())

    P_interp = norm_P.(itp_P.(t_out))

    S1_ = norm_S1.(sol[2,:])

    Qout_ =  exp.(ann_Q(permutedims([S1_ P_interp]),p[idcs_params_ann_Q])[1,:])

    return Qout_, sol

end

# --------------------------------------------
# M100

function NeuralODE_M100(p, t_out, ann, args...; S_init = [0.0, 0.0])

    function NeuralODE_M100_core!(dS,S,p,t)

        Lday = itp_Lday(t)
        P    = itp_P(t)
        T    = itp_T(t)

        g = ann([norm_S0(S[1]), norm_S1(S[2]), norm_P(P), norm_T(T)],p)

        melting = relu(step_fct(S[1])*sinh(g[3]))

        dS[1] = relu(sinh(g[4])*step_fct(-T)) - melting
        dS[2] = relu(sinh(g[5])) + melting - step_fct(S[2])*Lday*exp(g[1])- step_fct(S[2])*exp(g[2])

    end

    prob = ODEProblem(NeuralODE_M100_core!, S_init, Float64.((t_out[1], maximum(t_out))), p)

    sol = solve(prob, BS3(), dt=1.0, saveat=t_out, reltol=1e-3, abstol=1e-3, sensealg=BacksolveAdjoint())

    P_interp = norm_P.(itp_P.(t_out))
    T_interp = norm_T.(itp_T.(t_out))

    S0_ = norm_S0.(sol[1,:])
    S1_ = norm_S1.(sol[2,:])


    Qout_ =  exp.(ann(permutedims([S0_ S1_ P_interp T_interp]),p)[2,:])


    return Qout_, sol

end


# --------------------------------------------
# model wrapper

if chosen_model_id == "M50"
    function prep_pred_NODE(set_ann, set_p_bucket_fix, S_init, set_feature)
        (p, time_batch) -> NeuralODE_M50(p, time_batch, set_ann, set_p_bucket_fix, set_feature; S_init = S_init)
    end
elseif chosen_model_id == "M100"
    function prep_pred_NODE(set_ann, set_p_bucket_fix, S_init, set_feature)
        (p, time_batch) -> NeuralODE_M100(p, time_batch, set_ann, set_p_bucket_fix, set_feature; S_init = S_init)
    end
else
    println.("chosen_model_id not available!")
end


# Swiss Groundwater ==================================================================

# SNOW bucket

# parameters
k   = 10.0  # melting coeff
T_t = 0.0   # threshold melting temperature
# snow precipitation
Ps(P, T; T_t = 0.0) = step_fct(T_t-T)*P

# rain precipitation
Pr(P, T; T_t = 0.0) = step_fct(T-T_t)*P

melt(S0, T; k=10.0, T_t=0.0) = step_fct(T-T_t)*step_fct(S0)*(k*(T-T_t))

# INTERCEPT bucket

S1max = 2     # maximum storage capacity of the interception reservoir
k_v = 1.0      # vegetation coeff
Emax(PET; k_v = 1.0) = k_v*PET
Ei(S1,PET; k_v = 1.0) = step_fct(S1)*minimum([S1, Emax(PET; k_v = k_v)])

Pe(S1; S1max = 2) = step_fct(S1-S1max)*(S1-S1max)


# ROOT bucket

l_p   = 0.25  # root coeff
log_S2max = 0.0   # maximum storage capacity of the root zone reservoir
Et(S2, Emax, Ei; l_p = 0.25, log_S2max = 0.0) = step_fct(S2)*(Emax(PET; k_v = k_v)-Ei(S1,PET; k_v = k_v))*minimum([1.0, S2/(l_p*exp(log_S2max))])

gamma   = 2.0 # non-linearity
log_k_s = -6  # hydraulic conductivity
R(S; log_S2max=0.0, log_k_s = -6, gamma = 2.0) = step_fct(S)*exp(log_k_s)*(S/exp(log_S2max))^gamma

# Response Function

function swissGW_buckets(p_, t_out)


    function swissGW_buckets_core!(dS,S,ps,t)
        k, T_t, k_v, S1max, l_p, log_S2max, log_k_s, gamma = ps

        P    = itp_P(t)
        T    = itp_T(t)
        PET  = itp_PET(t)

        dS[1] = Ps(P,T;T_t = T_t) - melt(S[1], T; k = k, T_t = T_t)
        dS[2] = Pr(P,T;T_t = T_t) + melt(S[1], T; k = k, T_t = T_t) - Ei(S[2],PET; k_v = k_v) - Pe(S[2]; S1max = S1max)
        dS[3] = Pe(S[2]; S1max = S1max) - Et(S[3], Emax, Ei; l_p = l_p, log_S2max = log_S2max) - R(S[3]; log_S2max=log_S2max, log_k_s = log_k_s, gamma = gamma)

    end

    prob = ODEProblem(swissGW_buckets_core!, p_[1:2], Float64.((t_out[1], maximum(t_out))))

    sol = solve(prob, BS3(), u0 = p_[1:2], p=p_[3:end], saveat=t_out, reltol=1e-3, abstol=1e-3, sensealg= ForwardDiffSensitivity())

    # Marvin's version:
    # Qb_ = Qb.(sol[2,:], p_[3], p_[4], p_[5])
    # Qs_ = Qs.(sol[2,:], p_[4])
    # Qout_ = Qb_.+Qs_

    # Pastas in python:
    # t = np.arange(0,1e4)
    # step = p0 * scipy.special.gammainc(p1, t/p2)
    # block = step[1:] - step[:-1]
    # contrib = np.convolve(recharge, block)

    t = collect(1:1e4)
    step = p0 * gamma_inc(p1, t/p2, IND=0) # IND ∈ [0,1,2] sets accuracy: IND=0 means 14 significant digits accuracy, IND=1 means 6 significant digit, and IND=2 means only 3 digit accuracy.
    block = step[2:end] - step[1:end-1]
    contrib = conv(R(S[3]; log_S2max=log_S2max, log_k_s = log_k_s, gamma = gamma), block)

    GW_head = GW_avg + contrib # GW_avg is a specific number

    # return Qout_, sol
    return GW_head, sol

end
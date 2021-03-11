# using Manifolds, Manopt
import Polylogarithm: polylog

function distance_estimation_fisher(S, Cs, n₂)
    # S and Cs are symmetric semi definite positive matrices (\lambda >0)
    # 
    p = size(S, 1)
    F = S \ Cs  # Is it Symmetric
    if sum(isnan(F)) == 0
        c₂ = p / n₂
        λ = sort(real(eig(F)))
        ζ = sort((eig(diag(λ) - (1 / n₂) * sqrt(λ) * sqrt(λ)')))
        # Replace c1=0 in the expression of out.
        c₁ = 0
        # Define only once (λ ./ λ') and (ζ ./ λ') before use.
        # Define (2 / p) * ((λ ./ λ') .* log(abs(λ ./ λ')) ./ (1 - λ ./ λ')) as a function
        # Use loop to compute the sumation
        ker_vec = (2 / p) * ((λ ./ λ') .* log(abs(λ ./ λ')) ./ (1 - λ ./ λ'))
        ker_vec[isnan(ker_vec)] .= 0
        ker = sum(ker_vec) - 2
        # For loop for double sumation and divide out into several terms.
        out = real(
                (2 / p) * sum(sum(((ζ ./ λ') .* log(abs(ζ ./ λ')) ./ (1 - ζ ./ λ'))))
                - ker
                + (2 / p) * sum(log(λ))
                - (1 - c₂) / c₂
                    * (
                        log(1 - c₂)^2
                        - log(1 - c₁)^2
                        + sum(log(λ).^2 - log(ζ).^2)
                        )
                - 1 / p
                    * (2 * sum(
                                sum(
                                polylog(2, 1 - (ζ * ones(1, p)) ./ (ones(p, 1) * λ'))
                                - polylog(2, 1 - (λ * ones(1, p)) ./ (ones(p, 1) * λ'))
                                )
                            )
                        - sum(log((1 - c₁) * λ).^2)
                        )
                )
        r = sign(out)
        out .^= 2
    else
        out = 1000
    end
    return out, r
end

function estim_log(S, Cs, n₂)
    p = size(S, 1)
    F = S \ Cs
    c₂ = p / n₂
    λ_Ĉ₂ = sort(eig(F))
    out = 1 / p * sum(log(λ_Ĉ₂)) + (1 - c₂) / c₂ * log(1 - c₂) + 1
    r = sign(out)
    out .^= 2
    return out, r
end

function estim_t(S, Cs, n₂)
    F = S \ Cs
    λ_Ĉ₂ = sort(eig(F))
    out = mean(λ_Ĉ₂)
    r = sign(out)
    out .^= 2
    return out, r
end

function estim_log1st(S, Cs, n₂)
    s = 1
    p = size(S, 1)
    F = S \ Cs
    c₂ = p / n₂
    λ = sort(real(eig(F)))
    # define m(z, c₂, λ) outside since common to all estim_xxx functions
    m(z) = c₂ * mean(1 ./ (λ - z)) - (1 - c₂) ./ z
    κ_p = 0
    κ_m = -10
    while abs(κ_p - κ_m) > 1e-7 * abs(λ[p] - λ[1])
        κ_ = (κ_p + κ_m) / 2
        if m(κ_) > 1
            κ_p = κ_
        else
            κ_m = κ_
        end
    end
    κ_0_Ĉ₂ = (κ_p + κ_m) / 2
    out = (1 + s * κ_0_Ĉ₂ + log(abs(-s * κ_0_Ĉ₂))) / c₂
            + 1 / p * sum(log(abs(1 - λ / κ_0_Ĉ₂)))
    r = sign(out)
    out .^= 2
    return out, r
end

function estim_KL(S, Cs, n₂)
    p = size(S, 1)
    F = S \ Cs
    c₂ = p / n₂
    λ_Ĉ₂ = sort(eig(F))
    out = - 0.5 * (1 / p * sum(log(λ_Ĉ₂)) + (1 - c₂) / c₂ * log(1 - c₂) + 1)
            - 0.5
            + 0.5 * mean(λ_Ĉ₂)
    r = sign(out)
    out .^= 2
    return out, r
end

function estim_Battacharrya(S, Cs, n₂)
    p = size(S, 1)
    F = S \ Cs
    c₂ = p / n₂
    λ = sort(real(eig(F)))
    # define m(z, c₂, λ) outside since common to all estim_xxx functions
    m(z) = c₂ * mean(1 ./ (λ - z)) - (1 - c₂) ./ z
    κ_p = 0
    κ_m = -10
    # Put this in a function
    while abs(κ_p - κ_m) > 1e-7 * abs(λ[p] - λ[1])
        κ_ = (κ_p + κ_m) / 2
        if m(κ_) > 1
            κ_p = κ_
        else
            κ_m = κ_
        end
    end
    κ_0_Ĉ₂ = (κ_p + κ_m) / 2
    out = 0.5 * ((1 + s * κ_0_Ĉ₂ + log(abs(-s * κ_0_Ĉ₂))) / c₂
                    + 1 / p * sum(log(abs(1 - λ / κ_0_Ĉ₂)))
                    )
        - 0.25 * (1 / p * sum(log(λ))
                    + (1 - c₂) / c₂ * log(1 - c₂)
                    + 1)
        - 0.5 * log(2)
    r = sign(out)
    out .^= 2
    return out, r
end

function estim_Wasserstein(S, Cs, n₂)
    p = size(S, 1)
    c₂ = p / n₂
    F = S * Cs
    λ = sort(real(eig(F)))
    ζ = sort(eig(diag(λ) - (1 / n₂) * sqrt(λ) * sqrt(λ)'))
    # define m(z, c₂, λ) outside since common to all estim_xxx functions
    m(z) = c₂ * mean(1 ./ (λ - z)) - (1 - c₂) ./ z
    integrand_real(z) = (1 / (π * c₂)) * 2 * sqrt((m(z)))
    out = 0
    for i = 1:length(ζ)
        out += integral(integrand_real, ζ[i], λ[i])
    end
    out = (1 / p) * trace(Cs) + (1 / p) * trace(S) - 2 * out
    r = sign(out)
    out .^= 2

    return out, r
end

function estim_Inverse_Fisher(S, Cs, n₂)
    if sum(isnan(F)) == 0
        p = size(S, 1)
        F = Cs * S
        c₁, c₂ = 0, p/n₂
        λ = sort(real(eig(F)))
        ζ = sort((eig(diag(λ) - (1 / n₂) * sqrt(λ) * sqrt(λ)')))
        ker_vec = (2 / p) * ((λ ./ λ') .* log(abs(λ ./ λ')) ./ (1 - λ ./ λ'))
        ker_vec(isnan(ker_vec)) = 0
        ker = sum(sum((ker_vec))) - 2
        out = real(
            (2 / p) * sum(sum(((ζ ./ λ') .* log(abs(ζ ./ λ')) ./ (1 - ζ ./ λ'))))
            - ker
            + (2 / p) * sum(log(λ))
            - (1 - c₂) / c₂ * (log(1 - c₂)^2
                                - log(1 - c₁)^2
                                + sum(log(λ).^2 - log(ζ).^2))
            - 1/p * (2 * sum(
                            sum(polylog(2, 1 - (ζ * ones(1, p)) ./ (ones(p, 1) * λ'))
                                - polylog(2, 1 - (λ * ones(1, p))  ./(ones(p, 1) * λ'))
                            )
                        )
                    - sum(log((1 - c₁) * λ).^2)
                    )
            )
        r = sign(out)
        out .^= 2
    else
        out = 1000
    end
    return out, r
end

function estim_Inverse_log(S, Cs, n₂)
    p = size(S, 1)
    F = Cs * S
    c₂ = p / n₂
    λ_Ĉ₂ = real(sort(eig(F)))
    out = - 1/p * sum(log(λ_Ĉ₂)) - (1 - c₂) / c₂ * log(1 - c₂) - 1
    r = sign(out)
    out .^= 2
    return out, r
end

function estim_Inverse_log1st(S, Cs, n₂)
    p = size(S, 1)
    s = 1
    F = Cs * S
    c₂ = p / n₂
    λ = sort(real(eig(F)))
    # define m(z, c₂, λ) outside since common to all estim_xxx functions
    m(z) = c₂ * mean(1 ./ (λ - z)) - (1 - c₂) ./ z
    κ = zeros(p + 1, 1)
    λ₁ = [0; λ]
    κ_p = 0
    κ_m = -10
    # Put this in a function
    while abs(κ_p - κ_m) > 1e-7 * abs(λ[p] - λ[1])
        κ_ = (κ_p + κ_m) / 2
        if m(κ_) > 1/s
            κ_p = κ_
        else
            κ_m = κ_
        end
    end
    κ[1] = (κ_p + κ_m) / 2
    for i = 2:length(λ₁)
        κ_p = λ₁[i]
        κ_m = λ₁[i-1]
        while abs(κ_p - κ_m) > 1e-7 * abs(λ[p] - λ[1])
            κ_ = (κ_p + κ_m) / 2
            if m(κ_) > 1/s
                κ_p = κ_
            else
                κ_m = κ_
            end
        end
        κ[i] = (κ_p + κ_m) / 2
    end
    out = ((1 / c₂) * sum(λ - κ[2:end])
            - (1 / p) * sum(log(λ))
            + (1 / p) * sum(log(λ - κ[1]))
            + (1 / c₂ - 1) * sum(log(λ ./ κ[2:end]))
            - 1)
    r = sign(out)
    out .^= 2
    return out, r
end

function estim_Inverse_t(S, Cs, n₂)
    p = size(S, 1)
    F = Cs * S
    c₂ = p / n₂
    λ = sort(real(eig(F)))
    out = 0
    for i = 1:length(λ)
        λc = λ
        λc[i] = []
        out -= 1 / p * (c₂ / p * sum(1 ./ (λ[i] - λc)) - (1 - c₂) ./ λ[i])
    end
    r = sign(out)
    out .^= 2
    return out, r
end

function estim_Inverse_Battacharrya(S, Cs, n₂)
    p = size(S, 1)
    s = 1
    F = Cs * S
    c₂ = p / n₂
    λ = sort(real(eig(F)))
    # define m(z, c₂, λ) outside since common to all estim_xxx functions
    m(z) = c₂ * mean(1 ./ (λ - z)) - (1 - c₂) ./ z
    λ₁ = [0; λ]
    κ_p = 0
    κ_m = -10
    # Put this in a function
    while abs(κ_p - κ_m) > 1e-7 * abs(λ[p] - λ[1])
        κ_ = (κ_p + κ_m) / 2
        if m(κ_) > 1/s
            κ_p = κ_
        else
            κ_m = κ_
        end
    end
    κ = zeros(p+1)
    κ[1] = (κ_p + κ_m) / 2
    for i = 2:length(λ₁)
        κ_p = λ₁[i]
        κ_m = λ₁[i-1]
        while abs(κ_p - κ_m) > 1e-7 * abs(λ[p] - λ[1])
            κ_ = (κ_p + κ_m) / 2
            if m(κ_) > 1/s
                κ_p = κ_
            else
                κ_m = κ_
            end
        end
        κ[i] = (κ_p + κ_m) / 2
    end
    out = 0.5 * (
                    (1 / c₂) * sum(λ - κ[2:end])
                    - (1 / p) * sum(log(λ))
                    + (1 / p) * sum(log(λ - κ[1]))
                    + (1 / c₂ - 1) * sum(log(λ ./ κ[2:end])) - 1
                )
        - 0.25 * (
                    - 1/p * sum(log(λ))
                    - (1 - c₂) / c₂ * log(1 - c₂) - 1
                )
        - 0.5 * log(2)

    r = sign(out)
    out .^= 2

    return out, r
end

function estim_Inverse_KL(S, Cs, n₂)
    p = size(S, 1)
    F = Cs * S
    c₂ = p / n₂
    λ_Ĉ₂ = real(sort(eig(F)))
    out_t = 0
    for i = 1:length(λ_Ĉ₂)
        λc = λ_Ĉ₂
        λc[i] = []
        out_t -= (1 / p) * ((c₂ / p) * sum(1 ./ (λ_Ĉ₂[i] - λc))
                            - (1 - c₂) ./ λ_Ĉ₂[i])
    end
    out = - 0.5 * (
                    - 1 / p * sum(log(λ_Ĉ₂))
                    - (1 - c₂) / c₂ * log(1 - c₂)
                    - 1
                )
            + 0.5 * out_t
            - 0.5

    r = sign(out)
    out .^= 2

    return out, r
end

function out = ARHMM(x, Ns, Np, opts)
% ARHMM  Train a Gaussian autoregressive hidden Markov model by EM.
%
%   out = ARHMM(x, Ns, Np) fits an Ns-state, order-Np autoregressive HMM to
%   multi-trial time series by Expectation-Maximization. Each state s owns a
%   Gaussian AR process
%
%       x_t  =  mu_s  +  sum_{tau=1..Np} B_s(tau) * x_{t-tau}  +  noise,
%       noise ~ N(0, S_s),          z_{t+1} | z_t ~ Phi(z_t, :)
%
%   so the model is a switching-means / switching-AR-dynamics HMM with a
%   static (time-homogeneous) transition matrix Phi.
%
%   Inputs:
%     x    : Nch x Nw x Ntr raw observations (channels x timepoints x trials).
%            A single Nch x Nw matrix (one trial) is also accepted.
%     Ns   : number of hidden states.
%     Np   : autoregressive order (>= 1).
%     opts : optional struct with fields (defaults in []):
%            .Nq          [50]    number of EM iterations.
%            .verbose     [true]  print per-iteration log-likelihood.
%            .pseudocount [0]     Dirichlet-style ridge u added to the
%                                 diagonal of Phi each iteration,
%                                 Phi <- (Phi + u*I)/(1+u), to keep
%                                 self-transitions from collapsing.
%            .seed        [0]     RNG seed for the random mean init.
%            .ridge       [1e-6]  ridge added to S2 (AR regression) and S
%                                 (emission covariance) for conditioning.
%
%   Output struct "out" (original field names preserved):
%     .pr    : Ns x 1 prior over z_1.
%     .Phi   : Ns x Ns transition matrix, Phi(i,j) = P(z_{t+1}=j | z_t=i).
%     .mu    : Nchp x Ns state means (only the top Nch rows are used).
%     .B     : Nchp x Nchp x Ns AR companion matrices (top Nch rows are the
%              AR coefficients mapping x_aug(t-1) -> x(t)).
%     .S     : Nchp x Nchp x Ns state emission covariances.
%     .Omega : Nchp x Nchp x Ns state emission precisions (inv of S blocks).
%     .gamma : Nw x Ns x Ntr smoothed posteriors P(z_t | x_{1:Nw}).
%     .Xi    : (Nw-1) x Ns x Ns x Ntr expected joint state posteriors.
%     .l     : Nq x Ntr per-iteration, per-trial log-likelihood trace.
%     .vi    : Ntr x Nw Viterbi state paths (most likely state sequence).
%     .vmax  : Ntr x Nw Viterbi path log-probabilities.
%     .Nch, .Np, .Nchp : dimensions, for convenience.

    % ---------- defaults ----------
    if nargin < 4, opts = struct(); end
    if ~isfield(opts, 'Nq'),          opts.Nq          = 50;    end
    if ~isfield(opts, 'verbose'),     opts.verbose     = true;  end
    if ~isfield(opts, 'pseudocount'), opts.pseudocount = 0;     end
    if ~isfield(opts, 'seed'),        opts.seed        = 0;     end
    if ~isfield(opts, 'ridge'),       opts.ridge       = 1e-6;  end
    Nq = opts.Nq;
    u  = opts.pseudocount;

    if Np < 1, error('ARHMM: AR order Np must be >= 1'); end

    % ---------- build the companion (lag-embedded) observations ----------
    % X(:,t,n) = [x_t; x_{t-1}; ...; x_{t-Np+1}] so the top Nch rows are the
    % current sample and the predictor x_aug(t-1) holds the Np-lag history.
    if ismatrix(x), x = reshape(x, size(x,1), size(x,2), 1); end
    [Nch, T, Ntr] = size(x);
    Nchp = Nch * Np;
    Nw   = T - Np + 1;                 % usable timepoints after embedding
    if Nw < 2
        error('ARHMM: need at least Np+1 timepoints per trial (got %d)', T);
    end
    X = zeros(Nchp, Nw, Ntr);
    for n = 1:Ntr
        for t = Np:T
            X(:, t-Np+1, n) = reshape(x(:, t:-1:t-Np+1, n), Nchp, 1);
        end
    end

    % ---------- simple data-driven initialization ----------
    rng(opts.seed);
    % per-channel variance (top block) sets the emission covariance scale
    xall  = reshape(x, Nch, []);
    xvar  = var(xall, 0, 2);
    xvar(xvar <= 0 | ~isfinite(xvar)) = 1;

    pr  = ones(Ns, 1) / Ns;            % prior over z_1
    Phi = ones(Ns)   / Ns;            % uniform transitions

    mu    = zeros(Nchp, Ns);
    B     = zeros(Nchp, Nchp, Ns);
    S     = zeros(Nchp, Nchp, Ns);
    Omega = zeros(Nchp, Nchp, Ns);
    for s = 1:Ns
        mu(1:Nch, s) = sqrt(xvar) .* randn(Nch, 1) * 0.1;      % small random means
        if Np > 1                                              % companion shift block
            B(Nch+1:Nchp, :, s) = [eye(Nch*(Np-1)), zeros(Nch*(Np-1), Nch)];
        end
        S(1:Nch, 1:Nch, s)     = diag(xvar);
        Omega(1:Nch, 1:Nch, s) = diag(1 ./ xvar);
    end

    % ---------- allocate posteriors / traces ----------
    gamma = zeros(Nw,   Ns,     Ntr);
    Xi    = zeros(Nw-1, Ns, Ns, Ntr);
    l     = zeros(Nq,   Ntr);
    vi    = zeros(Ntr,  Nw);
    vmax  = zeros(Ntr,  Nw);

    % =================== EM ===================
    for q = 1:Nq
        % keep self-transitions alive (optional Dirichlet-style smoothing)
        if u > 0
            Phi = (Phi + u * eye(Ns)) / (1 + u);
        end

        % ---------- E-step: forward-backward per trial ----------
        for n = 1:Ntr
            b = bAR2(mu, B, Omega, X(:,:,n), Np);
            if q < Nq
                [gamma(:,:,n), Xi(:,:,:,n), l(q,n)] = alphabetavg(pr, Phi, b);
            else
                % request the Viterbi path on the final sweep only
                [gamma(:,:,n), Xi(:,:,:,n), l(q,n), vi(n,:), vmax(n,:)] = ...
                    alphabetavg(pr, Phi, b);
            end
        end

        if opts.verbose
            fprintf('  EM iter %2d/%d   sum log-lik = %.6g\n', q, Nq, sum(l(q,:)));
        end

        % ---------- M-step: emissions ----------
        % Sg(s) = sum over (n,t) of the responsibility gamma_{t,s,n}.
        Sg = sum(sum(gamma, 3), 1);
        mu = zeros(Nchp, Ns);
        S  = zeros(Nchp, Nchp, Ns);
        S2 = zeros(Nchp, Nchp, Ns);

        for s = 1:Ns
            % responsibility-weighted expectations of x_t and x_{t-1}
            mun = zeros(Nch,  1);          % <x_t>_s    (top Nch rows, all t)
            mum = zeros(Nchp, 1);          % <x_{t-1}>_s (full predictor, t>=2)
            for n = 1:Ntr
                for t = 1:Nw
                    mun = mun + gamma(t,s,n) * X(1:Nch, t, n);
                    if t > 1
                        mum = mum + gamma(t,s,n) * X(:, t-1, n);
                    end
                end
            end
            mu(1:Nch, s) = (mun - B(1:Nch, :, s) * mum) / Sg(s);

            % AR coefficient regression: B_top = (sum gamma r x') / (sum gamma x x')
            mi = mu(1:Nch, s);
            Bnum = zeros(Nch, Nchp);
            for n = 1:Ntr
                % t = 1 contributes to the (unconditioned) covariance only
                S(1:Nch,1:Nch,s) = S(1:Nch,1:Nch,s) + ...
                    gamma(1,s,n) * (X(1:Nch,1,n) - mi) * (X(1:Nch,1,n) - mi)';
                for t = 2:Nw
                    S2(:,:,s) = S2(:,:,s) + gamma(t,s,n) * X(:,t-1,n) * X(:,t-1,n)';
                    Bnum = Bnum + gamma(t,s,n) * (X(1:Nch,t,n) - mi) * X(:,t-1,n)';
                end
            end
            S2_reg = S2(:,:,s) + opts.ridge * eye(Nchp);
            if nnz(S2(:,:,s)) == 0
                B(1:Nch, :, s) = zeros(Nch, Nchp);
            else
                B(1:Nch, :, s) = Bnum / S2_reg;
            end
            if Np > 1                      % maintain companion shift block
                B(Nch+1:Nchp, :, s) = [eye(Nch*(Np-1)), zeros(Nch*(Np-1), Nch)];
            end

            % conditional covariance using the (now updated) mu and B
            for n = 1:Ntr
                for t = 2:Nw
                    res = X(1:Nch,t,n) - mi - B(1:Nch,:,s) * X(:,t-1,n);
                    S(1:Nch,1:Nch,s) = S(1:Nch,1:Nch,s) + gamma(t,s,n) * (res * res');
                end
            end
            S(:,:,s) = S(:,:,s) / Sg(s);

            % symmetrize + ridge, then invert to the emission precision Omega
            Sblk = (S(1:Nch,1:Nch,s) + S(1:Nch,1:Nch,s)') / 2 + opts.ridge * eye(Nch);
            S(1:Nch,1:Nch,s)     = Sblk;
            Omega(1:Nch,1:Nch,s) = eye(Nch) / Sblk;
        end

        % ---------- M-step: transitions + initial prior ----------
        for s1 = 1:Ns
            denom = sum(sum(sum(Xi(:,s1,:,:), 1), 3), 4);   % total outflow from s1
            for s2 = 1:Ns
                num = sum(sum(Xi(:,s1,s2,:), 4), 1);        % flow s1 -> s2
                if denom > 0
                    Phi(s1,s2) = num / denom;
                end
            end
            pr(s1) = sum(gamma(1,s1,:), 3) / Ntr;
        end
    end

    out = struct('pr', pr, 'Phi', Phi, 'mu', mu, 'B', B, 'S', S, ...
                 'Omega', Omega, 'gamma', gamma, 'Xi', Xi, 'l', l, ...
                 'vi', vi, 'vmax', vmax, 'Nch', Nch, 'Np', Np, 'Nchp', Nchp);
end

% ======================================================================
function [gamma, xi, L, vi, vmax] = alphabetavg(pr, Phi, b)
% ALPHABETAVG  Forward-backward (alpha-beta) smoothing for an HMM.
%   state transition matrix..Phi: ns x ns   (Phi(i,j) = P(s_{t+1}=j|s_t=i))
%   prior state probability...pr: ns x 1
%   emission likelihood........b: ns x n     (b(j,t) ~ P(y_t | s_t=j))
%   Returns smoothed posteriors gamma (n x ns), expected joints xi
%   (n-1 x ns x ns), log-likelihood L, and -- if requested -- the Viterbi
%   path vi and its log-probabilities vmax. Per-step alpha rescaling keeps
%   the recursion numerically stable.

    vit = (nargout >= 4);

    [ns, n] = size(b);
    xi = zeros(n-1, ns, ns);

    alpha = zeros(n, ns);
    alpha(1, :) = pr;                  % prior on the initial state
    beta  = zeros(n, ns);
    p     = zeros(n, 1);               % per-step normalizers; prod(p) = P(Y)

    % forward (alpha) recursion: P(y_{1..t}, s_t)
    if vit
        viterbi = zeros(n, ns);
        vistate = zeros(n, ns);
        vi   = zeros(n, 1);
        vmax = zeros(n, 1);
        for t = 1:n
            for j = 1:ns
                if t > 1
                    alphai = b(j,t) * Phi(:,j);
                    alpha(t,j) = alpha(t-1,:) * alphai;        % sum over paths into j
                    vaux = viterbi(t-1,:)' + log(alphai);      % all paths into j
                    viterbi(t,j) = max(vaux);                  % best path into j
                    vistate(t,j) = min(find(viterbi(t,j) == vaux)); %#ok<MXFND>
                else
                    alpha(1,j)   = alpha(1,j) * b(j,1);
                    viterbi(1,j) = log(pr(j) * b(j,1));
                end
            end
            p(t) = sum(alpha(t,:));
            alpha(t,:) = alpha(t,:) / p(t);
        end
        % backtrace the most likely state sequence
        vi(n) = find(max(viterbi(n,:)') == viterbi(n,:)'); %#ok<UDIM>
        vmax(n) = viterbi(n, vi(n));
        for t = n:-1:2
            vi(t-1)   = vistate(t, vi(t));
            vmax(t-1) = viterbi(t-1, vi(t-1));
        end
    else
        for t = 1:n
            for j = 1:ns
                if t > 1
                    alpha(t,j) = alpha(t-1,:) * Phi(:,j);      % marginal over prev state
                end
                alpha(t,j) = alpha(t,j) * b(j,t);
            end
            p(t) = sum(alpha(t,:));
            alpha(t,:) = alpha(t,:) / p(t);
        end
    end

    % backward (beta) recursion with joint posteriors xi
    beta(n,:) = alpha(n,:);
    for t = n-1:-1:1
        for j = 1:ns
            for k = 1:ns
                aq = alpha(t,:) * Phi(:,k);
                xi(t,j,k) = alpha(t,j) * beta(t+1,k) * Phi(j,k) / aq;
                beta(t,j) = beta(t,j) + xi(t,j,k);
            end
        end
    end
    gamma = beta;

    L = sum(log(p));                   % log P(Y|model) = sum_t log p(t)
end

% ======================================================================
function b = bAR2(mu, B, Omega, X, p)
% BAR2  State-conditional AR emission likelihood p(x_t | x_{t-1}, s_t).
%   mu: (np*nd) x ns   state means (top nd rows observed)
%   B : (np*nd) x (np*nd) x ns   AR companion matrices
%   Omega: (np*nd) x (np*nd) x ns   emission precisions (top nd block used)
%   X : (np*nd) x n    augmented observations for one trial
%   p : AR order np
%   Returns b: ns x n, the per-state observation likelihood.

    [nd, ns] = size(mu);
    n = size(X, 2);
    b = zeros(ns, n);
    nd = floor(nd / p);                % recover the observed channel count
    for a = 1:ns
        % t = 1: unconditioned Gaussian about the state mean
        x = X(:,1) - mu(:,a);
        b(a,1) = exp(-0.5 * x' * Omega(:,:,a) * x);
        for t = 2:n
            x = X(:,t) - mu(:,a) - B(:,:,a) * X(:,t-1);   % one-step AR residual
            b(a,t) = exp(-0.5 * x' * Omega(:,:,a) * x);
        end
        % Gaussian normalizing constant (2*pi)^(-nd/2) * det(Omega)^(1/2),
        % evaluated on the observed nd x nd precision block.
        b(a,:) = b(a,:) / sqrt((2*pi)^nd) * sqrt(det(Omega(1:nd,1:nd,a)));
    end
end

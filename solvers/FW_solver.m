function [U,it,fx,ttot,fh,timeVec,gnr] = FW_solver(U0, Ls, omega, epsilon,...
    verbosity, maxit, maxtime, gapstop, fstop, stopcr, variant)

% Implementation of the Frank-Wolfe (FW) algorithm for data segmentation. 
% FW solves the following optimization problem
%
%              min_U E(U) s.t. U ∈ Sigma
%
% where the quadratic objective E(U) is defined as
%
%       E(U) = 0.5*trace(U' Ls U) + 1/epsilon*trace(U(II-U')) +
%              0.5*trace((U-U0)'D_ω(U-U0))
%
% and Sigma is the cartesian products of Gibbs simplices.

% Here:
% U = (u_1, . . . , u_n)'∈ R^{n×K} where the kth component of ui ∈ R^K is
%         the strength for data point i to belong to class k.
% U0 = (u0_1, . . . , u0_n)'∈ R^{n×K} where the kth component of u0_i ∈ R^K
%         is the known value of the fidelity node.
% Ls ∈ R^{n×n} is the graph Laplacian 
% D_ω = diag(ω1, ..., ωn) ∈ R^{n×n} is a diagonal matrix with elements ωi 
%         associated to supervised data
% epsilon ∈ R^+ is the penalization parameter 
% II ∈ R^{K×n} is a matrix of all ones

% Two version of the FW algorithm can be called:
% - 'FW' the classic version (see Alg. 4.1 in [1])
% - 'GFW' a greedy version of FW (see Alg. 4.2 in [1])

% Input
% U0:        matrix associated to predefined nodes (the supervised part)
% Ls:        graph Laplacian (either in matrix form either or as a mat-vec
%            function)
% omega:     vector of nonnegative weights for the supervised data
% epsilon:   penalty parameter (>0)
% verbosity: printout level (0 none, >0 one line per iteration)
% maxit:     maximum number of iterations
% maxtime:   maximum elapsed time allowed (in seconds)
% gapstop:   tolerance for the stopping criterion on the optimality gap
% fstop:     tolerance for the stopping criterion on the function value
% stopcr:    1 - stopping criterion on the function value
%            2 - stopping criterion on the optimality gap
% variant:   'FW' classic FW
%            'GFW' Greedy FW

% Output
% U:         solution found
% it:        number of employed iterations
% fx:        objective function value at the solution (E(U))
% ttot:      employed elapsed time
% fh:        vector with obj. function values at every iterations
% timeVec:   vector with elapsed time at every iterations
% gnr:       final optimality gap

% References:
% [1] C. Faccio, M. Porcelli, F. Rinaldi, M. Stoll, 
% "Fast and Simple Multiclass Data Segmentation: An Eigendecomposition and
% Projection-Free Approach", pp. 1-21, 2025, arXiv:2508.09738.

% 
% Authors: C. Faccio, M. Porcelli, F. Rinaldi
% Date: July 2025


%parameter setting
gamma=1e-6;  % Armijo parameter

flagls=0;
if strcmp(variant,'GFW')
    % set tolerance for 0-1 elements
    delta = 1e-3;
end

[n,K] = size(U0);

Om =repmat(omega,1,K);
OmU0 = U0.*Om;
b = 1/epsilon*speye(n,K)-OmU0;

% Initialization of convergence history vectors
if (maxit < Inf)
    fh=zeros(1,maxit);
    timeVec=zeros(1,maxit);
else
    fh=zeros(1,100*n);
    timeVec=zeros(1,100*n);
end

it=1;
% Starting point
U = U0;

tstart = tic;
while (it <= maxit && flagls==0)
    
    % Compute the quadratic term xQx in the objective function 
    if isa(Ls, 'function_handle')
        for i=1:size(U,2)
            LsU(:,i) = Ls(U(:,i));
        end
    else
        LsU = Ls*U;
    end
    OmU = U.*Om;
    Qx = LsU - 2/epsilon*U+OmU;
    xQx = trace(U'*Qx);
    % Compute the linear term bx in the objective function
    bx = 1/epsilon*trace(U'*speye(n,K))-trace(U'*OmU0);
    % Compute the constant term c in the objective function
    c = 0.5*trace(U0'*OmU0);
    
    if (it==1)
        fx = 0.5*xQx + bx + c;
        timeVec(it) = 0;
    else
        timeVec(it) = toc(tstart);
    end
    
    fh(it)=fx;
    
    % Compute the gradient Qx+b
    G = Qx+b;
        
    if (timeVec(it) > maxtime)
        break;
    end
    
    switch variant
        case'FW'
            % compute LMO            
            [~,J] =min(G,[],2);
            S=sparse((1:n)', J, ones(n,1), n, K);

            % equivalent to
            %  S=zeros(n,K);
            %  for i = 1:n
            %     S(i, J(i))=1.0;
            %  end
            
        case 'GFW'
            % compute GLMO  
            mask = ( delta < U & U < 1-delta);
            G_tilde = G.*mask;
            [~,J] =min(G_tilde,[],2);
            ind = (sum(mask,2) ~= 0);
            compl_ind = logical(1-ind);
            vec = (1:n)';
            
            S = sparse(vec(ind), J(ind), ones(length(vec(ind)),1), n, K);
            S(compl_ind, :) = U(compl_ind, :);
    end
    % Compute the direction
    D = S - U; 
    
    % Compute the criticality measure
    gnr = trace(G'*D);
    % stopping criteria and test for termination
    switch stopcr
        case 1
            if (fx <= fstop)
               break;
            end
        case 2
            if (gnr >= -gapstop)
                break;
            end
        otherwise
            error('Unknown stopping criterion');
    end
    
    %Armijo line-search
   

    alpha=1;  %initial stepsize
    % alpha = 2/(it+2);
    btmax = 1000; % maximum number of backtracks
    bt = 0;       % Backtrack counter
    
    ref = gamma*gnr;
    
    % Compute the objective function at U + alpha(S-U)
    if isa(Ls, 'function_handle')        
        for i=1:size(U,2)
            LsS (:,i) = Ls(S(:,i));
        end
        Qs = LsS- 2/epsilon*S+S.*Om;
    else       
        LsS =Ls*S;
        Qs = LsS- 2/epsilon*S+S.*Om;
    end
    sQs = trace(S'*Qs);
    sQx = trace(S'*Qx);
    bs = 1/epsilon*trace(S'*speye(n,K))-trace(S'*OmU0);
    fz =  0.5*((1-alpha)^2*xQx + 2 * alpha*(1-alpha)*sQx + ...
        alpha^2*sQs)+ (1-alpha)*bx + alpha*bs + c;
    
    % Backtrack procedure
    while fz > fx + alpha * ref
        alpha = 0.5*alpha;
        fz =  0.5*((1-alpha)^2*xQx + 2 * alpha*(1-alpha)*sQx + ...
            alpha^2*sQs)+ (1-alpha)*bx + alpha*bs + c;
        if (bt == btmax)
            break 
        end
        bt = bt+1;
        
    end
    % Iteration and objective function update
    U = U + alpha*D;
    fx = fz;
    
    if (verbosity>0)
        fprintf(' it = %d, f(x) =  %4.2e, gap =  %4.2e, alpha = %4.2e, bt = %2.0f \n', it, fx, -gnr, full(alpha),bt)
    end
    it = it+1;
    
end
ttot = toc(tstart);

if (it < size(fh,2))
    fh = fh(1:it);
    timeVec = timeVec(1:it);
end

end

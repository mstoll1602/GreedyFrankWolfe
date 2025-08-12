function [u,it,tol_reached] = MBO_solver(u0, lambda, phi, omega, dt, ...
                                         MAX_ITER,tolit)
% segmentation of an image using the Merriman-Bence-Osher (MBO) method
%
% see: [1] C. Garcia-Cardona, E. Merkurjev, A.L. Bertozzi, A. Flenner, A.
% Percus, "Multiclass Data Segmentation using Diffuse Interface Methods on
% Graphs", IEEE, 2014.
%
% Input:
% u_0:      image with predifend points, the supervised part
% lambda:   k eigenvalues of the Graph Laplacian
% phi:      k eigenvectors of the Graph Laplacian
% omega:    weight in the diffusion update (fidelity)
% dt:       time-step or learning rate
% MAX_ITER: maximum number of updates
% tolit:    tolerance on the relative change in u
%
% Output:
% u:           segmented image
% it:          employed updates
% tol_reached: relative error between two updates
%
% Stoll/Bosch 2016
%
% Created on: 25.05.2016
%     Author: Jessica Bosch
%
%
% Modified: 	Kai Bergermann, 2020
%               Martin Stoll 2023
% 		removed random initialization of u_0. Use u_0 as described in 
%       [2, Section 3.5] instead
%
% see: [2] Kai Bergermann. Diffuse interface methods for node classification
%          on multilayer graphs. 
%          Master thesis, Technische Universität Chemnitz, 2020.
%


%% set parameters
% omega_0=10000;              % fidelity parameter
% epsilon = 1;                % interface parameter
% c = (2/epsilon)+omega_0;    % convexity parameter
% dt = 0.01;                  % time step size
% MAX_ITER = 1000;            % max. number of time steps
% tolit = 1e-5;               % stopping tolerance

lambda = diag(lambda);      % desired eigenvalues
k = size(lambda,1);         % k=number of desired eigenvalues
[n,N] = size(u0);          % n=number of unknowns, N=number of phases

N_s = 3;

MAX_ITER=round(MAX_ITER/N_s);
% Make sure u0 is feasible
u = rand(size(u0));
for j=1:n
    u(j,:)=projspx(u(j,:));
end
%% algorithm according to [p. 1605, 1]

Y = (speye(k)+dt/N_s*diag(lambda))\phi';

%% loop over time
for it = 1:MAX_ITER
    %% pure diffusion steps
    us = u;
    for j=1:N_s
        % fidelity part
        Fid = us-u0;
        Fid = Fid .* repmat(omega,1,N); %mu(U^n-hat U)
        
        % Fid = Domega*Fid;
        Z = Y*(us-(dt/N_s)*Fid);
        us = phi*Z;
        
        for j=1:n
            us(j,:)=projspx(us(j,:));
        end
    end
    % project the solution back to the Gibbs simplex
    u_new = us;
    % nur alle paar Diffusions-Schritte thresholden...
    for j=1:n
        dist_mat=repmat(u_new(j,:),N,1)-eye(N);
        norm_dist_mat = sqrt(sum(dist_mat .* dist_mat, 2));
        [~,ind_thresholding]=min(norm_dist_mat);
        u_new(j,:)=zeros(1,N);
        u_new(j,ind_thresholding)=1;
    end
    % norm for stopping criterion
    norm_diff = norm((u_new-u));
    norm_new = norm(u_new);
    % update old solution
    u=u_new;
    % test stopping criterion
    tol_reached = max(norm_diff)/max(norm_new);
    %fprintf('We are in iteration %0.5g and have reached tolerance %0.5e\n',it,tol_reached)
    if (tol_reached<tolit)
        break;
    end
    
end

end



function projy=projspx(y)

% projection of y onto the Gibbs simplex
% see: Y. Chen and X. Ye, "Projection onto A Simplex", arXiv preprint, 2011

n=length(y);
y_sort=sort(y);
th_set=0;

for i=(n-1):-1:1
    
    ti=0;
    for j=(i+1):n
        ti=ti+y_sort(j);
    end
    
    if ((ti-1)/(n-i)>=y_sort(i))
        th=(ti-1)/(n-i);
        th_set=1;
        break;
    end
    
end

if (th_set < 0.5)
    th=0;
    for j=1:n
        th=th+y(j);
    end
    th=(th-1)/n;
end

projy=y-th;
for j=1:n
    if (projy(j)<0)
        projy(j)=0;
    end
end

end
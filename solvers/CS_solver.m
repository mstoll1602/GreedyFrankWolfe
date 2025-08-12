function [u,it] = CS_solver(u_0, lambda, phi, ...
                      omega, mu, dt, c, MAX_ITER, tolit)
% segmentation of an image using the Allen-Cahn model with a smooth 
% potential and convexity splitting
%
% see: [1] C. Garcia-Cardona, E. Merkurjev, A.L. Bertozzi, A. Flenner, A.
% Percus, "Multiclass Data Segmentation using Diffuse Interface Methods on
% Graphs", IEEE, 2014.
%
% Input:
% u_0:      image with predifend points, the supervised part
% lambda:   k eigenvalues of the Graph Laplacian
% phi:      k eigenvectors of the Graph Laplacian
% omega:    weight in the GL energy function (fidelity)
% mu:       weight in the GL energy function (interface)
% dt:       time-step or learning rate
% c :       convexity splitting parameter (c = (2/mu)+omega)
% MAX_ITER: maximum number of updates
% tolit:    tolerance on the relative change in u
% 
%
% Output:
% u:        segmented image
% it:       employed updates
%
% Stoll/Bosch 2016
%
% Created on: 25.05.2016
%     Author: Jessica Bosch

%% set parameters
% omega_0=10000;                 % fidelity parameter
% mu = 1;                % interface parameter
% c = (2/mu)+omega_0;    % convexity parameter
% dt = 0.01;                  % time step size
% MAX_ITER = 1000;            % max. number of time steps
% tolit = 1e-5;               % stopping tolerance

lambda = diag(lambda);      % desired eigenvalues
k = size(lambda,1);         % k=number of desired eigenvalues
[n,N] = size(u_0);          % n=number of unknowns, N=number of phases

u=rand(size(u_0));
for j=1:n
    u(j,:)=projspx(u(j,:));
end

%% algorithm according to [p. 1605, 1]

Y=zeros(k,n);
for j=1:k
    Y(j,:) = (phi(:,j)/(1 + dt*(mu*lambda(j) + c)))';
end

% loop over time
for it = 1:MAX_ITER
    %     it
    % modified smooth potential (using L1 norm)
    % see [equ. (20), 1]
    T=zeros(n,N);

    for i=1:n
        norm_u=norm(u(i,:),1);
        for j=1:N
            for l=1:N
                tt_norm = 1;
                for m=1:N
                    if (m~=l)
                        tt_norm=tt_norm*0.25*(norm_u-abs(u(i,m))+abs(u(i,m)-1))^(2);
                    end
                end
                T(i,j)=T(i,j)+0.5*(1-2*double(j==l))*(norm_u-abs(u(i,l))+abs(u(i,l)-1))*tt_norm;
            end
        end
    end


    % fidelity part
    Fid=u-u_0;
    for j=1:n
        Fid(j,:)=Fid(j,:)*omega(j);
    end

    Z=Y*((1+c*dt)*u-(dt/(2*mu))*T-dt*Fid);

    u_new=phi*Z;
    % project the solution back to the Gibbs simplex
    for j=1:n    
        u_new(j,:)=projspx(u_new(j,:));
    end

    
    % norm for stopping criterion
    norm_diff=zeros(n,1);
    norm_new=zeros(n,1);
    for j=1:n
        norm_diff(j)=norm((u_new(j,:)-u(j,:))')^(2);
        norm_new(j)=norm((u_new(j,:))')^(2);
    end

    % update old solution
    u=u_new;

    % test stopping criterion
    if (max(norm_diff)/max(norm_new)<tolit)
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
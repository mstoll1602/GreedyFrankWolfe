% Main file to run CS, MBO and GFW method for the segmentation of syntethic
% data sets.

% References:
% [1] C. Faccio, M. Porcelli, F. Rinaldi, M. Stoll,
% "Fast and Simple Multiclass Data Segmentation: An Eigendecomposition and
% Projection-Free Approach", arxive 
%
% Authors: C. Faccio, M. Porcelli, F. Rinaldi, M. Stoll
% Date: July 2025

clear all;
close all;

% Add the relevant paths
% lihi functions are used for subroutines creating sparse
% graph laplacians, since the scaling is a little unsual
addpath(genpath('lihi'))
% Solvers are contained in ./solvers
addpath(genpath('solvers'))
% Auxiliary functions contained in ./auxils
addpath(genpath('auxils'))
% Data sets are contained in ./testsets/synthetic_tests
addpath(genpath('testsets'))
% Results will be displayed in the Command Window

% Set true/false to plot data and results
plots_on = true;

% Set the random seed and the number of repetitions
rng('default')
n_experiments = 1;

% choose the dataset
test_choice = input('Please enter the dataset:\n 1 = four corners,\n 2 = spiral data,\n 3 = diamond data,\n 4 = smile,\n 5 = square 4,\n');
% choose the solver
solver_choice = input('Please enter the solver:\n 1 = CS method,\n 2 = MBO method,\n 3 = GFW method,\n');

if  test_choice == 1
    disp('running FOUR CORNERS')
elseif test_choice == 2
    disp('running SPIRAL DATA')
elseif test_choice == 3
    disp('running DIAMOND DATA')
elseif test_choice == 4
    disp('running SMILE DATA')
elseif test_choice == 5
    disp('running SQUARE 4')
else
    error('ERROR. Wrong Input:'); %ask again
end

tot_time = zeros(n_experiments,1); accuracy = zeros(n_experiments,1); iter = zeros(n_experiments,1);

for test = 1:n_experiments

    close all;

    % Generate the test problem, matrix u0 associated to predefined data
    [u0, Kclass, y_truth, data] = generate_testsets(test_choice, plots_on);

    % Compute the Laplacian of R-neareast neighbour graph
    R = 12;
    [Ls, WW, D12, n] = compute_Laplacian_R_nngraph(R,data);

    if solver_choice == 1 || solver_choice == 2

        %% Set parameters for the learning model of CS and MBO
        [omega_0, mu, Keig, scale] = set_tuned_parameters_CS_MBO(test_choice);

        c = (2/mu)+omega_0;      % parameter for convexity splitting

        % Compute the fidelity matrix for the fidelty term
        omega = zeros(n,1);
        for i=1:n
            if (u0(i,1)>(1/Kclass) || u0(i,1)<(1/Kclass))
                omega(i,1)=omega_0;
            end
        end

        % Set the learning rate and further parameters for CS and MBO
        dt = .1; % time-step or learning rate
        maxit = 100*5; %maximum number of updates
        tol = 1e-4; % tolerance on the relative change in u
        %fprintf('computing svds... \n')
        tic
        [V, D] = compute_TSVD_Laplacian(Keig, D12, WW, scale);
        time_svds = toc;
        %fprintf('done! \n')
    end


    %% Call the CS method
    if solver_choice == 1
        method  = 'CS method';
        np = 1; %plot counter

        tic
        [u1_CS,  it_CS] = CS_solver(u0, D, V,omega,mu,dt,c,maxit,tol);
        time_solver = toc;

        % Evaluation of the model
        u1_segm = zeros(n,Kclass);
        for j=1:n
            [~,max_ind]=max(u1_CS(j,:));
            u1_segm(j,max_ind)=1;
        end

        time_tot = time_solver +  time_svds;
        tot_time(test) = time_tot;
        iter(test) = it_CS;

        % Create confusion matrix and make plots
        CM = create_confusion_matrix_plots_synthetic...
            (test_choice, data, u1_segm, y_truth, np, method, plots_on);

        accuracy(test) = sum(diag(CM))/sum(CM,'all');
    end

    %% Call the MBO method
    if solver_choice == 2
        fprintf('\n')
        method  = 'MBO method';
        np = 2; %plot counter
        tic
        [u1_MBO, it_MBO, ~] = MBO_solver(u0, D, V,omega,dt,maxit,tol);
        time_solver = toc;

        iter(test) = it_MBO;

        % Evaluation of the model
        u1_segm=zeros(n,Kclass);
        for j=1:n
            [~,max_ind]=max(u1_MBO(j,:));
            u1_segm(j,max_ind)=1;
        end

        time_tot = time_solver +  time_svds;
        tot_time(test) = time_tot;
        % Create confusion matrix and make plots
        CM = create_confusion_matrix_plots_synthetic...
            (test_choice, data, u1_segm, y_truth, np, method, plots_on);

        accuracy(test) = sum(diag(CM))/sum(CM,'all');
    end

    %% Call the Greedy FW method
    if solver_choice == 3
        % Set the FW parameters (see ./solvers/FW_solver.m documentation)
        verbosity = 0; %verbosity level
        stopcr = 2; fstop = -1e+6; gapstop = 1e-6;
        maxit = 30; maxtime = 1200;

        FW_variant ='GFW';
        method  = 'Greedy FW method';
        np = 3; %plot counter

        % Parameters for the learning model of GFW
        omega_0 = 1e3;              % fidelity parameter
        epsilon = 5e+1;             % penalization parameter

        mu = epsilon;
        % Compute the fidelity matrix for the fidelty term
        omega = zeros(n,1);
        for i=1:n
            if (u0(i,1)>(1/Kclass) || u0(i,1)<(1/Kclass))
                omega(i,1) = omega_0;
            end
        end

        tic
        [u1_GFW,it_GFW,~,~,~,~] = FW_solver(u0, Ls, omega, epsilon, ...
            verbosity,maxit,maxtime,gapstop,fstop,stopcr, FW_variant);
        time_solver_GFW = toc;

        iter(test) = it_GFW-1;
        % feasibility check
        % sum(u1,2)

        % Evaluation of the model
        u1_segm=zeros(n,Kclass);
        for j=1:n
            [~,max_ind]=max(u1_GFW(j,:));
            u1_segm(j,max_ind)=1;
        end

        time_tot = time_solver_GFW;  time_solver = time_tot;
        tot_time(test) = time_tot; time_svds = [];

        % Create confusion matrix and make plots
        CM = create_confusion_matrix_plots_synthetic...
            (test_choice, data, u1_segm, y_truth, np, method, plots_on);
        accuracy(test) = sum(diag(CM))/sum(CM,'all');
    end
    fprintf('%s - run #%d\n', method, test)
    fprintf('Model parameters: epsilon = %4.0e, omega_0 = %4.0e\n', mu, omega_0)
    fprintf('time svds = %6.3f, time solver = %6.3f, total time = %6.3f \n', ...
        time_svds, time_solver, time_tot')
    fprintf('accuracy percentage = %4.2f%%  \n', sum(diag(CM))/sum(CM,'all')*100)

end

%print average results
fprintf('\n')
fprintf('**** Print average results **** \n\n')
fprintf('Test problem %d: #nodes = %d, #classes = %d, sparsity of Ls = %2.2e\n', ...
    test_choice,  n , Kclass, nnz(Ls)/n^2)
fprintf('%s: total time = %6.3f, #iters = %3.1f, accuracy = %1.2f%% \n',...
    method,  mean(tot_time), mean(iter), mean(accuracy)*100);
fprintf('\n')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Auxiliary functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [u0, Kclass, y_truth, data] = ...
    generate_testsets(test_choice, plots_on)

% Generate the data set (data), the true segmented data (y_truth), the
% number Kclass of classes and the feasible starting point u0.

switch test_choice
    case 1
        %% Four corners
        % TODO add reference
        Kclass = 4; % number of classes
        n = 5000;
        % data = corners(n,5,1.5,4.2);
        data_tmp = corners(n);
        data = data_tmp(:,1:2);
        data = data - repmat(mean(data),size(data,1),1);
        data = data/max(max(abs(data)));
        dotsize = 2;
        ind0=find(data_tmp(:,3)==0);
        ind1=find(data_tmp(:,3)==1);
        ind2=find(data_tmp(:,3)==2);
        ind3=find(data_tmp(:,3)==3);
        y_truth = data_tmp(:,3);

        nsample=6;
        mini=min([length(ind0),length(ind1),length(ind2),length(ind3)]);
        indp=randperm(mini);

        u0=(1/Kclass)*ones(n,Kclass);
        u0(ind0(indp(1:nsample)),:)=0*u0(ind0(indp(1:nsample)),:);
        u0(ind1(indp(1:nsample)),:)=0*u0(ind1(indp(1:nsample)),:);
        u0(ind2(indp(1:nsample)),:)=0*u0(ind2(indp(1:nsample)),:);
        u0(ind3(indp(1:nsample)),:)=0*u0(ind3(indp(1:nsample)),:);
        u0(ind0(indp(1:nsample)),1)=1;
        u0(ind1(indp(1:nsample)),2)=1;
        u0(ind2(indp(1:nsample)),3)=1;
        u0(ind3(indp(1:nsample)),4)=1;

    case 2
        %% Spiral data
        %TODO add reference
        Kclass = 5; % number of classes
        nx = 500;
        [data, label] = generateSpiralDataWithLabels(Kclass ,nx,10,2);
        n = nx*Kclass;
        % 3d data
        data = data - repmat(mean(data),size(data,1),1);
        data = data/max(max(abs(data)));
        dotsize = 2;
        for i=1:Kclass
            ind{i}=find(label==i);
        end
        y_truth = label;

        nsample=5;

        mini = 1e10;
        for i=1:Kclass
            mini=min(mini,length(ind{i}));
        end

        indp=randperm(mini);
        u0=(1/Kclass)*ones(n,Kclass);
        for i=1:Kclass
            indx=ind{i}(indp(1:nsample));
            u0(indx,:)=0*u0(indx,:);
            u0(indx,i)=1;
        end

    case 3
        %% Diamonds data
        % https://github.com/milaan9/Clustering-Datasets/tree/master
        diamond9 = load('diamond9.mat');
        data = diamond9.data;
        data = data - repmat(mean(data),size(data,1),1);
        data = data/max(max(abs(data)));
        label = diamond9.label;

        [n,~] = size(data);

        Kclass=length(unique(label));
        elements = unique(label);
        y_truth = label;

        ind0=find(label==0);
        ind1=find(label==1);
        ind2=find(label==2);
        ind3=find(label==3);
        ind4=find(label==4);
        ind5=find(label==5);
        ind6=find(label==6);
        ind7=find(label==7);
        ind8=find(label==8);

        nsample=4;
        mini=min([length(ind0),length(ind1),length(ind2),length(ind3),length(ind4),length(ind5),length(ind6),length(ind7),length(ind8)]);
        indp=randperm(mini);

        u0=(1/Kclass)*ones(n,Kclass);
        u0(ind0(indp(1:nsample)),:)=0*u0(ind0(indp(1:nsample)),:);
        u0(ind1(indp(1:nsample)),:)=0*u0(ind1(indp(1:nsample)),:);
        u0(ind2(indp(1:nsample)),:)=0*u0(ind2(indp(1:nsample)),:);
        u0(ind3(indp(1:nsample)),:)=0*u0(ind3(indp(1:nsample)),:);
        u0(ind4(indp(1:nsample)),:)=0*u0(ind4(indp(1:nsample)),:);
        u0(ind5(indp(1:nsample)),:)=0*u0(ind5(indp(1:nsample)),:);
        u0(ind6(indp(1:nsample)),:)=0*u0(ind6(indp(1:nsample)),:);
        u0(ind7(indp(1:nsample)),:)=0*u0(ind7(indp(1:nsample)),:);
        u0(ind8(indp(1:nsample)),:)=0*u0(ind8(indp(1:nsample)),:);

        u0(ind0(indp(1:nsample)),1)=1;
        u0(ind1(indp(1:nsample)),2)=1;
        u0(ind2(indp(1:nsample)),3)=1;
        u0(ind3(indp(1:nsample)),4)=1;
        u0(ind4(indp(1:nsample)),5)=1;
        u0(ind5(indp(1:nsample)),6)=1;
        u0(ind6(indp(1:nsample)),7)=1;
        u0(ind7(indp(1:nsample)),8)=1;
        u0(ind8(indp(1:nsample)),9)=1;

        if plots_on
            figure(1000)
            hold on
            plot(data(ind0,1),data(ind0,2),'r.')
            plot(data(ind1,1),data(ind1,2),'b.')
            plot(data(ind2,1),data(ind2,2),'m.')
            plot(data(ind3,1),data(ind3,2),'k.')
            plot(data(ind4,1),data(ind4,2),'g.')
            plot(data(ind5,1),data(ind5,2),'c.')
            plot(data(ind6,1),data(ind6,2),'color', [0.9290 0.6940 0.1250], 'Marker', '.', 'Linestyle', 'none')
            plot(data(ind7,1),data(ind7,2),'color', [0.4940 0.1840 0.5560], 'Marker', '.', 'Linestyle', 'none')
            plot(data(ind8,1),data(ind8,2),'color', [0.4660 0.6740 0.1880], 'Marker', '.', 'Linestyle', 'none')
            title('truth')
        end

    case 4
        %% Smile data
        % https://github.com/milaan9/Clustering-Datasets/tree/master
        smile = load('smile2.mat');
        data = smile.data;
        data = data - repmat(mean(data),size(data,1),1);
        data = data/max(max(abs(data)));
        label = smile.label;
        [n,~] = size(data);

        Kclass=length(unique(label));
        elements = unique(label);
        y_truth = label;

        ind0=find(label==0);
        ind1=find(label==1);
        ind2=find(label==2);
        ind3=find(label==3);

        nsample=10;
        mini=min([length(ind0),length(ind1),length(ind2),length(ind3)]);
        indp=randperm(mini);

        u0=(1/Kclass)*ones(n,Kclass);
        u0(ind0(indp(1:nsample)),:)=0*u0(ind0(indp(1:nsample)),:);
        u0(ind1(indp(1:nsample)),:)=0*u0(ind1(indp(1:nsample)),:);
        u0(ind2(indp(1:nsample)),:)=0*u0(ind2(indp(1:nsample)),:);
        u0(ind3(indp(1:nsample)),:)=0*u0(ind3(indp(1:nsample)),:);

        u0(ind0(indp(1:nsample)),1)=1;
        u0(ind1(indp(1:nsample)),2)=1;
        u0(ind2(indp(1:nsample)),3)=1;
        u0(ind3(indp(1:nsample)),4)=1;

        if plots_on
            figure(1000)
            hold on
            plot(data(ind0,1),data(ind0,2),'r.')
            plot(data(ind1,1),data(ind1,2),'b.')
            plot(data(ind2,1),data(ind2,2),'g.')
            plot(data(ind3,1),data(ind3,2),'k.')
            title('truth')
        end

    case 5
        %% Square 4 data
        % https://github.com/milaan9/Clustering-Datasets/tree/master
        square4 = importdata('square4.arff');
        data_target = square4.data;
        data = data_target(:, 1:2);
        data = data - repmat(mean(data),size(data,1),1);
        data = data/max(max(abs(data)));
        label = data_target(:, 3);
        [n,~] = size(data);

        Kclass=length(unique(label));
        elements = unique(label);
        y_truth = label;

        ind0=find(label==0);
        ind1=find(label==1);
        ind2=find(label==2);
        ind3=find(label==3);

        nsample=4;
        mini=min([length(ind0),length(ind1),length(ind2),length(ind3)]);
        indp=randperm(mini);


        u0=(1/Kclass)*ones(n,Kclass);
        u0(ind0(indp(1:nsample)),:)=0*u0(ind0(indp(1:nsample)),:);
        u0(ind1(indp(1:nsample)),:)=0*u0(ind1(indp(1:nsample)),:);
        u0(ind2(indp(1:nsample)),:)=0*u0(ind2(indp(1:nsample)),:);
        u0(ind3(indp(1:nsample)),:)=0*u0(ind3(indp(1:nsample)),:);

        u0(ind0(indp(1:nsample)),1)=1;
        u0(ind1(indp(1:nsample)),2)=1;
        u0(ind2(indp(1:nsample)),3)=1;
        u0(ind3(indp(1:nsample)),4)=1;

        % plot ground truth
        if plots_on
            figure(100)
            hold on
            plot(data(ind0,1),data(ind0,2),'r.')
            plot(data(ind1,1),data(ind1,2),'b.')
            plot(data(ind2,1),data(ind2,2),'m.')
            plot(data(ind3,1),data(ind3,2),'k.')

            title('truth')
        end

end
end

%%%%%%%%%%
function [Ls, WW, D12, n] = compute_Laplacian_R_nngraph(R,X)

% Function to compute the Laplacian of R-neareast neighbour graph

% Input:   R   Radius of R-neareast neighbour graph
%          X   Data matrix
% Output:  Ls  The symmetrized Laplacian matrix
%          WW  The adjancency matrix
%          D12 The matrix D^{-1/2} for the symmetrized Laplacian
%

% Requires: scale_dist.m from ./lihi

D = dist2(X,X); % the distance matrix
% kernel parameter
n = size(D,1);% matrix dimension
[~,WW,~] = scale_dist(D,R); % create the adjancency matrix
A=WW;A(abs(A)<1e-12)=0;
WW = sparse(A);
DD = sum(WW,2); % the degree matrix
D12 = spdiags(1./sqrt(DD),0,n,n); % the matrix D^{-1/2} for the symmetrized Laplacian
Ls = speye(size(WW,1))-D12*WW*D12;

end

function [V, D] = compute_TSVD_Laplacian(Keig, D12, WW, scale)
% Compute the truncated SVD of the graph Laplacian

% Input:   Keig Number of eigenvectors
%          D12 The matrix D^{-1/2} for the symmetrized Laplacian
%          WW  The adjancency matrix
% Output:  V   2*Keig dominant eigenvectors
%          D   Diagonal matrix of the 2*Keig dominant eigevalues
%          D12 The matrix D^{-1/2} for the symmetrized Laplacian
%


%Keig = 6; % number of eigenvectors
opts.issym = 1; % symmetric
opts.isreal = 1; % real matrix
opts.sigma =  scale;
opts.tol = 1e-6; % tolerance for eigenvalue computations
opts.disp = 0; % display stuff on or off
%tic
[V,ss,V,flag] = svds(D12*WW*D12,2*Keig,'L',opts); % compute 2*K eigenvectors
%time_svds = toc;
V = V(:,1:Keig); % low-rank eigenvector matrix
ss =ss(1:Keig,1:Keig); % diagonal matrix of the relevant eigenvalues of (D12*WW*D12)
D=speye(size(ss,1))-(ss); % eigenvalue matrix of the Laplacian I-D12*WW*D12
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [omega_0, mu, Keig, scale] = set_tuned_parameters_CS_MBO(test_choice)

% Set tuned parameterd for CS and MBO

Keig = 6;

switch test_choice
    case {1,4}
        scale = 2.1;
    case {2,3,5}
        scale = 1.1;
end

switch test_choice
    case 1
        omega_0 = 1e4;
    case {2,5}
        omega_0 = 1e2;
    case {3,4}
        omega_0 = 1e3;
end

switch test_choice
    case {1,3,5}
        mu = 5e2;
    case {2,4}
        mu = 5e-2;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function CM = create_confusion_matrix_plots_synthetic...
    (test_choice, data, u1_segm, y_truth, np, method, plots_on)

% Function to compute the confusion matrix and make plots for
% all the data sets

[n,Kclass] = size(u1_segm);

switch test_choice
    case 1 % Four corners
        CM = zeros(Kclass,Kclass);
        for i = 1:Kclass
            indtruth=find(y_truth==i-1);
            for j=1:Kclass
                CM(i,j)=length(intersect(find(u1_segm(:,j)==1),indtruth));
            end
        end
        %% plot result
        UU1 = 2.*u1_segm-1;
        ind0=find(UU1(:,1)>0);
        ind1=find(UU1(:,2)>0);
        ind2=find(UU1(:,3)>0);
        ind3=find(UU1(:,4)>0);
        e=zeros(n,1);
        e(ind0)=0;
        e(ind1)=1;
        e(ind2)=2;
        e(ind3)=3;

        if plots_on
            figure(np)
            hold on
            plot(data(ind0,1),data(ind0,2),'r.')
            plot(data(ind1,1),data(ind1,2),'b.')
            plot(data(ind2,1),data(ind2,2),'m.')
            plot(data(ind3,1),data(ind3,2),'k.')
            title(method)
        end

    case 2
        %% Spirals
        CM = zeros(Kclass,Kclass);
        for i = 1:Kclass
            indtruth=find(y_truth==i);
            for j=1:Kclass
                CM(i,j)=length(intersect(find(u1_segm(:,j)==1),indtruth));
            end
        end

        UU1 = 2.*u1_segm-1;
        ind0=find(UU1(:,1)>0);
        ind1=find(UU1(:,2)>0);
        ind2=find(UU1(:,3)>0);
        ind3=find(UU1(:,4)>0);
        ind4=find(UU1(:,5)>0);

        e=zeros(n,1);
        e(ind0)=1;
        e(ind1)=2;
        e(ind2)=3;
        e(ind3)=4;
        e(ind4)=5;

        colorList = generateColorList(5);
        colorPlot = colorList(e,:);
        if plots_on
            figure(np); scatter3(data(:,1), data(:,2), data(:,3), 30, colorPlot,'filled');
            title(method)
        end

    case 3
        %% Diamonds
        CM = zeros(Kclass,Kclass);
        for i = 1:Kclass
            indtruth=find(y_truth==i-1);
            for j=1:Kclass
                CM(i,j)=length(intersect(find(u1_segm(:,j)==1),indtruth));
            end
        end
        %% plot result
        UU1 = 2.*u1_segm-1;
        ind0=find(UU1(:,1)>0);
        ind1=find(UU1(:,2)>0);
        ind2=find(UU1(:,3)>0);
        ind3=find(UU1(:,4)>0);
        ind4=find(UU1(:,5)>0);
        ind5=find(UU1(:,6)>0);
        ind6=find(UU1(:,7)>0);
        ind7=find(UU1(:,8)>0);
        ind8=find(UU1(:,9)>0);
        e=zeros(n,1);
        e(ind0)=0;
        e(ind1)=1;
        e(ind2)=2;
        e(ind3)=3;
        e(ind4)=4;
        e(ind5)=5;
        e(ind6)=6;
        e(ind7)=7;
        e(ind8)=8;

        if plots_on

            figure(np);
            hold on
            plot(data(ind0,1),data(ind0,2),'r.')
            plot(data(ind1,1),data(ind1,2),'b.')
            plot(data(ind2,1),data(ind2,2),'m.')
            plot(data(ind3,1),data(ind3,2),'k.')
            plot(data(ind4,1),data(ind4,2),'g.')
            plot(data(ind5,1),data(ind5,2),'c.')
            plot(data(ind6,1),data(ind6,2),'color', [0.9290 0.6940 0.1250], 'Marker', '.', 'Linestyle', 'none')
            plot(data(ind7,1),data(ind7,2),'color', [0.4940 0.1840 0.5560], 'Marker', '.', 'Linestyle', 'none')
            plot(data(ind8,1),data(ind8,2),'color' , [0.4660 0.6740 0.1880], 'Marker', '.', 'Linestyle', 'none')
            title(method)
        end

    case 4
        % Smile
        CM = zeros(Kclass,Kclass);
        for i = 1:Kclass
            indtruth=find(y_truth==i-1);
            for j=1:Kclass
                CM(i,j)=length(intersect(find(u1_segm(:,j)==1),indtruth));
            end
        end
        %% plot result
        UU1 = 2.*u1_segm-1;
        ind0=find(UU1(:,1)>0);
        ind1=find(UU1(:,2)>0);
        ind2=find(UU1(:,3)>0);
        ind3=find(UU1(:,4)>0);

        e=zeros(n,1);
        e(ind0)=0;
        e(ind1)=1;
        e(ind2)=2;
        e(ind3)=3;

        if plots_on
            figure(np);
            hold on
            plot(data(ind0,1),data(ind0,2),'r.')
            plot(data(ind1,1),data(ind1,2),'b.')
            plot(data(ind2,1),data(ind2,2),'g.')
            plot(data(ind3,1),data(ind3,2),'k.')

            title(method)
        end
    case 5
        %% Square 4
        CM = zeros(Kclass,Kclass);
        for i = 1:Kclass
            indtruth=find(y_truth==i-1);
            for j=1:Kclass
                CM(i,j)=length(intersect(find(u1_segm(:,j)==1),indtruth));
            end
        end
        %% plot result
        UU1 = 2.*u1_segm-1;
        ind0=find(UU1(:,1)>0);
        ind1=find(UU1(:,2)>0);
        ind2=find(UU1(:,3)>0);
        ind3=find(UU1(:,4)>0);

        e=zeros(n,1);
        e(ind0)=0;
        e(ind1)=1;
        e(ind2)=2;
        e(ind3)=3;

        if plots_on
            figure(np);
            hold on
            plot(data(ind0,1),data(ind0,2),'r.')
            plot(data(ind1,1),data(ind1,2),'b.')
            plot(data(ind2,1),data(ind2,2),'m.')
            plot(data(ind3,1),data(ind3,2),'k.')

            title(method)
        end

end

end


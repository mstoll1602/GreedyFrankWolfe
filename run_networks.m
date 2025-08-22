% Main file to run CS, MBO and GFW method for the segmentation of real
% networks: LFR benchmark data set, Twitch, LastFM, Facebook, Amazon

% References:
% [1] C. Faccio, M. Porcelli, F. Rinaldi, M. Stoll, 
% "Fast and Simple Multiclass Data Segmentation: An Eigendecomposition and
% Projection-Free Approach", pp. 1-21, 2025, arXiv:2508.09738.
%
% Authors: C. Faccio, M. Porcelli, F. Rinaldi, M. Stoll
% Date: July 2025

clear all;
close all;

% add the relevant paths
% used for subroutines creating sparse graph laplacians,
% since the scaling is a little unsual
%addpath(genpath('lihi'))
% Solvers are contained in ./solvers
addpath(genpath('solvers'))
% Auxiliary functions contained in ./auxils
addpath(genpath('auxils'))
% Data sets are contained in ./testsets/
addpath(genpath('testsets'))
% Results will be saved in ./results
%
% Set the file name for saving the results
fileID = fopen('./results/res_networks_summary.txt','a+');
fileID1 = fopen('./results/res_networks_CS.txt','a+');
fileID2 = fopen('./results/res_networks_MBO.txt','a+');
fileID3 = fopen('./results/res_networks_GFW.txt','a+');

% Set the random seed and the number of repetitions
rng('default')
n_experiments = 2;

% Choose the dataset
choice_usr = input(['Please enter the dataset:\n ', ...
    '1 = LFR Benchmark datasets\,\n ', ...
    '2 = Twitch network\,\n ' ...
    '3 = Last FM network\,\n ' ...
    '4 = Facebook network\,\n ' ...
    '5 = Amazon (computers) network\,\n ' ...
    '6 = Amazon (photos) network\n ']);

for test_choice = choice_usr
    switch test_choice
        case 1
            global nnodes mixcoef
            nnodes = input('number of nodes (1000,5000,10000,50000,100000)');
            mixcoef = input( 'mixing coefficient (0.1,0.2)');
            disp('running LFR')
            fprintf(fileID,'LFR Benchmark datasets (number of node %d, average degree = 5, min community size = 50, mixing coefficient = %2.1f)\n', nnodes, mixcoef);
            fprintf(fileID1,'LFR Benchmark datasets (number of node %d, average degree = 5, min community size = 50, mixing coefficient = %2.1f)\n', nnodes, mixcoef);
            fprintf(fileID2,'LFR Benchmark datasets (number of node %d, average degree = 5, min community size = 50, mixing coefficient = %2.1f)\n', nnodes, mixcoef);
            fprintf(fileID3,'LFR Benchmark datasets (number of node %d, average degree = 5, min community size = 50, mixing coefficient = %2.1f)\n', nnodes, mixcoef);
        case 2
            disp('running Twitch')
            fprintf(fileID,'Twitch network \n');
            fprintf(fileID1,'Twitch network \n');
            fprintf(fileID2,'Twitch network \n');
            fprintf(fileID3,'Twitch network \n');
        case 3
            disp('running LastFM')
            fprintf(fileID,'Last FM network \n');
            fprintf(fileID1,'Last FM network \n');
            fprintf(fileID2,'Last FM network \n');
            fprintf(fileID3,'Last FM network \n');
        case 4
            disp('running Facebook')
            fprintf(fileID,'Facebook network \n');
            fprintf(fileID1,'Facebook network \n');
            fprintf(fileID2,'Facebook network \n');
            fprintf(fileID3,'Facebook network \n');
        case 5
            disp('running Amazon (computers) network')
            fprintf(fileID,'Amazon (computers) network \n');
            fprintf(fileID1,'Amazon (computers) network \n');
            fprintf(fileID2,'Amazon (computers) network \n');
            fprintf(fileID3,'Amazon (computers) network \n');
        case 6
            disp('running Amazon (photos) network')
            fprintf(fileID,'Amazon (photos) network \n');
            fprintf(fileID1,'Amazon (photos) network \n');
            fprintf(fileID2,'Amazon (photos) network \n');
            fprintf(fileID3,'Amazon (photos) network \n');

        otherwise
            error('ERROR. Wrong Input:'); 
    end

    fprintf(fileID1,' test  &  Keig &  omega_0  &   mu        & method   &  time   & %%time svd &  iters     & accuracy \\\\ \n');
    fprintf(fileID2,' test  &  Keig &  omega_0  &             & method   &  time   & %%time svd &  iters     & accuracy \\\\ \n');
    fprintf(fileID3,' test  &  omega_0  &  epsilon    & method   &  time   & iters     & accuracy \\\\ \n');

    tot_time_CS = zeros(n_experiments,1);tot_time_MBO = zeros(n_experiments,1);tot_time_GFW = zeros(n_experiments,1);
    accuracy_CS = zeros(n_experiments,1); accuracy_MBO = zeros(n_experiments,1); accuracy_GFW = zeros(n_experiments,1);
    iter_CS = zeros(n_experiments,1); iter_MBO = zeros(n_experiments,1); iter_GFW = zeros(n_experiments,1);

    for test = 1:n_experiments

        fprintf('')
        fprintf('\nTEST %d \n', test)
        fprintf('')
        close all;

        %  Generate the test problem, matrix u0 associated to predefined
        %  data and corresponding Adjacency matrix WW

        [u0, G, WW, y_truth, elements] = generate_testsets(test_choice);

        % Retrieve the number of nodes and the number of classes
        [n, Kclass] = size(u0);

        % Compute the Laplacian associated to the network
        [Ls, D12] = compute_Laplacian_from_graph(WW);

        fprintf('#nodes = %d, #classes = %d, sparsity of Ls = %2.2e\n',n, Kclass, nnz(Ls)/n^2)
        fprintf(fileID,'#nodes = %d, #classes = %d, sparsity of Ls = %2.2e\n',n, Kclass, nnz(Ls)/n^2);

        %% Set parameters for the learning model of CS and MBO
        [omega_0_CSMBO, mu, Keig, scale] = set_tuned_parameters_CS_MBO(test_choice, n);

        c = (2/mu)+omega_0_CSMBO;      % parameter for convexity splitting

        % Compute the fidelity matrix for the fidelty term
        N = Kclass;
        omega = zeros(n,1);
        for i=1:n
            if (u0(i,1)>(1/N) || u0(i,1)<(1/N))
                omega(i,1)=omega_0_CSMBO;
            end
        end

        % Set the learning rate and further parameters for CS and MBO
        dt = .1; % time-step or learning rate
        maxit = 100*5; %maximum number of updates
        tol = 1e-4; % tolerance on the relative change in u
        fprintf('computing svds... \n')
        tic
        [V, D] = compute_TSVD_Laplacian(Keig, D12, WW, scale);
        time_svds = toc;
        fprintf('done! \n')

        %% Call the CS method
        fprintf('CS METHOD is running ... \n')
        method  = 'CS method';
        np = 1;
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
        tot_time_CS(test) = time_tot;
        iter_CS(test) = it_CS;

        % Create confusion matrix and make plots for LFR
        CM =  create_confusion_matrix_plots_networks(test_choice,G, u1_segm, y_truth, np, elements, method);
        accuracy_CS(test) = sum(diag(CM))/sum(CM,'all');

        fprintf(fileID1,'     %1.0f &  %4.0f &  %4.0e    &  %4.0e      & CS      &   %4.3f  &  %4.1f%%    & %4.2f       & %1.1f%% \\\\ \n',...
            test_choice, Keig, omega_0_CSMBO, mu, tot_time_CS(test), time_svds*100/time_tot, iter_CS(test), accuracy_CS(test)*100);


        %% Call the MBO method
        fprintf('\n')
        fprintf('MBO METHOD is running ... \n')
        method  = 'MBO method';
        np = 2;

        tic
        [u1_MBO, it_MBO, ~] = MBO_solver(u0, D, V,omega,dt,maxit,tol);
        time_solver = toc;

        iter_MBO(test) = it_MBO;

        % Evaluation of the model
        u1_segm=zeros(n,Kclass);
        for j=1:n
            [~,max_ind]=max(u1_MBO(j,:));
            u1_segm(j,max_ind)=1;
        end

        time_tot = time_solver +  time_svds;
        tot_time_MBO(test) = time_tot;

        % Create confusion matrix and make plots for LFR
        CM =  create_confusion_matrix_plots_networks(test_choice, G, u1_segm, y_truth, np, elements, method);
        accuracy_MBO(test) = sum(diag(CM))/sum(CM,'all');

        fprintf(fileID2,'     %1.0f &  %4.0f &  %4.0e    &  %4.0e       & MBO      &   %4.3f &  %4.1f%%   & %4.2f       & %1.1f%% \\\\ \n',...
            test_choice, Keig, omega_0_CSMBO, [], tot_time_MBO(test), time_svds*100/time_tot, iter_MBO(test), accuracy_MBO(test)*100);

        %% Call the Greedy FW method
        % Set the FW parameters (see ./solvers/FW_solver.m documentation)
        stopcr = 2; gapstop = 1e-6; maxit = 30; maxtime = 1200;
        fstop = -1e+6; verbosity = 0; %verbosity level

        % Parameters for the learning model of
        omega_0=1e3;                % fidelity parameter
        epsilon = 5e+1;             % penalization parameter

        % epsilon = 1e-1;           % penalization parameter for OSFW

        % Compute the fidelity matrix for the fidelty term
        N = Kclass;
        omega = zeros(n,1);
        for i=1:n
            if (u0(i,1)>(1/N) || u0(i,1)<(1/N))
                omega(i,1)=omega_0;
            end
        end

        fprintf('\n')
        fprintf('GREEDY FW METHOD is running ... \n')
        FW_variant ='GFW';
        method  = 'Greedy FW method';
        np = 3;

        tic
        [u1_GFW, it_GFW,~,~,~,~] = FW_solver(u0, Ls, omega, epsilon, ...
            verbosity, maxit, maxtime, gapstop, fstop, stopcr, FW_variant);
        time_solver_GFW = toc;

        iter_GFW(test) = it_GFW-1;
        % feasibility check
        % sum(u1,2)

        % Evaluation of the model
        u1_segm=zeros(n,Kclass);
        for j=1:n
            [~,max_ind]=max(u1_GFW(j,:));
            u1_segm(j,max_ind)=1;
        end

        time_tot = time_solver_GFW;  time_solver = time_tot;
        tot_time_GFW(test) = time_tot;

        % Create confusion matrix and make plots for LFR
        CM =  create_confusion_matrix_plots_networks(test_choice, G, u1_segm, y_truth, np, elements, method);
        accuracy_GFW(test) = sum(diag(CM))/sum(CM,'all');

        fprintf(fileID3,'     %1.0f &  %4.0e    &  %4.0e      & GFW      & %4.3f  & %4.2f       & %1.1f%% \\\\ \n', ...
            test_choice, omega_0, epsilon,  tot_time_GFW(test), iter_GFW(test),     accuracy_GFW(test)*100 );

    end

    % print results on files
    fprintf(fileID,' test  &  Keig  &  omega_0  &   mu/eps    & method   &  time  &  iters     & accuracy \\\\ \n');
    fprintf(fileID,'     %1.0f & %4.0f  &  %4.0e    &  %4.0e      & CS       & %4.3f  & %4.2f       & %1.1f%% \\\\ \n',...
        test_choice, Keig, omega_0_CSMBO, mu, mean(tot_time_CS), mean(iter_CS), mean(accuracy_CS)*100);
    fprintf(fileID,'     %1.0f & %4.0f  &  %4.0e    &  %4.0e       & MBO      & %4.3f  & %4.2f       & %1.1f%% \\\\ \n',...
        test_choice, Keig, omega_0_CSMBO, [], mean(tot_time_MBO), mean(iter_MBO), mean(accuracy_MBO)*100);
    fprintf(fileID,'     %1.0f & %4.0f  &  %4.0e    &  %4.0e      & GFW      & %4.3f  & %4.2f       & %1.1f%% \\\\ \n', ...
        test_choice, [],  omega_0, epsilon, mean(tot_time_GFW), mean(iter_GFW), mean(accuracy_GFW)*100);
    fprintf(fileID,'\n')

    %print average results on the Command Window
    fprintf('\n');
    fprintf('**** Print average results **** \n\n');
    fprintf('Test problem %d: #nodes = %d, #classes = %d, sparsity of Ls = %2.2e\n\n', ...
        test_choice,  n , Kclass, nnz(Ls)/n^2);
    fprintf('%s: total time = %6.3f, #iters = %3.1f, accuracy = %1.2f%% \n',...
        'CS  method',  mean(tot_time_CS), mean(iter_CS), mean(accuracy_CS)*100);
    fprintf('%s: total time = %6.3f, #iters = %3.1f, accuracy = %1.2f%% \n',...
        'MBO method',  mean(tot_time_MBO), mean(iter_MBO), mean(accuracy_MBO)*100);
    fprintf('%s: total time = %6.3f, #iters = %3.1f, accuracy = %1.2f%% \n',...
        'GFW method',  mean(tot_time_GFW), mean(iter_GFW), mean(accuracy_GFW)*100);
    fprintf('\n')

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Auxiliary functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [u0, G, WW, y_truth, elements] = ...
    generate_testsets(choice)

WW = []; G = []; elements = [];
switch choice
    case 1
        %% LFR Benchmark datasets
        % https://github.com/GiulioRossetti/cdlib_datasets
        global nnodes mixcoef
        addpath(genpath('Testsets'))
        network_file = strcat('LFR_N',num2str(nnodes),'_ad5_mc50_mu', num2str(mixcoef),'_network_file.csv');
        ground_truth_file = strcat('LFR_N',num2str(nnodes),'_ad5_mc50_mu', num2str(mixcoef),'_ground_truth_file.json');

        Edges = readtable(network_file);

        EdgesM = Edges{:,1:2};
        G = graph(EdgesM(:,1)+1,EdgesM(:,2)+1);
        WW =adjacency(G);
        n = size(WW,1);

        % Load the data
        file_ground_truth = fileread(ground_truth_file);
        data = jsondecode(file_ground_truth);

        Kclass=length(data.communities);

        y_truth = zeros(n,1);
        for ii = 1:Kclass
            indx = data.communities{ii}+1;
            y_truth(indx) = ii;
        end

        for i=1:Kclass
            ind{i}=data.communities{i}+1;
            % permute the entries
            idxperm=randperm(length(ind{i}));
            ind{i} = ind{i}(idxperm);
        end

        nsample=1/3; % fraction of training data that we use for training

        nphase=Kclass;
        u0=(1/nphase)*ones(n,nphase);
        for i=1:Kclass
            indx=ind{i}(1:round(nsample*length(ind{i}))); % percent of training data
            u0(indx,:)=0*u0(indx,:);
            u0(indx,i)=1;
        end

        rgbList = generateColorList(Kclass);

        % plot ground truth
        figure(1000)

        p = plot(G);
        p.NodeCData = y_truth;
        title('ground truth')
        colormap(rgbList)
        colorbar

    case 2
        %% Twitch network
        % https://arxiv.org/abs/2005.07959
        % https://github.com/benedekrozemberczki/datasets

        Edges = readtable('large_twitch_edges.csv');
        Info = readtable('large_twitch_features.csv');

        EdgesM = Edges{:,:};
        G=graph(EdgesM(:,1)+1,EdgesM(:,2)+1);
        WW=adjacency(G);
        n = size(WW,1);


        %% Load the data
        Kclass=length(unique(Info{:,8}));
        elements = unique(Info{:,8});

        label = Info{:,8};

        for i=1:Kclass
            ind{i}=find(strcmp(label,elements{i}));
            % permute the entries
            idxperm=randperm(length(ind{i}));
            ind{i} = ind{i}(idxperm);
        end
        y_truth = label;

        nsample=1/3; % fraction of training data that we use for training

        mini = 1e10;
        for i=1:Kclass
            mini=min(mini,length(ind{i}));
        end
        % indp=randperm(mini);

        nphase=Kclass;
        u0=(1/nphase)*ones(n,nphase);
        for i=1:Kclass
            indx=ind{i}(1:round(nsample*length(ind{i}))); % percent of training data
            u0(indx,:)=0*u0(indx,:);
            u0(indx,i)=1;
        end
    case 3
        % LastFM network
        % https://arxiv.org/abs/2005.07959
        % https://github.com/benedekrozemberczki/FEATHER
        Edges = readtable('lastfm_asia_edges.csv');
        Info = readtable('lastfm_asia_target.csv');
        EdgesM = Edges{:,:};
        G=graph(EdgesM(:,1)+1,EdgesM(:,2)+1);
        WW=adjacency(G);
        n = size(WW,1);

        % since the data are so unbalanced are going to join several
        % classes into one
        elements{1}=[0];
        elements{2}=[1,2,4,7,9,11,12,13,15,16];
        elements{3}=[3];
        elements{4}=[5];
        elements{5}=[6];
        elements{6}=[8];
        elements{7}=[10];
        elements{8}=[14];
        elements{9}=[17];

        Kclass=length(elements);
        label = Info{:,2};

        for i=1:Kclass
            ind{i}=find(ismember(label,elements{i})==1);
            % permute the entries
            idxperm=randperm(length(ind{i}));
            ind{i} = ind{i}(idxperm);
        end

        y_truth = label;
        nsample=.1; % fraction of training data that we use for training

        mini = 1e10;
        for i=1:Kclass
            mini=min(mini,length(ind{i}));
        end

        % indp=randperm(mini);

        nphase=Kclass;
        u0=(1/nphase)*ones(n,nphase);
        for i=1:Kclass
            indx=ind{i}(1:round(nsample*length(ind{i}))); % percent of training data
            u0(indx,:)=0*u0(indx,:);
            u0(indx,i)=1;
        end

    case 4
        %% Facebook network
        % https://github.com/benedekrozemberczki/MUSAE

        Edges = readtable('musae_facebook_edges.csv');
        Info = readtable('musae_facebook_target.csv');
        EdgesM = Edges{:,:};
        G=graph(EdgesM(:,1)+1,EdgesM(:,2)+1);
        WW=adjacency(G);
        n = size(WW,1);

        %% Load the data
        Kclass=length(unique(Info{:,4}));
        elements = unique(Info{:,4});
        label = Info{:,4};

        for i=1:Kclass
            ind{i}=find(strcmp(label,elements{i}));
            % permute the entries
            idxperm=randperm(length(ind{i}));
            ind{i} = ind{i}(idxperm);
        end
        y_truth = label;

        nsample=1/3; % fraction of training data that we use for training

        mini = 1e10;
        for i=1:Kclass
            mini=min(mini,length(ind{i}));
        end
        % indp=randperm(mini);

        nphase=Kclass;
        u0=(1/nphase)*ones(n,nphase);
        for i=1:Kclass
            indx=ind{i}(1:round(nsample*length(ind{i}))); % percent of training data
            u0(indx,:)=0*u0(indx,:);
            u0(indx,i)=1;
        end
    case 5
        %% Amazon network (computers)
        % https://github.com/shchur/gnn-benchmark/tree/master/data/npz

        % Load node data (assuming CSV has NO header)
        nodes = readtable('amazon_computers_nodes.csv', 'ReadVariableNames', false);

        % Extract features (all columns except the last)
        node_features = table2array(nodes(:, 1:end-1));  % 767-dimensional features

        % Extract labels (last column)
        node_labels_computers = table2array(nodes(:, end));        % Labels (0 to 9)

        % Load edges (assuming CSV has NO header)
        edges = readtable('amazon_computers_edges.csv', 'ReadVariableNames', false);

        % Convert to 1-based indexing (MATLAB convention)
        sources = edges.Var1 + 1;
        targets = edges.Var2 + 1;

        % Build the graph
        G_computers = graph(sources, targets);
        WW=adjacency(G_computers);
        n = size(WW,1);

        %% Load the data
        Kclass=length(unique(node_labels_computers))-1;
        elements = unique(node_labels_computers);
        elements = elements(1:end-1);
        label = nodes(2:end, end);

        for i=1:Kclass
            ind{i}=find(table2array(label)==elements(i));
            % permute the entries
            idxperm=randperm(length(ind{i}));
            ind{i} = ind{i}(idxperm);
        end

        y_truth = label;

        nsample=1/3; % fraction of training data that we use for training

        mini = 1e10;
        for i=1:Kclass
            mini=min(mini,length(ind{i}));
        end
        % indp=randperm(mini);

        nphase=Kclass;
        u0=(1/nphase)*ones(n,nphase);
        for i=1:Kclass
            indx=ind{i}(1:round(nsample*length(ind{i}))); % percent of training data
            u0(indx,:)=0*u0(indx,:);
            u0(indx,i)=1;
        end
    case 6
        %% Amazon network (photos)
        % https://github.com/shchur/gnn-benchmark/tree/master/data/npz

        % Load node data (assuming CSV has NO header)
        nodes = readtable('amazon_photo_nodes.csv', 'ReadVariableNames', false);

        % Extract features (all columns except the last)
        node_features = table2array(nodes(:, 1:end-1));  % 767-dimensional features

        % Extract labels (last column)
        node_labels_photos = table2array(nodes(:, end));        % Labels (0 to 9)

        % Load edges (assuming CSV has NO header)
        edges = readtable('amazon_photo_edges.csv', 'ReadVariableNames', false);

        % Convert to 1-based indexing (MATLAB convention)
        sources = edges.Var1 + 1;
        targets = edges.Var2 + 1;

        % Build the graph
        G_computers = graph(sources, targets);
        WW=adjacency(G_computers);
        n = size(WW,1);

        % Load the data
        Kclass=length(unique(node_labels_photos))-1;
        elements = unique(node_labels_photos);
        elements = elements(1:end-1);
        label = nodes(2:end, end);

        for i=1:Kclass
            ind{i}=find(table2array(label)==elements(i));
            % permute the entries
            idxperm=randperm(length(ind{i}));
            ind{i} = ind{i}(idxperm);
        end

        y_truth = label;

        nsample=1/3; % fraction of training data that we use for training

        mini = 1e10;
        for i=1:Kclass
            mini=min(mini,length(ind{i}));
        end
        % indp=randperm(mini);

        nphase=Kclass;
        u0=(1/nphase)*ones(n,nphase);
        for i=1:Kclass
            indx=ind{i}(1:round(nsample*length(ind{i}))); % percent of training data
            u0(indx,:)=0*u0(indx,:);
            u0(indx,i)=1;
        end

end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Ls, D12] = compute_Laplacian_from_graph(G)

% Input:   G   graph
% Output:  Ls  The symmetrized Laplacian matrix
%          WW  The adjancency matrix
%          D12 The matrix D^{-1/2} for the symmetrized Laplacian
%

n = size(G,1);
DD=sum(G,2); % the degree matrix
D12=spdiags(1./sqrt(DD),0,n,n); % the matrix D^{-1/2} for the symmetrized Laplacian
Ls = speye(size(G,1))-D12*G*D12;

end

function [V, D] = compute_TSVD_Laplacian(Keig, D12, WW, scale)
% Input:   Keig Number of eigenvectors
%          D12 The matrix D^{-1/2} for the symmetrized Laplacian
%          WW  The adjancency matrix
% Output:  V   2*Keig dominant eigenvectors
%          D   Diagonal matrix of the 2*Keig dominant eigevalues
%          D12 The matrix D^{-1/2} for the symmetrized Laplacian
%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [omega_0, mu, Keig, scale] = set_tuned_parameters_CS_MBO(choice,n)

% Set tuned parameters for CS and MBO

switch choice
    case 1
        if n<= 10000
            Keig = 100;
        else
            Keig = 200;
        end           
        mu = 1e2;
        omega_0 = 1e2;
    case 2
        Keig = 6;
        mu =1e1;
        omega_0 = 1e2;
    case 3
        Keig = 15;
        mu = 5e1;
        omega_0 = 1e3;
    case 4
        Keig = 100;
        mu = 1e1;
        omega_0 = 1e2;
    case {5,6}
        Keig = 150;
        mu = 5e2;
        omega_0 = 1e2;
end
% kernel parameter for all methods
scale = 1;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [CM] =  create_confusion_matrix_plots_networks...
    (test_choice, G, u1_segm, y_truth, np, elements, method)

% Function to compute the confusion matrix and make plots for the
% LFR data set

[n,Kclass] = size(u1_segm);
switch test_choice
    case 1 % LFR
        rgbList = generateColorList(Kclass);
        y_plot = zeros(n,1);
        for j = 1:n
            y_plot(j)=find(u1_segm(j,:)==1);
        end

        figure(np);

        p4 = plot(G);
        p4.NodeCData = y_plot;
        title(method)
        colormap(rgbList)
        colorbar

        CM = zeros(Kclass,Kclass);
        for i = 1:Kclass
            indtruth=find(y_truth==i);
            for j=1:Kclass
                CM(i,j)=length(intersect(find(u1_segm(:,j)==1),indtruth));
            end
        end

    case 2 %Twitch
        CM = zeros(Kclass,Kclass);
        for i = 1:Kclass
            indtruth=find(strcmp(y_truth,elements{i}));
            for j=1:Kclass
                CM(i,j)=length(intersect(find(u1_segm(:,j)==1),indtruth));
            end
        end

    case 3 %lastFM
        CM = zeros(Kclass,Kclass);
        for i = 1:Kclass
            indtruth=find(ismember(y_truth,elements{i})==1);
            for j=1:Kclass
                CM(i,j)=length(intersect(find(u1_segm(:,j)==1),indtruth));
            end
        end

    case 4 %facebook
        CM = zeros(Kclass,Kclass);
        for i = 1:Kclass
            indtruth=find(strcmp(y_truth,elements{i}));
            for j=1:Kclass
                CM(i,j)=length(intersect(find(u1_segm(:,j)==1),indtruth));
            end
        end
    case {5,6} %amazon
        CM = zeros(Kclass,Kclass);
        for i = 1:Kclass
            indtruth=find(table2array(y_truth)==elements(i));
            for j=1:Kclass
                CM(i,j)=length(intersect(find(u1_segm(:,j)==1),indtruth));
            end
        end

end

end



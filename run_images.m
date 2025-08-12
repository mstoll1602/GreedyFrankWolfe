% Main file to run CS, MBO and GFW method for image labelling.
% Four testing images are supplied: beach, 3 and 4 geometric figures and 4 
% papaers (references below).

% References:
% [1] C. Faccio, M. Porcelli, F. Rinaldi, M. Stoll, 
% "Fast and Simple Multiclass Data Segmentation: An Eigendecomposition and
% Projection-Free Approach", arxive 
%
% Authors: C. Faccio, M. Porcelli, F. Rinaldi, M. Stoll
% Date: July 2025

% Prerequisites:
%   Before using this code, you have to download the NFFT3 toolbox from
%   https://www-user.tu-chemnitz.de/~potts/nfft/ and run the 'configure'
%   script with option --with-matlab=/PATH/TO/MATLAB/HOME/FOLDER.
%   Afterwards run 'make' and 'make check'. When calling this function, the
%   folder %NFFT-MAIN%/matlab/fastsum must be on your MATLAB path.
%
%    [2] D. Alfke, D. Potts, M. Stoll, and T. Volkmer, 
%   "NFFT meets Krylov methods: Fast matrix-vector products for the graph 
%   Laplacian of fully connected networks", Frontiers in Applied Mathematics
%   and Statistics, 4 (2018), p. 61.
%
%    The following MATLAB Toolboxes are required:
%    Deep Learning Toolbox
%    Statistics and Machine Learning Toolbox

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
% Images are contained in ./testsets/images
addpath(genpath('testsets'))
% Results and segmented images will be saved in ./results

%% IMPORTANT
% Add here the path to your installation of nfft, e.g.
% addpath(genpath('C:\Users\username\nfft-3.5.3\'))

% for linux
%addpath(genpath('/home/margherita/Desktop/nfft-3.5.3/matlab') )
% for windows
%addpath(genpath('C:\Users\Margherita\Documents\nfft-3.5.3\'))
% rng('default')

% Set the name of the file for printing the results 
fileID = fopen('./results/image_tmp.txt','a+');

%% Choose and read image data
choice = input(['Please enter the number of the image data:\n ' ...
    '1 = beach,\n ' ...
    '2 = 3 geometric figures,\n ' ...
    '3 = 4 geometric figures,\n ' ...
    '4 = papers\n ']); 

%  Generate the test problem and the matrix u0 associated to predefined
%  pizels and corresponding Adjacency matrix WW
[u0, IM_full, data, coords, e_truth] = generate_testsets_images(choice);

% Retrieve the number of nodes and the number of classes
[n, Kclass] = size(u0);

%% NFFT-opts
% color layer
opts1.sigma=.25;                 %The scaling parameter in the Gaussian kernel.
opts1.sigma=.15; %choice 5
opts1.diagonalEntu1_segmry=0;    %The value on the diagonal of the adjacency matrix, i.e. the weight assigned to all graph loops.
opts1.N=16;                      %The NFFT band width, should be a power of 2.
opts1.m=5;                       %The NFFT window cutoff parameter.
opts1.eps_B=1/16;                %The NFFT regularization length.
opts1.p=2;                       %The NFFT regularization degree.
opts1.N_oversampling=2*opts1.N;  %The NFFT band width for oversampling.

% coordinate layer
opts2.sigma=4;                  %The scaling parameter in the Gaussian kernel.
opts2.diagonalEntry=0;          %The value on the diagonal of the adjacency matrix, i.e. the weight assigned to all graph loops.
opts2.N=16;                     %The NFFT band width, should be a power of 2.
opts2.m=5;                      %The NFFT window cutoff parameter.
opts2.eps_B=1/16;               %The NFFT regularization length.
opts2.p=5;                      %The NFFT regularization degree.
opts2.N_oversampling=2*opts2.N;  %The NFFT band width for oversampling.


%% NFFT-fastsumAdjacencySetup
%tic
S{1} = fastsumAdjacencySetup(data, opts1);
S{2} = fastsumAdjacencySetup(coords, opts2);
Ls = @(x) x - S{1}.applyNormalizedAdjacency(x);
%toc

%%  CS and MBO Methods

% Parameters for the learning model 
omega_0=1e5;          % fidelity parameter
dt = 5;               % time-step or learning rate
mu = 1e-1;            % interface parameter for CS
c = (2/mu)+omega_0;   % parameter for convexity splitting

% Fidelity matrix for the fidelty term
N = Kclass;
omega = zeros(n,1);
for i=1:n
    if (u0(i,1)>(1/N) || u0(i,1)<(1/N))
        omega(i,1)=omega_0;
    end
end

Keig = 6; % number of eigenvectors
if choice == 4; Keig = 20; end

opts.issym = 1;   % symmetric
opts.isreal = 1;  % real matrix
%opts.sigma = .25;
opts.tol = 1e-6;  % tolerance for eigenvalue computations
opts.disp = 0;    % display stuff on or off
if choice == 5, opts = opts1; end
tic
[V,ss, ~] = fastsumAdjacencyEigs(data, Keig, opts);
time_svds = toc;

V = V(:,1:Keig);          % low-rank eigenvector matrix
ss =ss(1:Keig,1:Keig);    % diagonal matrix of the relevant eigenvalues of (D12*WW*D12)
D=speye(size(ss,1))-(ss); % eigenvalue matrix of the Laplacian I-D12*WW*D12

%% Call the CS method
fprintf('CS method \n')
method = 'CS'; np = 1;

tic
u1_CS = CS_solver(u0, D, V,omega,mu,dt,c,500,1e-4);
time_CS = toc;

% Evaluation of the model. %TODO should introduce some more measures for this
% Segmented images are saved in ./results
CM_CS = plot_image_confmatr_results(choice, u1_CS, n, Kclass, IM_full, method, np, e_truth);
accuracy_CS = sum(diag(CM_CS))/sum(CM_CS,'all');

%% Call the MBO method
fprintf('MBO method \n')
method = 'MBO'; np = 2;
tic
u1_MBO = MBO_solver(u0, D, V,omega,dt,50,1e-8);
time_MBO = toc;

% Evaluation of the model.
% Segmented images are saved in ./results
CM_MBO = plot_image_confmatr_results(choice, u1_MBO, n, Kclass, IM_full, method, np, e_truth);
accuracy_MBO = sum(diag(CM_MBO))/sum(CM_MBO,'all');

%% Call the Greedy FW method
fprintf('Greedy FW method \n')
method = 'GFW'; np = 3; 

% Parameters for the learning model 
omega_0=1e5;          % fidelity parameter
epsilon = 1e-1;       % penalization parameter

% Fidelity matrix for the fidelty term
N = Kclass;
omega = zeros(n,1);
for i=1:n
    if (u0(i,1)>(1/N) || u0(i,1)<(1/N))
        omega(i,1)=omega_0;
    end
end

% FW parameters
stopcr = 2;
fstop = 1e-6; verbosity = 0;
gapstop = 1e-04;

tic
[u1_GFW,it_GFW,~,~,~,~] = FW_solver(u0, Ls, omega, epsilon, ...
    verbosity,30,120,gapstop,fstop,stopcr, 'GFW');
time_solver_GFW = toc;

% Evaluation of the model
% Segmented images are saved in ./results
CM_GFW = plot_image_confmatr_results(choice,u1_GFW, n, Kclass, IM_full, method, np, e_truth);
accuracy_GFW = sum(diag(CM_GFW))/sum(CM_GFW,'all');

%% Print the results
fprintf(fileID,'***************************************** \n');
fprintf(fileID,'Image %3.0f \n', choice);
fprintf(fileID,'omega = %6.1e, mu = %6.1e,  Keig = %3.0f, epsilon = %6.1e \n', ...
    omega_0, mu, Keig, epsilon);
fprintf(fileID,'***************************************** \n');
fprintf(fileID,'CS  accuracy = %6.1f%%, time solver = %6.3f, time svds = %6.2f%% \n', ...
    accuracy_CS*100, time_CS+time_svds, time_svds/(time_CS+time_svds)*100);
fprintf(fileID,'MBO accuracy = %6.1f%%, time solver = %6.3f, time svds = %6.2f%% \n', ...
    accuracy_MBO*100, time_MBO+time_svds, time_svds/(time_MBO+time_svds)*100);
fprintf(fileID,'GFW accuracy = %6.1f%%, time solver = %6.3f \n', ...
    accuracy_GFW*100, time_solver_GFW);

%% Print the results on the Command Window
fprintf('***************************************** \n');
fprintf('Image %3.0f \n', choice);
fprintf('omega = %6.1e, mu = %6.1e,  Keig = %3.0f, epsilon = %6.1e \n', ...
    omega_0, mu, Keig, epsilon);
fprintf('***************************************** \n');
fprintf('CS  accuracy = %6.1f%% time solver = %6.2f, time svds = %6.2f%% \n', ...
    accuracy_CS*100, time_CS+time_svds, time_svds/(time_CS+time_svds)*100);
fprintf('MBO accuracy = %6.1f%% time solver = %6.2f, time svds = %6.2f%% \n', ...
    accuracy_MBO*100, time_MBO+time_svds, time_svds/(time_MBO+time_svds)*100);
fprintf('GFW accuracy = %6.1f%% time solver = %6.2f \n', ...
    accuracy_GFW*100, time_solver_GFW);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [u0, IM_full, data, coords, e_truth] = generate_testsets_images(choice)


if  choice == 1
    % beach
    IM=imread('Image1_beach_painted.png');
    IM_full=imread('Image1_beach.png');

    % load the labels of the ground truth
    labels = load('Image1_beach_ground_truth_labels.mat');
    pix_labels = labels.pix_lab;
    nx=size(IM_full,1); ny=size(IM_full,2);
    e_truth = reshape(pix_labels, nx*ny, 1);
elseif choice == 2
    % 3 geometric figures
    % https://github.com/Visillect/colorsegdataset
    IM=imread('Image2_PRZM2CUB_reduced_painted.png');
    IM_full=imread('Image2_PRZM2CUB_reduced.png');

    % load the labels of the ground truth
    labels = load('Image2_PRZM2CUB_ground_truth_labels.mat');
    pix_labels = labels.pix_lab;
    nx=size(IM_full,1); ny=size(IM_full,2);
    e_truth = reshape(pix_labels, nx*ny, 1);

elseif choice == 3
    % 4 geometric figures
    % https://github.com/Visillect/colorsegdataset
    IM=imread('Image3_PYRPRCU_reduced_painted.png');
    IM_full=imread('Image3_PYRPRCU_reduced.png');

    % load the labels of the ground truth
    labels = load('Image3_PYRPRCU_ground_truth_labels.mat');
    pix_labels = labels.pix_lab;
    nx=size(IM_full,1); ny=size(IM_full,2);
    e_truth = reshape(pix_labels, nx*ny, 1);

elseif choice == 4
    % sheets of paper
    % https://github.com/Visillect/colorsegdataset
    IM=imread('Image4_mondrian_8_bit_paper1_reduced_painted.png');
    IM_full=imread('Image4_mondrian_8_bit_paper1_reduced.png');
 
    % load the labels of the ground truth
    labels = load('Image4_mondrian_8_bit_paper1_ground_truth_labels.mat');
    pix_labels = labels.pix_lab;
    nx=size(IM_full,1); ny=size(IM_full,2);
    e_truth = reshape(pix_labels, nx*ny, 1);
end

% get the size
nx=size(IM,1); ny=size(IM,2); n=nx*ny;

%% RGB layer
% vetorize each color channel
data=double([reshape(IM_full(:,:,1),[n,1]),reshape(IM_full(:,:,2),[n,1]),reshape(IM_full(:,:,3),[n,1])]);

% center and scale
data = data - repmat(mean(data),size(data,1),1); %feature-wise centering
data = data/max(max(abs(data))); %layer-wise scaling

%% Coordinate layer
% spatial pixel coordinates
[X,Y]=ndgrid(1:nx, 1:ny);
coords=[reshape(X,[n,1]),reshape(Y,[n,1])];
% center and scale
coords = coords - repmat(mean(coords),size(coords,1),1); %feature-wise centering
coords = coords/max(max(abs(coords))); %layer-wise scaling

if choice == 1 || choice == 2
    
    %% extract painted regions
    % blue
    B_log=(IM(:,:,1)<5 & IM(:,:,2)<5 & IM(:,:,3)>250);
    
    % red
    R_log=(IM(:,:,1)>250 & IM(:,:,2)<5 & IM(:,:,3)<5);
    
    % yellow
    Y_log=(IM(:,:,1)>250 & IM(:,:,2)>250 & IM(:,:,3)<5);
    
    % green
    G_log=(IM(:,:,1)<5 & IM(:,:,2)>123 & IM(:,:,2)<133 & IM(:,:,3)<5);
      
    %% initial conditions
    % assemble initial conditions
    u0=zeros(n,4);
    
    % known labels
    u0(:,1)=reshape(B_log,[n,1]);
    u0(:,2)=reshape(R_log,[n,1]);
    u0(:,3)=reshape(Y_log,[n,1]);
    u0(:,4)=reshape(G_log,[n,1]);
    
    % unknown labels
    for i=1:n
        if(sum(u0(i,:))==0)
            u0(i,:)=[1/4,1/4,1/4,1/4];
        end
    end
    
elseif choice == 3
    %% extract painted regions
    % blue
    B_log=(IM(:,:,1)<5 & IM(:,:,2)<5 & IM(:,:,3)>250);
    
    % red
    R_log=(IM(:,:,1)>250 & IM(:,:,2)<5 & IM(:,:,3)<5);
    
    % yellow
    Y_log=(IM(:,:,1)>250 & IM(:,:,2)>250 & IM(:,:,3)<5);
    
    % green
    G_log=(IM(:,:,1)<5 & IM(:,:,2)>123 & IM(:,:,2)<133 & IM(:,:,3)<5);
    
    % pink
    P_log=(IM(:,:,1)>250 & IM(:,:,2)<5  & IM(:,:,3)>250);

    %% initial conditions
    % assemble initial conditions
    u0=zeros(n,5);
    
    % known labels
    u0(:,1)=reshape(B_log,[n,1]);
    u0(:,2)=reshape(R_log,[n,1]);
    u0(:,3)=reshape(Y_log,[n,1]);
    u0(:,4)=reshape(G_log,[n,1]);
    u0(:,5)=reshape(P_log,[n,1]);
    
    % unknown labels
    for i=1:n
        if(sum(u0(i,:))==0)
            u0(i,:)=[1/5,1/5,1/5,1/5,1/5];
        end
    end
    
elseif choice == 4
    %% extract painted regions
    % blue
    B_log=(IM(:,:,1)<5 & IM(:,:,2)<5 & IM(:,:,3)>250);
    
    % red
    R_log=(IM(:,:,1)>250 & IM(:,:,2)<5 & IM(:,:,3)<5);
    
    % yellow
    Y_log=(IM(:,:,1)>250 & IM(:,:,2)>250 & IM(:,:,3)<5);
    
    % green
    G_log=(IM(:,:,1)<5 & IM(:,:,2)>123 & IM(:,:,2)<133 & IM(:,:,3)<5);
    
    % pink
    P_log=(IM(:,:,1)>250 & IM(:,:,2)<5  & IM(:,:,3)>250);
    
    % cyan
    C_log=(IM(:,:,1)<5 & IM(:,:,2)>250  & IM(:,:,3)>250);
    
    % white
    BB_log=(IM(:,:,1)>250 & IM(:,:,2)>250  & IM(:,:,3)>250);
    
    % light green
    lG_log=(IM(:,:,1)<5 & IM(:,:,2)>250  & IM(:,:,3)<5);
        
    %% initial conditions
    % assemble initial conditions
    u0=zeros(n,8);
    
    % known labels
    u0(:,1)=reshape(B_log,[n,1]);
    u0(:,2)=reshape(R_log,[n,1]);
    u0(:,3)=reshape(Y_log,[n,1]);
    u0(:,4)=reshape(G_log,[n,1]);
    u0(:,5)=reshape(P_log,[n,1]);
    u0(:,6)=reshape(C_log,[n,1]);
    u0(:,7)=reshape(BB_log,[n,1]);
    u0(:,8)=reshape(lG_log,[n,1]);
    
    % unknown labels
    for i=1:n
        if(sum(u0(i,:))==0)
            u0(i,:)=[1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8];
        end
    end
    elseif choice == 5
    u0=zeros(n,Kclass);
    nphase=Kclass;
    select_number = 200; % number of known pixels, please change!
    for i=1:nphase
        idx_selected{i}=randi(length(idx{i}),1,select_number); % randomly select select_number many known pixels for training
        u0(idx{i}(idx_selected{i}),i)=1;
    end
    for i=1:n
        if(sum(u0(i,:))==0)
            u0(i,:)=1/Kclass*ones(1,Kclass);
        end
    end
    utruth=zeros(n,Kclass);
    for i=1:nphase
        utruth(idx{i},i)=1;
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function CM = plot_image_confmatr_results(choice,u1, n, Kclass, IM_full, method, np, e_truth) 

nx=size(IM_full,1); ny=size(IM_full,2);
u1_segm=zeros(n,Kclass);

for j=1:n
    [~,max_ind]=max(u1(j,:));
    u1_segm(j,max_ind)=1;
end

U = u1_segm;

% retrieve solution and plot it
u_sol=zeros(n,Kclass);
[~, I_U] = max(U,[],2);
for i=1:n
    u_sol(i,I_U(i))=1;
end

if choice == 1
    ind0 = find(u_sol(:,1)>0);
    ind1 = find(u_sol(:,2)>0);
    ind2 = find(u_sol(:,3)>0);
    ind3 = find(u_sol(:,4)>0);
    y_sol = zeros(nx*ny,1);
    y_sol(ind0)=1;
    y_sol(ind1)=2;
    y_sol(ind2)=3;
    y_sol(ind3)=4;

    www = repmat(255,1,3);

    f = figure(np);
    %sgtitle(' ') 
    sgtitle(method)
    subplot(221)
    part_1=IM_full;
    u_sol1 = reshape(u_sol(:,1),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol1(ix,iy)
                part_1(ix,iy,:) = www;
            end
        end
    end
    image(part_1);
    title('tree')
    %imwrite(part_1,'tree.png');
    %
    subplot(222)
    part_2=IM_full;
    u_sol2 = reshape(u_sol(:,2),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol2(ix,iy)
                part_2(ix,iy,:) = www;
            end
        end
    end
    image(part_2);
    title('beach')% R
    %imwrite(part_2,'beach.png');
    %
    subplot(223)
    part_3=IM_full;
    u_sol3 = reshape(u_sol(:,3),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol3(ix,iy)
                part_3(ix,iy,:) = www;
            end
        end
    end
    image(part_3);
    title('sea')% Y
    %imwrite(part_3,'sea.png');
    %
    subplot(224)
    part_4=IM_full;
    u_sol4 = reshape(u_sol(:,4),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol4(ix,iy)
                part_4(ix,iy,:) = www;
            end
        end
    end
    image(part_4);
    title('sky')% G
    %imwrite(part_4,'sky.png');
    saveas(f,strcat('./results/Image1_beach_',method),'png');

elseif choice == 2
    ind0 = find(u_sol(:,1)>0);
    ind1 = find(u_sol(:,2)>0);
    ind2 = find(u_sol(:,3)>0);
    ind3 = find(u_sol(:,4)>0);
    y_sol = zeros(nx*ny,1);
    y_sol(ind0)=1;
    y_sol(ind1)=2;
    y_sol(ind2)=3;
    y_sol(ind3)=4;

    www = repmat(255,1,3);

    f = figure(np);
    %sgtitle(' ') 
    sgtitle(method)
    subplot(221)
    part_1=IM_full;
    u_sol1 = reshape(u_sol(:,1),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol1(ix,iy)
                part_1(ix,iy,:) = www;
            end
        end
    end
    image(part_1);
    title('background')% B
    %imwrite(part_1,'tree.png');
    %
    subplot(222)
    part_2=IM_full;
    u_sol2 = reshape(u_sol(:,2),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol2(ix,iy)
                part_2(ix,iy,:) = www;
            end
        end
    end
    image(part_2);
    title('yellow prism')% R
    %imwrite(part_2,'beach.png');
    %
    subplot(223)
    part_3=IM_full;
    u_sol3 = reshape(u_sol(:,3),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol3(ix,iy)
                part_3(ix,iy,:) = www;
            end
        end
    end
    image(part_3);
    title('cube')% Y
    %imwrite(part_3,'sea.png');
    %
    subplot(224)
    part_4=IM_full;
    u_sol4 = reshape(u_sol(:,4),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol4(ix,iy)
                part_4(ix,iy,:) = www;
            end
        end
    end
    image(part_4);
    title('light blue prism')% G
    %imwrite(part_4,'sky.png');
    saveas(f,strcat('./results/Image2_3geofig_',method),'png');
        

elseif choice == 3
    ind0 = find(u_sol(:,1)>0);
    ind1 = find(u_sol(:,2)>0);
    ind2 = find(u_sol(:,3)>0);
    ind3 = find(u_sol(:,4)>0);
    ind4 = find(u_sol(:,5)>0);
    y_sol = zeros(nx*ny,1);
    y_sol(ind0)=1;
    y_sol(ind1)=2;
    y_sol(ind2)=3;
    y_sol(ind3)=4;
    y_sol(ind4)=5;

    www = repmat(255,1,3);

    f = figure(np);
    %sgtitle(' ') 
    sgtitle(method)
    subplot(231)
    part_1=IM_full;
    u_sol1 = reshape(u_sol(:,1),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol1(ix,iy)
                part_1(ix,iy,:) = www;
            end
        end
    end
    image(part_1);
    title('background')% B
    %imwrite(part_1,'tree.png');
    %
    subplot(232)
    part_2=IM_full;
    u_sol2 = reshape(u_sol(:,2),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol2(ix,iy)
                part_2(ix,iy,:) = www;
            end
        end
    end
    image(part_2);
    title('yellow prism')% R
    %imwrite(part_2,'beach.png');
    %
    subplot(233)
    part_3=IM_full;
    u_sol3 = reshape(u_sol(:,3),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol3(ix,iy)
                part_3(ix,iy,:) = www;
            end
        end
    end
    image(part_3);
    title('cube')% Y
    %imwrite(part_3,'sea.png');
    %
    subplot(234)
    part_4=IM_full;
    u_sol4 = reshape(u_sol(:,4),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol4(ix,iy)
                part_4(ix,iy,:) = www;
            end
        end
    end
    image(part_4);
    title('light blue prism')% G
    %imwrite(part_4,'sky.png');
    %
    subplot(235)
    part_5=IM_full;
    u_sol5 = reshape(u_sol(:,5),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol5(ix,iy)
                part_5(ix,iy,:) = www;
            end
        end
    end
    image(part_5);
    title('red pyramid')% P
    %imwrite(part_4,'sky.png');
    saveas(f,strcat('./results/Image3_4geofig_', method),'png');

elseif choice == 4
     ind0 = find(u_sol(:,1)>0);
    ind1 = find(u_sol(:,2)>0);
    ind2 = find(u_sol(:,3)>0);
    ind3 = find(u_sol(:,4)>0);
    ind4 = find(u_sol(:,5)>0);
    ind5 = find(u_sol(:,6)>0);
    ind6 = find(u_sol(:,7)>0);
    ind7 = find(u_sol(:,8)>0);
    y_sol = zeros(nx*ny,1);
    y_sol(ind0)=1;
    y_sol(ind1)=2;
    y_sol(ind2)=3;
    y_sol(ind3)=4;
    y_sol(ind4)=5;
    y_sol(ind5)=6;
    y_sol(ind6)=7;
    y_sol(ind7)=8;
    www = repmat(255,1,3);

    f = figure(np);
    %sgtitle(' ') 
    sgtitle(method)
    subplot(331)
    part_1=IM_full;
    u_sol1 = reshape(u_sol(:,3),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol1(ix,iy)
                part_1(ix,iy,:) = www;
            end
        end
    end
    image(part_1);
    title('background')
    %imwrite(part_1,'tree.png');
    %
    subplot(332)
    part_2=IM_full;
    u_sol2 = reshape(u_sol(:,4),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol2(ix,iy)
                part_2(ix,iy,:) = www;
            end
        end
    end
    image(part_2);
    title('red paper')% R
    %imwrite(part_2,'beach.png');
    %
    subplot(333)
    part_3=IM_full;
    u_sol3 = reshape(u_sol(:,2),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol3(ix,iy)
                part_3(ix,iy,:) = www;
            end
        end
    end
    image(part_3);
    title('white paper')% Y
    %imwrite(part_3,'sea.png');
    %
    subplot(334)
    part_4=IM_full;
    u_sol4 = reshape(u_sol(:,5),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol4(ix,iy)
                part_4(ix,iy,:) = www;
            end
        end
    end
    image(part_4);
    title('yellow paper')% G
    %imwrite(part_4,'sky.png');
    %
    subplot(335)
    part_5=IM_full;
    u_sol5 = reshape(u_sol(:,1),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol5(ix,iy)
                part_5(ix,iy,:) = www;
            end
        end
    end
    image(part_5);
    title('pink paper')% P
    %imwrite(part_4,'sky.png');
    %
    subplot(336)
    part_6=IM_full;
    u_sol6 = reshape(u_sol(:,6),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol6(ix,iy)
                part_6(ix,iy,:) = www;
            end
        end
    end
    image(part_6);
    title('green paper')
    %imwrite(part_1,'tree.png');
    %
    subplot(337)
    part_7=IM_full;
    u_sol7 = reshape(u_sol(:,7),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol7(ix,iy)
                part_7(ix,iy,:) = www;
            end
        end
    end
    image(part_7);
    title('blue paper')
    %imwrite(part_1,'tree.png');
    %
    subplot(338)
    part_8=IM_full;
    u_sol8 = reshape(u_sol(:,8),nx,ny);
    for ix=1:nx
        for iy=1:ny
            if ~u_sol8(ix,iy)
                part_8(ix,iy,:) = www;
            end
        end
    end
    image(part_8);
    title('orange paper')
    %imwrite(part_1,'tree.png');
    %
    saveas(f,strcat('./results/Image4_papers_',method),'png');
end

CM = confusionmat(e_truth, y_sol);

% plot confusion matrix for MBO
f200 = figure(np+200);
if choice == 1
        confusionchart(CM, [["tree"];["beach"];["sea"];["sky"]], ...
        'Title',strcat('Image',string(choice), ...
        ': confusion matrix - ', method));
elseif choice ==2
    confusionchart(CM, [["background"];["yellow prism"];["cube"];...
        ["light blue prism"]], 'Title',strcat('Image',string(choice), ...
        ': confusion matrix - ', method));
elseif choice == 3
    confusionchart(CM, [["background"];["yellow prism"];["cube"];...
        ["light blue prism"]; ["red pyramid"]], 'Title',strcat('Image',string(choice), ...
        ': confusion matrix - ', method));
elseif choice == 4
    confusionchart(CM, [["background"];["red paper"];["white paper"];...
        ["yellow paper"]; ["pink paper"]; ['green paper']; ['blue paper'];['orange paper']], 'Title',strcat('Image',string(choice), ...
        ': confusion matrix - ', method));
end
saveas(f200,strcat('./results/Image',string(choice), '_confusion_matrix_', method),'png');
end


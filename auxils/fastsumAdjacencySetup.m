function S = fastsumAdjacencySetup(data, opts)
% FASTSUMADJACENCYSETUP Setup data for fast adjacency multiplication.
%
% Description:
%   This function sets up the fastsum algorithm from the NFFT3 toolbox to
%   enable fast multiplication with the (normalized or non-normalized)
%   adjacency matrix of the fully connected graph of a given dataset. The
%   adjacency weights are given by W_ij = exp(-||v_i - v_j||^2 / sigma^2), 
%   where v_i and v_j are the d-dimensional feature vectors of data points
%   i and j, and sigma is a scaling parameter. By default, the graph
%   contains no loops. Using NFFT, products with the full adjacency matrix
%   can be approximated in O(n) time without storing any matrix entries.
%
% Reference:
%   D. Alfke, D. Potts, M. Stoll, T. Volkmer - NFFT meets Krylov methods: 
%       Fast matrix-vector products for the graph Laplacian of fully 
%       connected networks (2018). 
%       Preprint available at https://arxiv.org/abs/1808.04580
%
% Prerequisites:
%   Before using this function, you have to download the NFFT3 toolbox from
%   https://www-user.tu-chemnitz.de/~potts/nfft/ and run the 'configure'
%   script with option --with-matlab=/PATH/TO/MATLAB/HOME/FOLDER.
%   Afterwards run 'make' and 'make check'. When calling this function, the
%   folder %NFFT-MAIN%/matlab/fastsum must be on your MATLAB path.
%
% Syntax:
%   S = fastsumAdjacencySetup(data)
%   S = fastsumAdjacencySetup(data, opts)
%
% Inputs:
%   data - Matrix with n rows and d columns, where n is the number of data
%       points and d is the data point feature dimension. Each row holds
%       the feature vector of a data point.
%   opts - Struct holding the algorithm options and parameters in its
%       fields. If a field is not present or holds [], the default value is
%       used for that option. If the opts input is omitted, default values
%       are used for all options.
%
% Options [default]:
%   opts.sigma [1] - The scaling parameter in the Gaussian kernel.
%   opts.diagonalEntry [0] - The value on the diagonal of the adjacency
%       matrix, i.e. the weight assigned to all graph loops.
%   opts.N [64] - The NFFT band width, should be a power of 2.
%   opts.m [5] - The NFFT window cutoff parameter.
%   opts.eps_B [1/16] - The NFFT regularization length.
%   opts.p [5] - The NFFT regularization degree.
%   opts.N_oversampling [2*opts.N] - The NFFT band width for oversampling.
% 
% Outputs:
%   S - Struct with the following fields:
%   S.n - Number of data points.
%   S.FS - Struct holding the fastsum plan, cf. the NFFT3 Matlab
%       documentation.
%   S.Dinvsqrt_vec - Vector holding the inverted square roots of the vertex
%       degrees, i.e. the diagonal of D^(-1/2) where D is the degree
%       matrix.
%   S.applyAdjacency - Function handle. For n-dimensional column vectors x,
%       S.applyAdjacency(x) returns an approximation to W*x, where W is the
%       weighted adjacency matrix of the dataset.
%   S.applyNormalizedAdjacency - Function handle. For n-dimensional column
%       vectors x, S.applyNormalizedAdjacency(x) returns an approximation
%       to D^(-1/2)*W*D^(-1/2)*x, where W is the weighted adjacency matrix
%       of the dataset and D is the corresponding diagonal degree matrix.
%
% This software is distributed under the GNU General Public License v2. See 
% COPYING for the full license text. If that file is not available, see 
% <http://www.gnu.org/licenses/>.
%
% Authors:
%   Dominik Alfke, TU Chemnitz, dominik.alfke(at)mathematik.tu-chemnitz.de
%   Toni Volkmer, TU Chemnitz, toni.volkmer(at)mathematik.tu-chemnitz.de
    if nargin == 0
        error('Too few input arguments for fastsumAdjacencySetup.');
    elseif nargin == 1
        opts = struct();
    end
    
    % Sigma of gaussian kernel exp(- |x-y|^2 / sigma^2)
    if ~isfield(opts, 'sigma') || isempty(opts.sigma)
        opts.sigma = 1;
    end
    
    % Number on the diagonal of W
    if ~isfield(opts, 'diagonalEntry') || isempty(opts.diagonalEntry)
        opts.diagonalEntry = 0;
    end
    
    % Fastsum parameters
    if ~isfield(opts, 'N') || isempty(opts.N)
        opts.N = 64;
    end
    if ~isfield(opts, 'p') || isempty(opts.p)
        opts.p = 5;
    end
    if ~isfield(opts, 'eps_B') || isempty(opts.eps_B)
        opts.eps_B = 1/16;
    end
    if ~isfield(opts, 'm') || isempty(opts.m)
        opts.m = 5;
    end
    if ~isfield(opts, 'N_oversampling') || isempty(opts.N_oversampling)
        opts.N_oversampling = 2 * opts.N;
    end
    
    [n, d] = size(data);
    
    %% Shift & scale points
    midPoint = mean([min(data); max(data)]);
    data = data - midPoint;
    
    maxSquaredNorm = max(sum(data.^2, 2));
    rho = (0.2499 - 0.5*opts.eps_B) / sqrt(maxSquaredNorm);
    data = rho * data;
    opts.sigma = rho * opts.sigma;
    % 
    %% Setup fastsum object
    
    FS = fastsum(...
        d, ...                  % data point dimension
        'laplacian_rbf', ...         % kernel type
        opts.sigma, ...         % c (kernel parameter)
        0, ...                  % flags (could be EXACT_NEARFIELD or NEARFIELD_BOXES)
        opts.N, ...             % N (expansion degree)
        opts.p, ...             % p (degree of smoothness of regularization)
        1/16, ...                  % eps_I (inner boundary)
        opts.eps_B, ...         % eps_B (outer boundary)
        opts.N_oversampling, ...% oversampling parameter for NFFT
        opts.m);                % m (cut-off parameter for NFFT)
    
    FS.x = data;
    FS.y = data;
    
    %% Compute inverse square root of degree matrix 
    FS.alpha = ones(n, 1);
    fastsum_trafo(FS);
    dd = real(FS.f) + (opts.diagonalEntry-1)*FS.alpha;
    Dinvsqrt_vec = 1./sqrt(dd);
    
    %% Setup struct and multiplication function handles
    S.n = n;
    S.FS = FS;
    S.Dinvsqrt_vec = Dinvsqrt_vec;
    S.applyAdjacency = @(x) applyAdjacency(x, FS, opts.diagonalEntry);
    S.applyNormalizedAdjacency = @(x) applyNormalizedAdjacency(x, FS, opts.diagonalEntry, Dinvsqrt_vec);
end

function y = applyAdjacency(x, FS, diagonalEntry)
    FS.alpha = x;
    fastsum_trafo(FS);
    y = real(FS.f) + (diagonalEntry-1)*x;
end

function y = applyNormalizedAdjacency(x, FS, diagonalEntry, Dinvsqrt_vec)
    FS.alpha = Dinvsqrt_vec .* x;
    fastsum_trafo(FS);
    y = Dinvsqrt_vec .* (real(FS.f) + (diagonalEntry-1)*FS.alpha);
end


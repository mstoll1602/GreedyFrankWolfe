function [V, D, S] = fastsumAdjacencyEigs(data, nev, opts)
% FASTSUMADJACENCYEIGS Compute eigenvalues of a graph adjacency matrix.
%
% Description:
%   This function uses the fastsum algorithm from the NFFT3 toolbox and
%   Matlab's eigs function to compute the largest eigenvalues of a 
%   (normalized or non-normalized) adjacency matrix of the fully connected
%   graph of a given dataset. The adjacency weights are given by 
%   W_ij = exp(-||v_i - v_j||), where v_i and v_j are the d-dimensional
%   feature vectors of data points i and j, and sigma is a scaling
%   parameter. By default, the graph contains no loops. Note that the
%   largest eigenvalues of the adjacency matrix correspond to the smallest
%   eigenvalues of the graph Laplacian operator, whose eigenvectors play an
%   important role in data science. Using NFFT, the required matrix-vector
%   products are significantly accelerated without setting up or storing
%   the full matrix, thus making this function applicable even for large
%   datasets.
%
% Reference:
%   D. Alfke, D. Potts, M. Stoll, T. Volkmer - NFFT meets Krylov methods: 
%       Fast matrix-vector products for the graph Laplacian of fully 
%       connected networks (2018). [to be submitted]
%
% Prerequisites:
%   Before using this function, you have to download the NFFT3 toolbox from
%   https://www-user.tu-chemnitz.de/~potts/nfft/ and run the 'configure'
%   script with option --with-matlab=/PATH/TO/MATLAB/HOME/FOLDER.
%   Afterwards run 'make' and 'make check'. When calling this function, the
%   folder %NFFT-MAIN%/matlab/fastsum must be on your MATLAB path.
%
% Syntax:
%   [lambda] = fastsumAdjacencyEigs(data, nev)
%   [lambda] = fastsumAdjacencyEigs(data, nev, opts)
%   [V, D] = fastsumAdjacencyEigs(...)
%   [V, D, S] = fastsumAdjacencyEigs(...)
%
% Inputs:
%   data - Matrix with n rows and d columns, where n is the number of data
%       points and d is the data point feature dimension. Each row holds
%       the feature vector of a data point.
%   nev - The number of eigenpairs to be computed.
%   opts - Struct holding the algorithm options and parameters in its
%       fields. If a field is not present or holds [], the default value is
%       used for that option. If the opts input is omitted, default values
%       are used for all options.
%
% Options [default]:
%   opts.doNormalize [1] - If zero, the eigenvalues of the non-normalized
%       adjacency matrix are computed, else the normalized adjacency matrix
%       is used.
%   opts.sigma [1] - The scaling parameter in the Gaussian kernel.
%   opts.diagonalEntry [0] - The value on the diagonal of the adjacency
%       matrix, i.e. the weight assigned to all graph loops.
%   opts.N [64] - The NFFT band width, should be a power of 2.
%   opts.m [5] - The NFFT window cutoff parameter.
%   opts.eps_B [1/16] - The NFFT regularization length.
%   opts.p [5] - The NFFT regularization degree.
%   opts.N_oversampling [2*opts.N] - The NFFT band width for oversampling.
%   opts.eigs_tol [1e-14] - Tolerance for eigs function.
% 
% Outputs:
%   lambda - Column vector holding the computed eigenvalues.
%   V - Matrix with n rows and nev columns holding the computed
%       eigenvectors in its columns.
%   D - Diagonal matrix holding the computed eigenvalues.
%   S - Struct with output of FASTSUMADJACENCYSETUP.
%
% This software is distributed under the GNU General Public License v2. See 
% COPYING for the full license text. If that file is not available, see 
% <http://www.gnu.org/licenses/>.
%
% Authors:
%   Dominik Alfke, TU Chemnitz, dominik.alfke(at)mathematik.tu-chemnitz.de
%   Toni Volkmer, TU Chemnitz, toni.volkmer(at)mathematik.tu-chemnitz.de

    if nargin < 2
        error('Too few input arguments for fastsumAdjacencySetup.');
    elseif nargin == 2
        opts = struct();
    end
    
    if ~isfield(opts, 'doNormalize') || isempty(opts.doNormalize)
        opts.doNormalize = 1;
    end
    

    S = fastsumAdjacencySetup(data, opts);
    
    eigsOpts.issym = 1;
    eigsOpts.isreal = 1;
    eigsOpts.disp = 0;

    if isfield(opts, 'eigs_tol')
      eigsOpts.tol = opts.eigs_tol;
    end

    if opts.doNormalize
        f = S.applyNormalizedAdjacency;
    else
        f = S.applyAdjacency;
    end
    
    [V, D] = eigs(f, S.n, nev, 'LR', eigsOpts);
    
    if nargout == 1
        V = diag(D);
    end
end


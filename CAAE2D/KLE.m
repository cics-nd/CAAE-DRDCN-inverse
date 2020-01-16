clear all
% generate one set of lhs for now. If needed (revision later),
% we can generate more sets. (similar to random split)

kle_num = 3200;
n_data = [40000];
% n_data = 1;

% doe = 'lhs';
doe = 'mc';

seeds = n_data;
if strcmp(doe, 'mc')
    seeds = n_data + 10131;
end

% 1: abs
% 2: exp
% 3: rbf
kernel = 2;
% unit size of domain
sx = 20; sy = 10;

% resolution
% global ngx ngy 
ngx = 80; ngy = 40;

n_grids = ngx * ngy;

% number of KL terms that are preserved
% kle_num = 100;
% percentage to be preserved
kle_percentage = 1.0;
% set the kle terms by kle_percentage ('0') or kle_num ('1')
trunc_with_num = 0;

% Correlation lengths
ls_x = 4;
ls_y = 2;

MeanY0 = 0.0;
MeanY1 = 4.0;
VarY = 0.5;

% Correlation function
C = nan(n_grids, n_grids);

x = linspace(0, sx, ngx);
y = linspace(0, sy, ngy);
[X, Y] = meshgrid(x, y);
grids = [X(:), Y(:)];

tic
% too low... 
if kernel == 1
    % abs
    for i = 1 : n_grids
        for j = 1 : n_grids
            C(i,j) = VarY*exp(-(abs((grids(i,1)-grids(j,1))/ls_x)...
                +abs((grids(i,2)-grids(j,2))/ls_y)));
        end
    end
    
elseif kernel == 2
    % exp
    for i = 1 : n_grids
        for j = 1 : n_grids
            C(i,j) = VarY * exp(- sqrt(...
                ((grids(i, 1) - grids(j, 1)) / ls_x)^2 + ...
                ((grids(i, 2) - grids(j, 2)) / ls_y)^2));
        end
    end
    
elseif kernel == 3
    % rbf
    for i = 1 : n_grids
        for j = 1 : n_grids
            C(i,j) = VarY * exp(- (...
                ((grids(i, 1) - grids(j, 1)) / ls_x)^2 + ...
                ((grids(i, 2) - grids(j, 2)) / ls_y)^2));
        end
    end
end
toc        
% Calculate eigenvalues and eigenvectors

if trunc_with_num
    % eig_vecs: M*N x kle_num
    % eig_vals: kle_num x kle_num
    [eig_vecs, eig_vals] = eigs(C, kle_num);
    ratio = cumsum(diag(eig_vals)) / (n_grids * VarY);
    kle_percentage = ratio(end);
    fprintf('truncated to %d terms preserving %.5f energy, ls = %.2f\n', ...
        kle_num, kle_percentage, min(ls_x, ls_y))
       
else
    tic
    [eig_vecs, eig_vals] = eig(C);
    toc
    size(C)
    fprintf('done eig\n')
    eig_vals = eig_vals(end : -1 : 1, end : -1 : 1);
    eig_vecs = eig_vecs(:, end : -1 : 1);
    
    if kle_percentage == 1.0
        fprintf('direct sampling\n')
    else
        ratio = cumsum(diag(eig_vals)) / sum(diag(eig_vals));
        kle_num = find(ratio > kle_percentage, 1);
        eig_vals = eig_vals(1:kle_num, 1:kle_num);
        eig_vecs = eig_vecs(:, 1:kle_num);

        figure
        plot(ratio)
        grid on
        fprintf('truncated to preserve %.5f energy with %d terms, ls = %.2f\n', ...
            kle_percentage, kle_num, min(ls_x, ls_y))
    end
end
toc

for i=1:length(seeds)

    % KLE coefficients
    if strcmp(doe, 'lhs')
        % LHS design
        disp(['lhs for ', num2str(n_data(i)), ' data'])
        rng(seeds(i))
        xi = -1 + 2 * lhsdesign(n_data(i), kle_num);
        kle_terms0 = sqrt(2) * erfinv(xi);
        rng(seeds(i)+100)
        xi = -1 + 2 * lhsdesign(n_data(i), kle_num);
        kle_terms1 = sqrt(2) * erfinv(xi);
    elseif strcmp(doe, 'mc')
        disp(['MC for ', num2str(n_data(i)), ' data'])
        % MC sampling
        rng(seeds(i))
        kle_terms0 = randn(n_data(i), kle_num);
        rng(seeds(i)+100)
        kle_terms1 = randn(n_data(i), kle_num);
    end
    
    binaryK = load('K.mat');
    binaryK = binaryK.K;
   
    log_K0 = MeanY0 + eig_vecs * sqrt(eig_vals) * kle_terms0';
    K0 = exp(log_K0);
    log_K1 = MeanY1 + eig_vecs * sqrt(eig_vals) * kle_terms1';
    K1 = exp(log_K1);
    % save the input into separate files
    for n=1:n_data(i) 
        cond = nan(ngy,ngx);
        cond_temp0 = reshape(K0(:, n),ngy,ngx);
        cond_temp1 = reshape(K1(:, n),ngy,ngx);
        binary = squeeze(binaryK(n,:,:));
        nonzero = find(binary>0);
        zero = find(binary == 0);
        cond(zero) = cond_temp0(zero);
        cond(nonzero) = cond_temp1(nonzero);
        save(['input/cond', num2str(n), '.mat'],'cond');
    end

end    


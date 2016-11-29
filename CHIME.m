%% Parameter estimation and clustering for a two-class Gaussian mixture via the EM
% It is based on the EM algorithm, and iteratively estimates the mixing
% ratio omega, component means mu_1, mu_2 and beta=inv(Sigma)*(mu_1 - mu_2).

%% Outputs:
% omega, mu, beta: estimated parameters for the Gaussian mixtures
% RI, aRI: rand index and adjusted rand index when comparing the estimated
%    class index with the true index as vectors. 
% optRI, optaRI: when RI and aRI are vectors, the optimal values for RI and
%    aRI are also returned. 
% group_member: vector of class membership
% Inputs: 
% z: N by p data matrix
% zt: Nt by p training tata
% TRUE_INDEX: true labels of the test data, used for evaluating the
%   clustering performance. If unknown, set a random index, but ignore the
%   output aRI, RI. Note if TRUE_INDEX is not available, one can input a
%   vector consisting of all ones, but the output RI, aRI should be ignored.
% omega0: initialization for \omega
% mu0: p x 2, initialization of [\mu_1, \mu_2]
% beta0: p x 1, initialization of beta, 
% rho: a vector used as constant multiplier for the penalty parameter
% lambda: a scalar, the penalty parameter for estimating sparse beta. Default
%    is 0.1
% maxIter: maximum number of iterations, default is 50.
% tol: tolerance level of stability of the final estimates, default is 1e-06.
%

function [omega, mu, beta, RI, aRI, optRI, optaRI, group_member] = CHIME(z, zt, TRUE_INDEX, omega0, mu0, beta0, rho, lambda, maxIter, tol)

if (nargin < 8), lambda = 0.1;  end
if (nargin < 9), maxIter = 50;  end
if (nargin < 10), tol = 1e-06;  end

[N,p] = size(z);
Nt = size(zt,1);

nrho = length(rho);
aRI = zeros(nrho,1);
RI = zeros(nrho,1);
omega = zeros(nrho,1);
mu = zeros(p,2,nrho);
beta = zeros(p,nrho);
IDX = zeros(Nt, nrho);
for loop_rho = 1:nrho
    lam_c = lambda + rho(loop_rho) * sqrt(log(p)/N);
    
    old_omega = omega0;
    old_mu = mu0;
    old_beta = beta0; 
    
    iter = 1;
    diff = 100;
    
    done = (diff < tol) | (iter >= maxIter);
    while (~done)
        % E-step: calculate gamma
        gamma = old_omega./((1-old_omega)*exp((z - ones(N,1)*mean(old_mu, 2)')*old_beta) + old_omega);
         
        % M-step: update omega,mu
        new_omega = mean(gamma);
        tmp1 = mean(diag(1-gamma) * z)'/(1-new_omega); 
        tmp2 = mean(diag(gamma) * z)'/new_omega;
        new_mu = [tmp1, tmp2];
        
        % Update the empirical covariance matrix Vn
        x = bsxfun(@times, sqrt(1-gamma), z-(tmp1*ones(1,N))');
        y = bsxfun(@times, sqrt(gamma), z-(tmp2*ones(1,N))');
        Vn = 1/N* (x')* x + 1/N*(y')*y;
        while cond(Vn) > 1e+6
            Vn = Vn + sqrt(log(p)/N)*diag(ones(1,p)); 
        end
        
        % M-step: update beta
        delta = tmp1 - tmp2;
        beta_init = Vn \ delta;
        % The tuning parameter in clime is updated for every iteration.
        new_beta = clime(beta_init, Vn, delta, lam_c);
        
        lam_c = 0.7 * lam_c + rho(loop_rho) * sqrt(log(p)/N);
        
        % Calculate the difference between the new value and the old value
        diff = norm(new_beta - old_beta) + norm(new_mu - old_mu) + abs(new_omega - old_omega);
             
        old_omega = new_omega;
        old_mu = new_mu;
        old_beta = new_beta;        

        iter = iter + 1;
        done = (diff < tol) | (iter >= maxIter);       
    end
    % Save the estimate
    omega(loop_rho) = new_omega;
    mu(:,:,loop_rho) = new_mu;
    beta(:,loop_rho) = new_beta;
    
    % Clustering on the test data
    IDX(:,loop_rho) = ((zt - ones( Nt,1)*mean(mu(:,:,loop_rho), 2)')*beta(:,loop_rho)>=log( omega(loop_rho)/(1-omega(loop_rho) + 1e-06) ) ) + 1;
    [aRIl,RIl,~,~] = RandIndex(IDX(:,loop_rho),TRUE_INDEX);
    aRI(loop_rho) = aRIl;
    RI(loop_rho) = RIl;
end

% optimal clustering is selected as the one that maximizes aRI
target = aRI(1);
target_index = 1;
for loop_rho = 1:nrho
    if target < aRI(loop_rho)
        target = aRI(loop_rho);
        target_index = loop_rho;
    end
end
group_member = IDX(:,target_index);
optRI = RI(target_index);
optaRI = aRI(target_index);
beta = beta(:,target_index);
mu = mu(:,:,target_index);
omega = omega(target_index);

end

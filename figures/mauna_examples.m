%% PRNG

seed=3;   % fixing the seed of the random generators
randn('state',seed); %#ok<RAND>
rand('state',seed); %#ok<RAND>

%% Other setup

fig_count = 1;

%% Load data

load('03-mauna2003.mat');

%% Fit long lengthscale SE

mean_fn = @meanZero;
cov_fn = {@covSum, {@covSEiso, @covNoise}};
lik_fn = @likDelta;
hyp.cov = [0, 0, 0];
hyp.mean = [];
hyp.lik = [];

hyp = minimize(hyp, @gp, -500, @infDelta, ...
               mean_fn, cov_fn, lik_fn, X, y);

K = feval(cov_fn{:}, hyp.cov, X);
K_star = feval(cov_fn{:}, hyp.cov, X, [X;Xtest]);
K_starstar = feval(cov_fn{:}, hyp.cov, [X;Xtest], [X;Xtest]);

mu = K_star' / K * y;
post_K = K_starstar - K_star' / K * K_star;

figure(fig_count);
fig_count = fig_count + 1;

samples_density_plot(X, y, [X;Xtest], mu, post_K);

drawnow;
save2pdf([ 'mauna-plots/' 'SE-long' '.pdf'], gcf, 600, true);

%% Fit long lengthscale SE

mean_fn = @meanZero;
cov_fn = {@covSum, {@covSEiso, @covNoise}};
lik_fn = @likDelta;
hyp.cov = [-2, 0, 0];
hyp.mean = [];
hyp.lik = [];

hyp = minimize(hyp, @gp, -500, @infDelta, ...
               mean_fn, cov_fn, lik_fn, X, y);

K = feval(cov_fn{:}, hyp.cov, X);
K_star = feval(cov_fn{:}, hyp.cov, X, [X;Xtest]);
K_starstar = feval(cov_fn{:}, hyp.cov, [X;Xtest], [X;Xtest]);

mu = K_star' / K * y;
post_K = K_starstar - K_star' / K * K_star;

figure(fig_count);
fig_count = fig_count + 1;

samples_density_plot(X, y, [X;Xtest], mu, post_K);
ylim([-30,40]);

drawnow;
save2pdf([ 'mauna-plots/' 'SE-short' '.pdf'], gcf, 600, true);

%% Fit SE + SE

mean_fn = @meanZero;
cov_fn = {@covSum, {@covSEiso, @covSEiso, @covNoise}};
lik_fn = @likDelta;
hyp.cov = [-2, 0, 0, 0, 0];
hyp.mean = [];
hyp.lik = [];

hyp = minimize(hyp, @gp, -500, @infDelta, ...
               mean_fn, cov_fn, lik_fn, X, y);

K = feval(cov_fn{:}, hyp.cov, X);
K_star = feval(cov_fn{:}, hyp.cov, X, [X;Xtest]);
K_starstar = feval(cov_fn{:}, hyp.cov, [X;Xtest], [X;Xtest]);

mu = K_star' / K * y;
post_K = K_starstar - K_star' / K * K_star;

figure(fig_count);
fig_count = fig_count + 1;

samples_density_plot(X, y, [X;Xtest], mu, post_K);

drawnow;
save2pdf([ 'mauna-plots/' 'SE-SE' '.pdf'], gcf, 600, true);

%% Fit SE + SE * Per + SE + SE

mean_fn = @meanZero;
cov_fn = {@covSum, {@covSEiso, @covSEiso @covSEiso, ...
                    @covPeriodicNoDC, ...
                    @covNoise}};
lik_fn = @likDelta;
hyp.cov = [0, 0, 3.8, 3.7, -3, 0, 0, 0, 0, 0];
hyp.mean = [];
hyp.lik = [];

hyp = minimize(hyp, @gp, -500, @infDelta, ...
               mean_fn, cov_fn, lik_fn, X, y);

K = feval(cov_fn{:}, hyp.cov, X);
K_star = feval(cov_fn{:}, hyp.cov, X, [X;Xtest]);
K_starstar = feval(cov_fn{:}, hyp.cov, [X;Xtest], [X;Xtest]);

mu = K_star' / K * y;
post_K = K_starstar - K_star' / K * K_star;

figure(fig_count);
fig_count = fig_count + 1;

samples_density_plot(X, y, [X;Xtest], mu, post_K);

drawnow;
save2pdf([ 'mauna-plots/' 'Complex' '.pdf'], gcf, 600, true);

%% Close all

close all;